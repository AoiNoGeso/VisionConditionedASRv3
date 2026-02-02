import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file
import librosa
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 自作モジュールのインポート
from model import VisionConditionedASRv3
from dataloader import create_dataloader

@dataclass
class AttentionAnalysisConfig:
    """アテンション分析設定"""
    # モデルチェックポイント
    checkpoint_dir: str = "../../Models/VisionConditionedASRv3/white/epoch_15"
    
    # データセットパス
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定
    vocab_size: Optional[int] = None
    visual_tokens: int = 64
    dropout: float = 0.1
    use_visual_pos_embed: bool = True
    
    # データローダー設定
    batch_size: int = 1  # アテンション分析は1サンプルずつ
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = False
    
    # 分析条件
    noise_types: List[str] = field(default_factory=lambda: ["white", "pink", "babble"])
    snr_dbs: List[float] = field(default_factory=lambda: [-10.0, 0.0, 10.0])
    babble_path: str = "../../Datasets/NOISEX92/babble/signal.wav"
    
    # 分析設定
    num_samples: int = 100  # 分析するサンプル数
    save_attention_maps: bool = False  # 個別マップ保存（容量注意）
    
    # デバイス
    device: str = "cuda:1"
    
    # 結果保存
    results_dir: str = "results/attention_analysis/white"
    
    def __post_init__(self):
        print(f"\n{'='*60}")
        print("Attention Analysis Configuration")
        print(f"{'='*60}")
        print(f"Visual Tokens: {self.visual_tokens}")
        print(f"Results directory: {self.results_dir}")
        print(f"Noise Types: {self.noise_types}")
        print(f"SNR Levels: {self.snr_dbs}")
        print(f"Num Samples: {self.num_samples}")
        print(f"{'='*60}\n")


class NoiseAugmenter:
    """ノイズ付加クラス"""
    def __init__(
        self,
        noise_type: str = "none",
        snr_db: float = 10.0,
        babble_path: Optional[str] = None,
        sample_rate: int = 16000
    ):
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.babble_path = babble_path
        self.sample_rate = sample_rate
        self.babble_audio = None
        
        if noise_type == "babble":
            if not babble_path or not os.path.exists(babble_path):
                raise ValueError(f"Babble noise file not found: {babble_path}")
            self.babble_audio, sr = librosa.load(babble_path, sr=self.sample_rate, mono=True)
            self.babble_audio = self.babble_audio.astype(np.float32)
    
    def _compute_rms(self, audio: np.ndarray) -> float:
        return np.sqrt(np.mean(audio ** 2))
    
    def _adjust_noise_level(self, signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        signal_rms = self._compute_rms(signal)
        noise_rms = self._compute_rms(noise)
        
        if noise_rms == 0:
            return noise
        
        snr_linear = 10 ** (snr_db / 10.0)
        target_noise_rms = signal_rms / np.sqrt(snr_linear)
        noise_scaled = noise * (target_noise_rms / noise_rms)
        
        return noise_scaled
    
    def _generate_white_noise(self, length: int) -> np.ndarray:
        return np.random.randn(length).astype(np.float32)
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        white = np.random.randn(length)
        fft = np.fft.rfft(white)
        freqs = np.arange(len(fft))
        freqs[0] = 1
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=length)
        pink = pink / np.max(np.abs(pink))
        return pink.astype(np.float32)
    
    def _get_babble_noise(self, target_length: int) -> np.ndarray:
        if self.babble_audio is None:
            raise ValueError("Babble audio not loaded")
        
        noise = self.babble_audio
        
        if len(noise) < target_length:
            num_repeats = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, num_repeats)[:target_length]
        else:
            start_idx = np.random.randint(0, len(noise) - target_length + 1)
            noise = noise[start_idx:start_idx + target_length]
        
        return noise.astype(np.float32)
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_type == "none":
            return audio
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        length = len(audio)
        
        if self.noise_type == "white":
            noise = self._generate_white_noise(length)
        elif self.noise_type == "pink":
            noise = self._generate_pink_noise(length)
        elif self.noise_type == "babble":
            noise = self._get_babble_noise(length)
        else:
            return audio
        
        noise_adjusted = self._adjust_noise_level(audio, noise, self.snr_db)
        noisy_audio = audio + noise_adjusted
        
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        return noisy_audio


class AttentionExtractor:
    """
    Transformer Encoderからアテンション重みを抽出するクラス
    """
    def __init__(self, model: VisionConditionedASRv3, visual_tokens: int = 64):
        self.model = model
        self.visual_tokens = visual_tokens
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self):
        """各層にフックを登録"""
        self.attention_maps = {}
        self.hooks = []
        
        # Transformer Encoderの各層にフックを登録
        for i, layer in enumerate(self.model.encoder.layers):
            hook = layer.self_attn.register_forward_hook(
                self._make_attention_hook(i)
            )
            self.hooks.append(hook)
    
    def _make_attention_hook(self, layer_idx: int):
        """アテンション重みを保存するフック関数を生成"""
        def hook(module, input, output):
            # outputはtupleで、output[1]がアテンション重み
            # shape: (batch_size, num_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights = output[1]
            else:
                # アテンション重みが直接返される場合
                attention_weights = output
            
            if attention_weights is not None:
                self.attention_maps[f'layer_{layer_idx}'] = attention_weights.detach().cpu()
        
        return hook
    
    def remove_hooks(self):
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_attention(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        1バッチのアテンション重みを抽出
        
        Returns:
            各層のアテンションマップを含む辞書
        """
        self.attention_maps = {}
        
        with torch.no_grad():
            _ = self.model(batch)
        
        return self.attention_maps.copy()
    
    def compute_visual_attention_ratio(
        self, 
        attention_map: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """
        視覚トークンへのアテンション比率を計算
        
        Args:
            attention_map: shape (batch, num_heads, seq_len, seq_len)
        
        Returns:
            - 全体の視覚アテンション比率（スカラー）
            - 時間ステップごとの視覚アテンション比率（配列）
        """
        if attention_map.dim() == 4:
            # (batch, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
            # 平均を取る
            attention_map = attention_map.mean(dim=(0, 1))
        
        # 視覚トークン部分へのアテンション: [:, :visual_tokens]
        visual_attn = attention_map[:, :self.visual_tokens].sum(dim=-1)  # (seq_len,)
        total_attn = attention_map.sum(dim=-1)  # (seq_len,)
        
        # 比率計算
        ratio_per_step = (visual_attn / (total_attn + 1e-10)).numpy()
        overall_ratio = ratio_per_step.mean()
        
        return float(overall_ratio), ratio_per_step


def load_model_weights(model: nn.Module, checkpoint_dir: str, device: torch.device):
    """チェックポイントから重みをロード"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\n{'='*60}")
    print("Loading Checkpoint")
    print(f"{'='*60}")
    print(f"From: {checkpoint_dir}")
    
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    
    print(f"Epoch: {epoch_num}")
    print(f"{'='*60}\n")
    
    return int(epoch_num)


def analyze_attention(
    model: VisionConditionedASRv3,
    dataloader: DataLoader,
    noise_augmenter: NoiseAugmenter,
    config: AttentionAnalysisConfig,
    noise_type: str,
    snr: Optional[float]
) -> Dict[str, Any]:
    """
    特定のノイズ条件下でアテンションを分析
    """
    model.eval()
    extractor = AttentionExtractor(model, config.visual_tokens)
    extractor.register_hooks()
    
    num_layers = len(model.encoder.layers)
    
    # 結果格納用
    results = {
        'samples': [],
        'layer_stats': {
            f'layer_{i}': {
                'visual_attention_ratios': [],
                'visual_attention_per_step': []
            } for i in range(num_layers)
        }
    }
    
    print(f"\n[Analysis] Noise: {noise_type}, SNR: {snr} dB")
    
    sample_count = 0
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Analyzing", total=min(config.num_samples, len(dataloader)))):
            if sample_count >= config.num_samples:
                break
            
            # ノイズ付加
            if noise_augmenter.noise_type != "none":
                batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]
            
            # アテンション抽出
            attention_maps = extractor.extract_attention(batch)
            
            sample_data = {
                'reference': batch["text"][0],
                'layers': {}
            }
            
            # 各層のアテンションを分析
            for layer_name, attn_map in attention_maps.items():
                layer_idx = int(layer_name.split('_')[1])
                
                # 視覚アテンション比率を計算
                overall_ratio, ratio_per_step = extractor.compute_visual_attention_ratio(attn_map)
                
                sample_data['layers'][layer_name] = {
                    'visual_attention_ratio': overall_ratio,
                    'visual_attention_per_step': ratio_per_step.tolist(),
                    'attention_map_shape': list(attn_map.shape)
                }
                
                # 統計情報に追加
                results['layer_stats'][layer_name]['visual_attention_ratios'].append(overall_ratio)
                results['layer_stats'][layer_name]['visual_attention_per_step'].append(ratio_per_step)
                
                # 個別マップ保存（オプション）
                if config.save_attention_maps:
                    sample_data['layers'][layer_name]['attention_map'] = attn_map.numpy().tolist()
            
            results['samples'].append(sample_data)
            sample_count += 1
    
    finally:
        extractor.remove_hooks()
    
    # 統計計算
    for layer_name in results['layer_stats']:
        ratios = results['layer_stats'][layer_name]['visual_attention_ratios']
        results['layer_stats'][layer_name]['mean_ratio'] = float(np.mean(ratios))
        results['layer_stats'][layer_name]['std_ratio'] = float(np.std(ratios))
        
        # 時間ステップごとの平均
        per_step_arrays = results['layer_stats'][layer_name]['visual_attention_per_step']
        # 各サンプルの長さが異なる可能性があるため、最小長に合わせる
        min_len = min(len(arr) for arr in per_step_arrays)
        truncated = [arr[:min_len] for arr in per_step_arrays]
        results['layer_stats'][layer_name]['mean_per_step'] = np.mean(truncated, axis=0).tolist()
    
    return results


def plot_layer_snr_heatmap(
    all_results: Dict[str, Dict[str, Dict]],
    config: AttentionAnalysisConfig,
    noise_type: str
):
    """
    層 × SNR のヒートマップを作成
    """
    snr_list = ["clean"] if noise_type == "none" else [f"{snr}dB" for snr in config.snr_dbs]
    num_layers = len(list(list(list(all_results.values())[0].values())[0]['layer_stats'].keys()))
    layer_names = [f"Layer {i}" for i in range(num_layers)]
    
    # データ行列作成
    data_matrix = np.zeros((num_layers, len(snr_list)))
    
    for snr_idx, snr_str in enumerate(snr_list):
        cond_key = f"{noise_type}" if snr_str == "clean" else f"{noise_type}_snr{snr_str[:-2]}"
        
        if noise_type in all_results and cond_key in all_results[noise_type]:
            layer_stats = all_results[noise_type][cond_key]['layer_stats']
            
            for layer_idx in range(num_layers):
                layer_name = f'layer_{layer_idx}'
                if layer_name in layer_stats:
                    data_matrix[layer_idx, snr_idx] = layer_stats[layer_name]['mean_ratio']
    
    # プロット
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_matrix,
        xticklabels=snr_list,
        yticklabels=layer_names,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Visual Attention Ratio'}
    )
    plt.xlabel('SNR Level')
    plt.ylabel('Transformer Layer')
    plt.title(f'Visual Attention Ratio by Layer and SNR\n({noise_type.capitalize()} Noise)')
    plt.tight_layout()
    
    output_dir = os.path.join(config.results_dir, noise_type)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'layer_snr_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"  Saved: {output_dir}/layer_snr_heatmap.png")


def plot_temporal_attention_heatmap(
    results: Dict[str, Any],
    config: AttentionAnalysisConfig,
    noise_type: str,
    snr: Optional[float],
    max_timesteps: int = 100
):
    """
    時間ステップ × 層 のヒートマップを作成
    """
    num_layers = len(results['layer_stats'])
    layer_names = [f"L{i}" for i in range(num_layers)]
    
    # データ行列作成（時間ステップ × 層）
    temporal_data = []
    for layer_idx in range(num_layers):
        layer_name = f'layer_{layer_idx}'
        mean_per_step = results['layer_stats'][layer_name]['mean_per_step']
        temporal_data.append(mean_per_step[:max_timesteps])
    
    temporal_data = np.array(temporal_data).T  # (timesteps, layers)
    
    # プロット
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        temporal_data,
        xticklabels=layer_names,
        yticklabels=range(0, len(temporal_data), max(1, len(temporal_data)//10)),
        cmap='viridis',
        cbar_kws={'label': 'Visual Attention Ratio'}
    )
    plt.xlabel('Layer')
    plt.ylabel('Time Step')
    
    snr_str = "Clean" if snr is None else f"{snr}dB"
    plt.title(f'Temporal Visual Attention Pattern\n({noise_type.capitalize()}, SNR: {snr_str})')
    plt.tight_layout()
    
    output_dir = os.path.join(config.results_dir, noise_type)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'temporal_heatmap_{snr_str.lower().replace(" ", "")}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"  Saved: {output_dir}/{filename}")


def plot_snr_vs_attention_lines(
    all_results: Dict[str, Dict[str, Dict]],
    config: AttentionAnalysisConfig,
    noise_type: str
):
    """
    SNR vs アテンション重み の折れ線グラフ（層ごと）
    """
    snr_values = config.snr_dbs if noise_type != "none" else [0]
    num_layers = len(list(list(list(all_results.values())[0].values())[0]['layer_stats'].keys()))
    
    plt.figure(figsize=(12, 6))
    
    for layer_idx in range(num_layers):
        layer_name = f'layer_{layer_idx}'
        attention_values = []
        
        for snr in snr_values:
            if noise_type == "none":
                cond_key = noise_type
            else:
                cond_key = f"{noise_type}_snr{snr}"
            
            if noise_type in all_results and cond_key in all_results[noise_type]:
                layer_stats = all_results[noise_type][cond_key]['layer_stats']
                if layer_name in layer_stats:
                    attention_values.append(layer_stats[layer_name]['mean_ratio'])
                else:
                    attention_values.append(np.nan)
            else:
                attention_values.append(np.nan)
        
        plt.plot(snr_values, attention_values, marker='o', label=f'Layer {layer_idx}', linewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Visual Attention Ratio', fontsize=12)
    plt.title(f'Visual Attention vs SNR by Layer\n({noise_type.capitalize()} Noise)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = os.path.join(config.results_dir, noise_type)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'snr_vs_attention_lines.png'), dpi=300)
    plt.close()
    
    print(f"  Saved: {output_dir}/snr_vs_attention_lines.png")


def plot_distribution_comparison(
    all_results: Dict[str, Dict[str, Dict]],
    config: AttentionAnalysisConfig,
    noise_type: str
):
    """
    Clean vs Noisy のアテンション分布比較
    """
    num_layers = len(list(list(list(all_results.values())[0].values())[0]['layer_stats'].keys()))
    
    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        layer_name = f'layer_{layer_idx}'
        ax = axes[layer_idx]
        
        # Noisy条件（最も低いSNR）
        if noise_type != "none" and len(config.snr_dbs) > 0:
            clean_snr = max(config.snr_dbs)
            noisy_snr = min(config.snr_dbs)
            
            clean_key = f"{noise_type}_snr{clean_snr}"
            noisy_key = f"{noise_type}_snr{noisy_snr}"
            
            clean_ratios = []
            noisy_ratios = []
            
            if noise_type in all_results:
                if clean_key in all_results[noise_type]:
                    clean_ratios = all_results[noise_type][clean_key]['layer_stats'][layer_name]['visual_attention_ratios']
                
                if noisy_key in all_results[noise_type]:
                    noisy_ratios = all_results[noise_type][noisy_key]['layer_stats'][layer_name]['visual_attention_ratios']
            
            if clean_ratios:
                ax.hist(clean_ratios, bins=20, alpha=0.5, label=f'High SNR ({clean_snr}dB)', color='blue')
            if noisy_ratios:
                ax.hist(noisy_ratios, bins=20, alpha=0.5, label=f'Low SNR ({noisy_snr}dB)', color='red')
            
            ax.set_xlabel('Visual Attention Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Layer {layer_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 使用しないサブプロットを非表示
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Attention Distribution: High SNR vs Low SNR\n({noise_type.capitalize()} Noise)', fontsize=14)
    plt.tight_layout()
    
    output_dir = os.path.join(config.results_dir, noise_type)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=300)
    plt.close()
    
    print(f"  Saved: {output_dir}/distribution_comparison.png")


def save_statistics_csv(
    all_results: Dict[str, Dict[str, Dict]],
    config: AttentionAnalysisConfig
):
    """
    統計情報をCSVで保存
    """
    import csv
    
    csv_path = os.path.join(config.results_dir, 'statistics.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Noise Type', 'SNR (dB)', 'Layer', 'Mean Visual Attention Ratio', 'Std Visual Attention Ratio'])
        
        for noise_type in all_results:
            for cond_key in all_results[noise_type]:
                # SNR抽出
                if '_snr' in cond_key:
                    snr = cond_key.split('_snr')[1]
                else:
                    snr = 'Clean'
                
                layer_stats = all_results[noise_type][cond_key]['layer_stats']
                
                for layer_name in layer_stats:
                    layer_idx = layer_name.split('_')[1]
                    mean_ratio = layer_stats[layer_name]['mean_ratio']
                    std_ratio = layer_stats[layer_name]['std_ratio']
                    
                    writer.writerow([noise_type, snr, layer_idx, f'{mean_ratio:.4f}', f'{std_ratio:.4f}'])
    
    print(f"\n[Statistics] Saved to: {csv_path}")


def create_summary_report(
    all_results: Dict[str, Dict[str, Dict]],
    config: AttentionAnalysisConfig,
    epoch: int
):
    """サマリーレポートをテキストファイルで保存"""
    summary_path = os.path.join(config.results_dir, "summary_report.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VISUAL ATTENTION ANALYSIS - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model Checkpoint: {config.checkpoint_dir}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Visual Tokens: {config.visual_tokens}\n")
        f.write(f"Analyzed Samples: {config.num_samples}\n\n")
        
        f.write("="*80 + "\n")
        f.write("ATTENTION STATISTICS BY NOISE TYPE AND SNR\n")
        f.write("="*80 + "\n\n")
        
        for noise_type in sorted(all_results.keys()):
            f.write(f"\n{'─'*80}\n")
            f.write(f"Noise Type: {noise_type.upper()}\n")
            f.write(f"{'─'*80}\n")
            
            for cond_key in sorted(all_results[noise_type].keys()):
                # SNR抽出
                if '_snr' in cond_key:
                    snr = cond_key.split('_snr')[1]
                    snr_str = f"{snr} dB"
                else:
                    snr_str = "Clean"
                
                f.write(f"\n  SNR: {snr_str}\n")
                f.write(f"  {'-'*76}\n")
                
                layer_stats = all_results[noise_type][cond_key]['layer_stats']
                num_layers = len(layer_stats)
                
                # 各層の統計
                for layer_idx in range(num_layers):
                    layer_name = f'layer_{layer_idx}'
                    if layer_name in layer_stats:
                        mean_ratio = layer_stats[layer_name]['mean_ratio']
                        std_ratio = layer_stats[layer_name]['std_ratio']
                        f.write(f"    Layer {layer_idx:2d}: Mean = {mean_ratio:.4f}, Std = {std_ratio:.4f}\n")
                
                # 全体平均
                all_means = [layer_stats[f'layer_{i}']['mean_ratio'] for i in range(num_layers) if f'layer_{i}' in layer_stats]
                overall_mean = np.mean(all_means)
                f.write(f"\n    Overall Mean: {overall_mean:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        # 主要な発見を自動生成
        findings = analyze_key_findings(all_results, config)
        for idx, finding in enumerate(findings, 1):
            f.write(f"{idx}. {finding}\n\n")
        
        f.write("="*80 + "\n")
        f.write("VISUALIZATION FILES\n")
        f.write("\n")
        for noise_type in config.noise_types:
            f.write(f"  [Directory: {noise_type}/]\n")
            f.write(f"  - layer_snr_heatmap.png: 層別・SNR別のアテンション比率ヒートマップ\n")
            f.write(f"  - snr_vs_attention_lines.png: SNRに対するアテンション変化の折れ線グラフ\n")
            f.write(f"  - distribution_comparison.png: Clean vs Noisyの分布比較\n")
            f.write(f"  - temporal_heatmap_*.png: 時間方向のアテンション推移\n")

    print(f"\n[Summary] Report saved to: {summary_path}")


def analyze_key_findings(
    all_results: Dict[str, Dict[str, Dict]],
    config: AttentionAnalysisConfig
) -> List[str]:
    """
    結果から主要な傾向を自動抽出する
    """
    findings = []
    
    # 1. ノイズによるアテンションの変化傾向 (全体)
    for noise_type in config.noise_types:
        if noise_type == "none" or noise_type not in all_results:
            continue
            
        clean_key = f"{noise_type}" if f"{noise_type}" in all_results else None
        # Cleanが見つからない場合はスキップまたは他のノイズタイプから推測可能だが、簡易的に実装
        
        # 最も低いSNR（高ノイズ）
        min_snr = min(config.snr_dbs)
        noisy_key = f"{noise_type}_snr{min_snr}"
        
        if noisy_key in all_results[noise_type]:
            # 全層の平均を計算
            stats = all_results[noise_type][noisy_key]['layer_stats']
            avg_noisy = np.mean([d['mean_ratio'] for d in stats.values()])
            
            findings.append(
                f"Noise Impact ({noise_type.upper()}): "
                f"SNR {min_snr}dBにおける平均Visual Attention Ratioは {avg_noisy:.4f} です。"
            )

    # 2. 最もアテンションが高い層の特定
    layer_scores = defaultdict(list)
    for noise_type in all_results:
        for cond in all_results[noise_type]:
            stats = all_results[noise_type][cond]['layer_stats']
            for layer_name, data in stats.items():
                layer_scores[layer_name].append(data['mean_ratio'])
    
    if layer_scores:
        # 各層の全条件平均を計算
        avg_scores = {k: np.mean(v) for k, v in layer_scores.items()}
        max_layer = max(avg_scores, key=avg_scores.get)
        max_score = avg_scores[max_layer]
        
        findings.append(
            f"Layer Sensitivity: 全条件を通して、最も視覚情報を利用しているのは {max_layer} "
            f"(平均 VAR: {max_score:.4f}) です。"
        )

    # 3. SNR低下に伴うアテンション上昇の有無（単純な傾向分析）
    for noise_type in config.noise_types:
        if noise_type == "none" or noise_type not in all_results:
            continue
            
        trends = []
        # SNRが高い順（Clean -> 10dB -> ... -> -5dB）に並べる
        sorted_snrs = sorted(config.snr_dbs, reverse=True)
        
        ratios = []
        for snr in sorted_snrs:
            key = f"{noise_type}_snr{snr}"
            if key in all_results[noise_type]:
                stats = all_results[noise_type][key]['layer_stats']
                ratios.append(np.mean([d['mean_ratio'] for d in stats.values()]))
        
        if len(ratios) > 1:
            # 簡易的な傾向判定: 最後（低SNR）が最初（高SNR）より高いか
            if ratios[-1] > ratios[0] * 1.1: # 10%以上増加
                findings.append(
                    f"Trend ({noise_type.upper()}): ノイズレベルが上がる（SNRが下がる）につれて、"
                    "視覚情報への依存度が増加する傾向が見られます。"
                )
            elif ratios[-1] < ratios[0] * 0.9:
                findings.append(
                    f"Trend ({noise_type.upper()}): ノイズレベルが上がるにつれて、"
                    "視覚情報へのアテンションが減少しています。"
                )

    return findings


class NumpyEncoder(json.JSONEncoder):
    """NumpyデータをJSON保存するためのエンコーダー"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def main():
    # 設定の初期化
    config = AttentionAnalysisConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 結果保存ディレクトリの作成
    os.makedirs(config.results_dir, exist_ok=True)

    # トークナイザーの準備（語彙サイズ取得のため）
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # または指定のトークナイザー
    config.vocab_size = tokenizer.vocab_size

    # モデルの構築
    print("Building model...")
    model = VisionConditionedASRv3(
        vocab_size=config.vocab_size,
        visual_tokens=config.visual_tokens,
        dropout=config.dropout,
        use_visual_pos_embed=config.use_visual_pos_embed
    ).to(device)

    # 重みのロード
    epoch = load_model_weights(model, config.checkpoint_dir, device)

    # データローダーの作成
    # 注: 検証用データを使用
    dataloader = create_dataloader(
        json_path=config.val_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=False,
        visual_tokens=config.visual_tokens,
        max_audio_length=config.max_audio_length
    )

    # 全結果を格納する辞書
    all_results = {}

    # ノイズ条件ごとにループ
    for noise_type in config.noise_types:
        all_results[noise_type] = {}
        
        # ノイズオーグメンターの準備（ベースノイズのロードなど）
        # 各SNRでのループの外で重いロードを行う
        augmenter_base = NoiseAugmenter(
            noise_type=noise_type,
            babble_path=config.babble_path
        )
        
        # SNRレベルごとにループ
        # Clean状態も比較用に一度だけ実行したい場合、noise_type='none'を含めるか、
        # noise_typeごとのループ内でSNR=None (Clean) を扱うか設計次第だが、
        # ここでは仕様通りsnr_dbsリストに従う
        
        # ノイズなし(Clean)比較用データの取得（最初のループで一度だけ取得など工夫可能だが、単純化のため都度実施）
        # ただし、今回はConfigのnoise_typesに"white", "pink"などが指定されている前提
        
        target_snrs = config.snr_dbs
        
        # Cleanデータも各Noise Typeフォルダにリファレンスとして保存するため、リストに追加して処理
        # ただしNoiseAugmenterの仕様上、snrを無視してnoneにする必要がある
        
        # 分析ループ
        for snr in target_snrs:
            # ノイズ設定更新
            augmenter_base.snr_db = snr
            
            # 結果キー
            result_key = f"{noise_type}_snr{snr}"
            
            # 分析実行
            results = analyze_attention(
                model=model,
                dataloader=dataloader,
                noise_augmenter=augmenter_base,
                config=config,
                noise_type=noise_type,
                snr=snr
            )
            
            all_results[noise_type][result_key] = results
            
            # 時間ヒートマップの保存（各条件ごと）
            plot_temporal_attention_heatmap(results, config, noise_type, snr)

        # 各ノイズタイプ終了ごとの可視化
        plot_layer_snr_heatmap(all_results, config, noise_type)
        plot_snr_vs_attention_lines(all_results, config, noise_type)
        plot_distribution_comparison(all_results, config, noise_type)

    # 最終データのJSON保存
    json_path = os.path.join(config.results_dir, 'attention_data.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2)
    print(f"\nSaved full data to: {json_path}")

    # CSV統計の保存
    save_statistics_csv(all_results, config)

    # サマリーレポート作成
    create_summary_report(all_results, config, epoch)

    print("\nAnalysis Completed Successfully.")


if __name__ == "__main__":
    main()
        