import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from tqdm import tqdm
import jiwer
from safetensors.torch import load_file
import librosa
import csv
import json
from collections import defaultdict
import random
from scipy import stats

# 自作モジュールのインポート
from model import VisionConditionedASRv3
from dataloader import create_dataloader
from train import TrainingConfig


@dataclass
class UnrelatedTestConfig:
    """無関係画像テスト設定"""
    # モデルチェックポイント
    checkpoint_dir: str = "../../Models/VisionConditionedASRv3/babble/epoch_15"
    
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
    batch_size: int = 32
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = False
    
    # 評価条件（リストで指定）
    noise_types: List[str] = field(default_factory=lambda: ["none", "babble"])
    snr_dbs: List[float] = field(default_factory=lambda: [-10.0, -5.0, 0.0, 5.0, 10.0])
    babble_path: str = "../../Datasets/NOISEX92/babble/signal.wav"
    
    # 繰り返し設定
    num_runs: int = 3  # 各条件での実行回数
    
    # デバイス
    device: str = "cuda:1"
    
    # 結果保存
    save_results: bool = True
    results_dir: str = "results/unrelated_test/babble"
    
    def __post_init__(self):
        print(f"\n{'='*60}")
        print("Unrelated vs Zero Vision Test Configuration")
        print(f"{'='*60}")
        print(f"Test Conditions: Unrelated Image vs Zero Vision")
        print(f"Results directory: {self.results_dir}")
        print(f"Noise Types: {self.noise_types}")
        print(f"SNR Levels: {self.snr_dbs}")
        print(f"Number of runs: {self.num_runs}")
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


class UnrelatedImageMapper:
    """
    データセット内の画像IDを管理し、無関係画像を割り当てるクラス
    """
    def __init__(self, dataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        
        # 画像ID（画像パス）をキーとして、そのインデックスリストを保持
        self.image_id_to_indices = defaultdict(list)
        
        for idx, item in enumerate(dataset.data):
            image_path = item["image"]
            self.image_id_to_indices[image_path].append(idx)
        
        # すべての画像IDのリスト
        self.all_image_ids = list(self.image_id_to_indices.keys())
        
        print(f"\n[UnrelatedImageMapper] Initialized")
        print(f"  Total samples: {len(dataset.data)}")
        print(f"  Unique images: {len(self.all_image_ids)}")
        print(f"  Seed: {seed}")
    
    def get_unrelated_image(self, current_image_path: str):
        """
        現在の画像とは異なる画像IDを持つサンプルからランダムに画像を選択
        
        Args:
            current_image_path: 現在のサンプルの画像パス
        
        Returns:
            PIL.Image: 無関係画像
        """
        # 現在の画像ID以外の候補を抽出
        candidate_image_ids = [img_id for img_id in self.all_image_ids if img_id != current_image_path]
        
        if len(candidate_image_ids) == 0:
            raise ValueError("No unrelated images available")
        
        # ランダムに画像IDを選択
        selected_image_id = random.choice(candidate_image_ids)
        
        # その画像IDを持つサンプルからランダムに1つ選択
        selected_idx = random.choice(self.image_id_to_indices[selected_image_id])
        
        # 画像を読み込んで返す
        from PIL import Image
        return Image.open(selected_image_id).convert("RGB")


def decode_predictions(logits: torch.Tensor, tokenizer, blank_token_id: int = 0) -> List[str]:
    """CTCの予測結果をテキストにデコード"""
    decoded_texts = []
    id2char = {v: k for k, v in tokenizer.get_vocab().items()}
    
    for logit_seq in logits:
        indices = torch.argmax(logit_seq, dim=-1)
        indices = torch.unique_consecutive(indices, dim=0)
        indices = [i.item() for i in indices if i.item() != blank_token_id]
        
        chars = []
        for idx in indices:
            char = id2char.get(idx, '')
            if char == '|':
                chars.append(' ')
            elif char and char not in ['<pad>', '<s>', '</s>', '<unk>']:
                chars.append(char)
        
        decoded_text = ''.join(chars)
        decoded_texts.append(decoded_text)
    
    return decoded_texts


def compute_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """WER（Word Error Rate）とその詳細を計算"""
    output = jiwer.process_words(references, hypotheses)
    
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    
    for alignment in output.alignments:
        for op in alignment:
            if op.type == 'substitute':
                total_substitutions += 1
            elif op.type == 'delete':
                total_deletions += 1
            elif op.type == 'insert':
                total_insertions += 1
            elif op.type == 'equal':
                total_hits += 1
    
    return {
        'wer': output.wer * 100,
        'mer': output.mer * 100,
        'wil': output.wil * 100,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': total_insertions,
        'hits': total_hits
    }


def load_model_weights(model: nn.Module, checkpoint_dir: str, device: torch.device):
    """チェックポイントから重みをロード"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint files incomplete in: {checkpoint_dir}")
    
    print(f"\n{'='*60}")
    print("Loading Checkpoint")
    print(f"{'='*60}")
    print(f"From: {checkpoint_dir}")
    
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    
    epoch = checkpoint_state.get('epoch', -1)
    print(f"Epoch: {epoch + 1}")
    print(f"{'='*60}\n")
    
    return epoch + 1


def evaluate_single_condition(
    model: VisionConditionedASRv3,
    condition: str,  # "unrelated", "zerovision"
    dataloader: DataLoader,
    unrelated_mapper: UnrelatedImageMapper,
    tokenizer,
    noise_augmenter: NoiseAugmenter,
    run_number: int,
    seed: int
) -> Dict[str, Any]:
    """
    1つの条件での評価を実行
    
    Args:
        condition: "unrelated", "zerovision"のいずれか
    """
    # シード設定
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    model.eval()
    
    # Zero Visionの場合のフック設定
    hook_handle = None
    if condition == "zerovision":
        def zero_vision_output_hook(module, input, output):
            return torch.zeros_like(output)
        hook_handle = model.vision_encoder.register_forward_hook(zero_vision_output_hook)
    
    all_references = []
    all_hypotheses = []
    all_samples = []
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Run {run_number} ({condition})", leave=False)):
                # ノイズ付加
                if noise_augmenter.noise_type != "none":
                    batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]
                
                # Unrelated条件の場合、画像を差し替え
                if condition == "unrelated":
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = min(start_idx + len(batch["image"]), len(dataloader.dataset))
                    
                    new_images = []
                    for i in range(start_idx, end_idx):
                        current_image_path = dataloader.dataset.data[i]["image"]
                        unrelated_image = unrelated_mapper.get_unrelated_image(current_image_path)
                        new_images.append(unrelated_image)
                    
                    batch["image"] = new_images
                
                # 推論
                logits = model(batch)
                
                # デコード
                hypotheses = decode_predictions(logits, tokenizer, blank_token_id=0)
                references = batch["text"]
                
                all_references.extend(references)
                all_hypotheses.extend(hypotheses)
                
                # サンプル保存（最初の20件）
                if len(all_samples) < 20:
                    for ref, hyp in zip(references, hypotheses):
                        all_samples.append({'reference': ref, 'hypothesis': hyp})
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    # 指標計算
    wer_metrics = compute_wer(all_references, all_hypotheses)
    
    return {
        'wer_metrics': wer_metrics,
        'num_samples': len(all_references),
        'references': all_references,
        'hypotheses': all_hypotheses,
        'samples': all_samples,
        'seed': seed,
        'condition': condition
    }


def save_run_details(
    result: Dict,
    output_dir: str,
    base_filename: str,
    noise_info: Dict
):
    """個別の実行結果（txt, csv）を保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # TXT保存
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Evaluation Details: {base_filename}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Condition: {result['condition']}\n")
        f.write(f"Noise Type: {noise_info['type']}\n")
        f.write(f"SNR: {noise_info['snr']} dB\n")
        f.write(f"Seed: {result['seed']}\n")
        f.write(f"\nTotal Samples: {result['num_samples']}\n")
        f.write(f"WER: {result['wer_metrics']['wer']:.2f}%\n")
        f.write(f"MER: {result['wer_metrics']['mer']:.2f}%\n")
        f.write(f"WIL: {result['wer_metrics']['wil']:.2f}%\n")
        f.write(f"\nError Breakdown:\n")
        f.write(f"  Substitutions: {result['wer_metrics']['substitutions']}\n")
        f.write(f"  Deletions:     {result['wer_metrics']['deletions']}\n")
        f.write(f"  Insertions:    {result['wer_metrics']['insertions']}\n")
        f.write(f"  Hits:          {result['wer_metrics']['hits']}\n")
        f.write("\nSample Predictions:\n")
        for i, sample in enumerate(result['samples']):
            f.write(f"{i+1}. Ref: {sample['reference']}\n")
            f.write(f"   Hyp: {sample['hypothesis']}\n")
    
    # CSV保存
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Reference', 'Hypothesis'])
        for i, (ref, hyp) in enumerate(zip(result['references'], result['hypotheses'])):
            writer.writerow([i+1, ref, hyp])


def compute_paired_statistics(wer_list_a: List[float], wer_list_b: List[float]) -> Dict[str, float]:
    """
    2つのWERリストに対してpaired t-testを実行
    
    Returns:
        統計情報の辞書
    """
    if len(wer_list_a) != len(wer_list_b):
        raise ValueError("WER lists must have the same length")
    
    # サンプルごとのWER差分
    diff = np.array(wer_list_a) - np.array(wer_list_b)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(wer_list_a, wer_list_b)
    
    return {
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),  # bool()で変換
        'significant_at_0.01': bool(p_value < 0.01)   # bool()で変換
    }


def interpret_results(
    wer_unrelated: float,
    wer_zero: float,
    stats_unrel_vs_zero: Dict
) -> str:
    """
    結果の解釈を自動生成（2条件比較版）
    
    期待されるパターン:
    - 意味的補助: Unrelated ≈ Zero (無関係画像は視覚情報なしと同等)
    - 正則化: Unrelated < Zero (無関係画像でも性能向上)
    """
    epsilon = 2.0  # WER差が2%以内なら「同等」とみなす
    
    diff_unrel_zero = wer_unrelated - wer_zero
    sig = stats_unrel_vs_zero['significant_at_0.05']
    
    # パターン判定
    if abs(diff_unrel_zero) < epsilon or not sig:
        # Unrelated ≈ Zero (統計的に有意でない、または差が小さい)
        return "Semantic Assistance: Unrelated images provide no benefit over zero vision. Vision requires semantic relevance to be useful."
    
    elif diff_unrel_zero < -epsilon and sig:
        # Unrelated < Zero (無関係画像の方が性能が良い)
        return "Regularization Effect: Even unrelated images improve performance over no vision. Visual features act primarily as regularization."
    
    elif diff_unrel_zero > epsilon and sig:
        # Unrelated > Zero (無関係画像の方が性能が悪い)
        return "Negative Transfer: Unrelated images actively harm performance compared to no vision. The model may be misled by incorrect visual context."
    
    else:
        return "Inconclusive: Results do not fit expected patterns. Further investigation needed."


def save_summary_report(results: Dict, config: UnrelatedTestConfig, epoch: int):
    """サマリーレポートをテキストファイルで保存（2条件版）"""
    summary_path = os.path.join(config.results_dir, "summary_report.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("UNRELATED vs ZERO VISION TEST - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model Checkpoint: {config.checkpoint_dir}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Number of Runs: {config.num_runs}\n")
        f.write(f"Test Purpose: Compare unrelated images vs zero vision\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY CONDITION\n")
        f.write("="*80 + "\n\n")
        
        for noise_type in sorted(results.keys()):
            f.write(f"\n{'─'*80}\n")
            f.write(f"Noise Type: {noise_type.upper()}\n")
            f.write(f"{'─'*80}\n")
            
            for cond_key in sorted(results[noise_type].keys()):
                cond_data = results[noise_type][cond_key]
                
                f.write(f"\n  Condition: {cond_key}\n")
                f.write(f"  {'-'*76}\n")
                
                # 各条件のWER
                for condition in ['unrelated', 'zerovision']:
                    wer = cond_data[condition]['wer_metrics']['wer']
                    f.write(f"    {condition.capitalize():12s}: WER = {wer:6.2f}%\n")
                
                # 分析結果
                if 'analysis' in cond_data:
                    analysis = cond_data['analysis']
                    f.write(f"\n  WER Difference:\n")
                    f.write(f"    Unrelated - Zero: {analysis['wer_diff_unrelated_vs_zero']:+6.2f}%\n")
                    
                    f.write(f"\n  Statistical Significance (p-value):\n")
                    stats = analysis['statistics']['unrelated_vs_zero']
                    f.write(f"    Unrelated vs Zero: p = {stats['p_value']:.4f} ")
                    f.write(f"({'*' if stats['significant_at_0.05'] else 'n.s.'})\n")
                    
                    f.write(f"\n  Interpretation:\n")
                    f.write(f"    {analysis['interpretation']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        # 全体的な傾向の分析
        all_interpretations = []
        for noise_type in results.keys():
            for cond_key in results[noise_type].keys():
                if 'analysis' in results[noise_type][cond_key]:
                    all_interpretations.append(results[noise_type][cond_key]['analysis']['interpretation'])
        
        semantic_count = sum(1 for interp in all_interpretations if 'Semantic' in interp)
        regularization_count = sum(1 for interp in all_interpretations if 'Regularization' in interp)
        negative_count = sum(1 for interp in all_interpretations if 'Negative' in interp)
        
        f.write(f"Across all {len(all_interpretations)} test conditions:\n")
        f.write(f"  - Semantic Assistance patterns: {semantic_count}\n")
        f.write(f"  - Regularization patterns: {regularization_count}\n")
        f.write(f"  - Negative Transfer patterns: {negative_count}\n")
        f.write(f"  - Other/Inconclusive: {len(all_interpretations) - semantic_count - regularization_count - negative_count}\n\n")
        
        if semantic_count > regularization_count:
            f.write("→ Vision primarily provides SEMANTIC ASSISTANCE.\n")
            f.write("  Unrelated images do not help (similar to no vision).\n")
        elif regularization_count > semantic_count:
            f.write("→ Vision primarily provides REGULARIZATION.\n")
            f.write("  Even unrelated images improve performance.\n")
        elif negative_count > 0:
            f.write("→ NEGATIVE TRANSFER detected.\n")
            f.write("  Unrelated images may mislead the model.\n")
        else:
            f.write("→ Results are MIXED or INCONCLUSIVE.\n")
            f.write("  Further investigation may be needed.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("* p < 0.05, ** p < 0.01\n")
        f.write("="*80 + "\n")
    
    print(f"\n[Summary] Report saved to: {summary_path}")


def run_unrelated_test():
    """無関係画像テストのメイン関数"""
    config = UnrelatedTestConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    print("\n[Setup] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルのロード
    print("\n[Setup] Loading VisionConditionedASRv3...")
    model = VisionConditionedASRv3(
        vocab_size=config.vocab_size,
        visual_tokens=config.visual_tokens,
        dropout=config.dropout,
        use_visual_pos_embed=config.use_visual_pos_embed,
        freeze_vision_encoder=True,
        device=device
    ).to(device)
    epoch = load_model_weights(model, config.checkpoint_dir, device)
    
    # データローダー作成
    print("\n[Setup] Creating DataLoader...")
    dataloader = create_dataloader(
        json_path=config.val_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=False,  # 重要: シャッフルしない（インデックス追跡のため）
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
    # UnrelatedImageMapperの初期化
    print("\n[Setup] Initializing UnrelatedImageMapper...")
    unrelated_mapper = UnrelatedImageMapper(dataloader.dataset, seed=42)
    
    # 結果集計用
    final_results = defaultdict(lambda: defaultdict(dict))
    
    print(f"\n[Start] Starting unrelated image test loop...")
    
    for noise_type in config.noise_types:
        # Noneノイズの場合はSNRループは1回だけ
        snr_list = [None] if noise_type == "none" else config.snr_dbs
        
        for snr in snr_list:
            # ノイズ増強器の準備
            current_snr = snr if snr is not None else 0.0
            augmenter = NoiseAugmenter(
                noise_type=noise_type,
                snr_db=current_snr,
                babble_path=config.babble_path
            )
            
            # 保存用ディレクトリパス
            snr_str = "clean" if snr is None else f"snr_{snr}dB"
            output_dir = os.path.join(config.results_dir, noise_type, snr_str)
            
            # 条件キー（JSON集計用）
            cond_key = f"{noise_type}" + (f"_snr{snr}" if snr is not None else "")
            
            # 複数回実行の平均を取るためのリスト
            runs_metrics = defaultdict(list)
            
            for run in range(1, config.num_runs + 1):
                seed = 42 + run - 1
                print(f"\n--- {noise_type} | {snr_str} | Run {run} ---")
                
                # Condition B: Unrelated
                res_unrelated = evaluate_single_condition(
                    model, "unrelated", dataloader, unrelated_mapper,
                    tokenizer, augmenter, run, seed
                )
                save_run_details(res_unrelated, output_dir, f"run{run}_unrelated", {'type': noise_type, 'snr': current_snr})
                runs_metrics['unrelated'].append(res_unrelated)
                
                # Condition C: Zero Vision
                res_zero = evaluate_single_condition(
                    model, "zerovision", dataloader, unrelated_mapper,
                    tokenizer, augmenter, run, seed
                )
                save_run_details(res_zero, output_dir, f"run{run}_zerovision", {'type': noise_type, 'snr': current_snr})
                runs_metrics['zerovision'].append(res_zero)
            
            # 平均メトリクスの計算
            for condition in ['unrelated', 'zerovision']:
                avg_wer = np.mean([r['wer_metrics']['wer'] for r in runs_metrics[condition]])
                
                aggregated = {
                    'wer_metrics': {
                        'wer': avg_wer,
                        'mer': np.mean([r['wer_metrics']['mer'] for r in runs_metrics[condition]]),
                        'wil': np.mean([r['wer_metrics']['wil'] for r in runs_metrics[condition]])
                    },
                    'num_samples': runs_metrics[condition][0]['num_samples'],
                    'all_runs_wer': [r['wer_metrics']['wer'] for r in runs_metrics[condition]]
                }
                final_results[noise_type][cond_key][condition] = aggregated
            
            # WER差分と統計検定
            wer_unrelated = [r['wer_metrics']['wer'] for r in runs_metrics['unrelated']]
            wer_zero = [r['wer_metrics']['wer'] for r in runs_metrics['zerovision']]
            
            avg_unrelated = np.mean(wer_unrelated)
            avg_zero = np.mean(wer_zero)
            
            # 統計検定
            stats_unrel_vs_zero = compute_paired_statistics(wer_unrelated, wer_zero)
            
            final_results[noise_type][cond_key]['analysis'] = {
                'wer_diff_unrelated_vs_zero': avg_unrelated - avg_zero,
                'statistics': {
                    'unrelated_vs_zero': stats_unrel_vs_zero
                },
                'interpretation': interpret_results(avg_unrelated, avg_zero, stats_unrel_vs_zero)
            }
            
            print(f"\n  > Avg WER (Unrelated):  {avg_unrelated:.2f}%")
            print(f"  > Avg WER (Zero):       {avg_zero:.2f}%")
            print(f"  > Δ (Unrel - Zero):     {avg_unrelated - avg_zero:.2f}% (p={stats_unrel_vs_zero['p_value']:.4f})")
    
    # 最終JSONの保存
    json_path = os.path.join(config.results_dir, "comprehensive_results.json")
    
    output_json = {
        '_metadata': {
            'checkpoint': config.checkpoint_dir,
            'epoch': epoch,
            'noise_types': config.noise_types,
            'snr_values': config.snr_dbs,
            'num_runs': config.num_runs,
            'dataset_size': len(dataloader.dataset),
            'test_description': 'Unrelated vs Zero Vision Test: Comparing unrelated images with zero vision condition'
        }
    }
    
    for k, v in final_results.items():
        output_json[k] = v
    
    os.makedirs(config.results_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # サマリーテキストの保存
    save_summary_report(final_results, config, epoch)
    
    print(f"\n{'='*60}")
    print(f"All tests finished. Results saved to: {config.results_dir}")
    print(f"Summary JSON: {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_unrelated_test()