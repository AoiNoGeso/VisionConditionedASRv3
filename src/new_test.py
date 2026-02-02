import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor
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

# 自作モジュールのインポート
from model import VisionConditionedASRv3
from dataloader import create_dataloader

# 同ディレクトリのfinetune_noise.pyからクラスをインポート
from finetune_noise import PureWav2Vec2ASR


@dataclass
class TestConfig:
    """ノイズ負荷テスト設定"""
    # 評価対象モデルの選択（複数選択可能）
    eval_models: List[str] = field(default_factory=lambda: ["vision"])
    # 選択肢: "baseline" (Wav2Vec2単体), "vision" (VASR+画像), "novision" (VASR画像なし)
    
    # 提案モデル(VASR)チェックポイント
    checkpoint_dir: str = "../../Models/VisionConditionedASRv3/white/epoch_15"
    
    # ベースラインモデル(Wav2Vec2単体)チェックポイント
    baseline_checkpoint_dir: str = "../../Models/wav2vec2-finetune/white/epoch_15"
    
    # データセットパス
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定
    vocab_size: Optional[int] = None
    visual_tokens: int = 64
    dropout: float = 0.0
    use_visual_pos_embed: bool = True
    
    # データローダー設定
    batch_size: int = 30
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = False
    
    # 評価条件（リストで指定）
    noise_types: List[str] = field(default_factory=lambda: ["babble"]) # "none", "white", "pink", "babble"
    snr_dbs: List[float] = field(default_factory=lambda: [-10.0, -5.0, 0.0, 5.0, 10.0])
    babble_path: str = "../../Datasets/NOISEX92/babble/signal.wav"
    
    # 繰り返し設定
    num_runs: int = 3  # 各条件での実行回数
    
    # デバイス
    device: str = "cuda:1"
    
    # 結果保存
    save_results: bool = True
    results_dir: str = "results/PrefixVASR_white/"
    
    def __post_init__(self):
        # 評価対象モデルの検証
        valid_models = ["baseline", "vision", "novision"]
        for model in self.eval_models:
            if model not in valid_models:
                raise ValueError(f"Invalid model type: {model}. Choose from {valid_models}")
        
        print(f"\n{'='*60}")
        print("Comprehensive Test Configuration")
        print(f"{'='*60}")
        print(f"Evaluation Models: {self.eval_models}")
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
        print(f"Warning: Checkpoint not found at {checkpoint_dir}")
        return 0

    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = load_file(model_path, device=str(device))
        model.load_state_dict(state_dict)
        print("Loaded successfully.")
        return epoch_num
    else:
        print(f"Model file not found: {model_path}")
        return 0


def evaluate_single_run(
    model: nn.Module,
    model_name: str,  # "baseline", "vision", "novision"
    dataloader: DataLoader,
    tokenizer,
    noise_augmenter: NoiseAugmenter,
    run_number: int,
    seed: int
) -> Dict[str, Any]:
    """1回の評価実行を行う関数"""
    
    # シード設定
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    model.eval()
    
    # novisionの場合のフック設定（VisionConditionedASRv3のみ）
    hook_handle = None
    if model_name == "novision":
        def zero_vision_output_hook(module, input, output):
            return torch.zeros_like(output)
        # VisionConditionedASRv3はvision_encoder属性を持つと仮定
        if hasattr(model, 'vision_encoder'):
            hook_handle = model.vision_encoder.register_forward_hook(zero_vision_output_hook)
        
    all_references = []
    all_hypotheses = []
    all_samples = []
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Run {run_number} ({model_name})", leave=False):
                # ノイズ付加
                if noise_augmenter.noise_type != "none":
                    batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]
                
                # 推論
                logits = model(batch)
                
                # デコード
                hypotheses = decode_predictions(logits, tokenizer, blank_token_id=0)
                references = batch["text"]
                
                all_references.extend(references)
                all_hypotheses.extend(hypotheses)
                
                # サンプル保存（最初の20件程度）
                if len(all_samples) < 20:
                    for ref, hyp in zip(references, hypotheses):
                        all_samples.append({'reference': ref, 'hypothesis': hyp})
                        
    except Exception as e:
        print(f"Error during evaluation: {e}")
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
        'model_name': model_name
    }


def save_run_details(
    result: Dict,
    output_dir: str,
    base_filename: str,
    config: TestConfig,
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
        f.write(f"Model: {result['model_name']}\n")
        f.write(f"Noise Type: {noise_info['type']}\n")
        f.write(f"SNR: {noise_info['snr']} dB\n")
        f.write(f"Seed: {result['seed']}\n")
        f.write(f"\nTotal Samples: {result['num_samples']}\n")
        f.write(f"WER: {result['wer_metrics']['wer']:.2f}%\n")
        f.write(f"MER: {result['wer_metrics']['mer']:.2f}%\n")
        f.write(f"WIL: {result['wer_metrics']['wil']:.2f}%\n")
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


def run_comprehensive_test():
    """全条件を一括実行するメイン関数"""
    config = TestConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    print("\n[Setup] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルのロード（選択されたもののみ）
    models = {}
    
    if "vision" in config.eval_models or "novision" in config.eval_models:
        print("\n[Setup] Loading Proposed Model (VisionConditionedASRv3)...")
        proposed_model = VisionConditionedASRv3(
            vocab_size=config.vocab_size,
            visual_tokens=config.visual_tokens,
            dropout=config.dropout,
            use_visual_pos_embed=config.use_visual_pos_embed,
            freeze_vision_encoder=True,
            device=device
        ).to(device)
        prop_epoch = load_model_weights(proposed_model, config.checkpoint_dir, device)
        
        if "vision" in config.eval_models:
            models["vision"] = proposed_model
        if "novision" in config.eval_models:
            models["novision"] = proposed_model  # 同じモデルを使用（hookで制御）
    
    if "baseline" in config.eval_models:
        print("\n[Setup] Loading Baseline Model (PureWav2Vec2ASR)...")
        baseline_model = PureWav2Vec2ASR(
            model_name="facebook/wav2vec2-base-960h",
            device=device
        ).to(device)
        base_epoch = load_model_weights(baseline_model, config.baseline_checkpoint_dir, device)
        models["baseline"] = baseline_model
    
    # データローダー作成
    print("\n[Setup] Creating DataLoader...")
    dataloader = create_dataloader(
        json_path=config.val_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
    # 結果集計用
    # 構造: results[noise_type][condition_key][model_name] = {...}
    final_results = defaultdict(lambda: defaultdict(dict))
    
    # ループ開始
    print(f"\n[Start] Starting comprehensive evaluation loop...")
    print(f"Evaluating models: {config.eval_models}")
    
    for noise_type in config.noise_types:
        # Noneノイズの場合はSNRループは1回だけ（None）
        snr_list = [None] if noise_type == "none" else config.snr_dbs
        
        for snr in snr_list:
            # ノイズ増強器の準備
            current_snr = snr if snr is not None else 0.0
            augmenter = NoiseAugmenter(
                noise_type=noise_type,
                snr_db=current_snr,
                babble_path=config.babble_path
            )
            
            # 保存用ディレクトリパスの作成
            snr_str = "Clean" if snr is None else f"snr_{snr}dB"
            output_dir = os.path.join(config.results_dir, noise_type, snr_str)
            
            # 条件キー（JSON集計用）
            cond_key = f"{noise_type}" + (f"_snr{snr}" if snr is not None else "")
            
            # 複数回実行の平均を取るためのリスト
            runs_metrics = defaultdict(list)
            
            for run in range(1, config.num_runs + 1):
                seed = 42 + run - 1
                print(f"\n--- {noise_type} | {snr_str} | Run {run} ---")
                
                # 選択されたモデルのみ評価
                for model_name in config.eval_models:
                    model = models[model_name]
                    
                    result = evaluate_single_run(
                        model, model_name, dataloader, tokenizer, 
                        augmenter, run, seed
                    )
                    save_run_details(
                        result, output_dir, f"{model_name}_run{run}", 
                        config, {'type': noise_type, 'snr': current_snr}
                    )
                    runs_metrics[model_name].append(result)
            
            # 平均メトリクスの計算とJSONへの格納
            for model_name in config.eval_models:
                avg_wer = np.mean([r['wer_metrics']['wer'] for r in runs_metrics[model_name]])
                
                # 集計データの作成
                aggregated = {
                    'wer_metrics': {
                        'wer': avg_wer,
                        'mer': np.mean([r['wer_metrics']['mer'] for r in runs_metrics[model_name]]),
                        'wil': np.mean([r['wer_metrics']['wil'] for r in runs_metrics[model_name]])
                    },
                    'num_samples': runs_metrics[model_name][0]['num_samples'],
                    'all_runs_wer': [r['wer_metrics']['wer'] for r in runs_metrics[model_name]]
                }
                final_results[noise_type][cond_key][model_name] = aggregated
                
                print(f"  > Avg WER ({model_name}): {avg_wer:.2f}%")

            # 貢献度の計算（vision/novision/baselineが全て揃っている場合のみ）
            if all(m in config.eval_models for m in ["vision", "novision", "baseline"]):
                avg_vis = final_results[noise_type][cond_key]['vision']['wer_metrics']['wer']
                avg_novis = final_results[noise_type][cond_key]['novision']['wer_metrics']['wer']
                avg_base = final_results[noise_type][cond_key]['baseline']['wer_metrics']['wer']
                
                imp_vis = avg_novis - avg_vis
                imp_base = avg_base - avg_vis
                
                final_results[noise_type][cond_key]['contribution'] = {
                    'wer_improvement': imp_vis,
                    'improvement_rate': (imp_vis / avg_novis * 100) if avg_novis > 0 else 0,
                    'wer_improvement_vs_baseline': imp_base,
                    'improvement_rate_vs_baseline': (imp_base / avg_base * 100) if avg_base > 0 else 0,
                    'wer_vision': avg_vis,
                    'wer_novision': avg_novis,
                    'wer_baseline': avg_base
                }

    # 最終的なJSONの保存
    json_path = os.path.join(config.results_dir, "comprehensive_results.json")
    
    # メタデータ付与
    output_json = {
        '_metadata': {
            'eval_models': config.eval_models,
            'checkpoint': config.checkpoint_dir,
            'baseline_checkpoint': config.baseline_checkpoint_dir,
            'noise_types': config.noise_types,
            'snr_values': config.snr_dbs,
            'num_runs': config.num_runs,
            'dataset_size': len(dataloader.dataset)
        }
    }
    # resultsの中身をマージ
    for k, v in final_results.items():
        output_json[k] = v
        
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
        
    print(f"\n{'='*60}")
    print(f"All tests finished. Results saved to: {config.results_dir}")
    print(f"Summary JSON: {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_comprehensive_test()