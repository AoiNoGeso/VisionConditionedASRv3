import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from tqdm import tqdm
import jiwer
from safetensors.torch import load_file
import librosa
import csv

from train import TrainingConfig
from model import VisionConditionedASRv3
from dataloader import create_dataloader


@dataclass
class TestConfig:
    """VisionConditionedASRv3 テスト設定"""
    # モデルチェックポイント
    checkpoint_dir: str = "../../Models/VisionConditionedASRv3/epoch_10"
    # checkpoint_dir: str = "../../Models/wav2vec2-finetune/babble/epoch_10"
    
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
    
    # ノイズ設定
    noise_type: str = "white"  # "none", "white", "pink", "babble"
    snr_db: float = -5.0
    babble_path: str = "../../Datasets/NOISEX92/babble/signal.wav"
    
    # Vision設定
    use_image: bool = True  # Falseの場合、visual特徴をゼロにする
    
    # デバイス
    device: str = "cuda:0"
    
    # 結果保存
    save_results: bool = True
    results_dir: str = "results/VisionConditionedASRv3/"
    
    def __post_init__(self):
        """結果保存ディレクトリを動的に設定"""
        base_dir = self.results_dir
        
        # ノイズタイプでディレクトリ分け
        if self.noise_type != "none":
            base_dir += f"{self.noise_type}/snr_{self.snr_db}dB/"
        else:
            base_dir += "none/"
        
        # Vision有無でさらに分け
        if self.use_image:
            base_dir += "vision/"
        else:
            base_dir += "novision/"
        
        self.results_dir = base_dir
        
        print(f"\n{'='*60}")
        print("Test Configuration")
        print(f"{'='*60}")
        print(f"Results directory: {self.results_dir}")
        print(f"Noise: {self.noise_type}")
        if self.noise_type != "none":
            print(f"SNR: {self.snr_db} dB")
        print(f"Use image: {self.use_image}")
        print(f"{'='*60}\n")


class NoiseAugmenter:
    """
    独自実装のノイズ付加クラス（dB単位でSNR制御）
    train_v3.pyと同じ実装
    """
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
        
        print(f"\n[NoiseAugmenter] Type: {noise_type}")
        print(f"  SNR: {snr_db} dB")
        
        if noise_type == "none":
            print("  No noise augmentation")
        elif noise_type == "white":
            print(f"  White Noise")
        elif noise_type == "pink":
            print(f"  Pink Noise (1/f)")
        elif noise_type == "babble":
            if not babble_path or not os.path.exists(babble_path):
                raise ValueError(f"Babble noise file not found: {babble_path}")
            print(f"  Babble Noise")
            print(f"  Loading: {babble_path}")
            
            self.babble_audio, sr = librosa.load(babble_path, sr=self.sample_rate, mono=True)
            self.babble_audio = self.babble_audio.astype(np.float32)
            print(f"  Loaded: {len(self.babble_audio)} samples ({len(self.babble_audio)/self.sample_rate:.2f}s)")
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
    
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


def decode_predictions(
    logits: torch.Tensor,
    tokenizer,
    blank_token_id: int = 0
) -> List[str]:
    """
    CTCの予測結果をテキストにデコード
    
    注意: tokenizer.decode()は連続する同じトークンを統合してしまうため使用しない
    例: [R, O, O, M] -> "ROM" (Oが1つに統合される)
    
    Args:
        logits: モデル出力 (batch_size, seq_len, vocab_size)
        tokenizer: トークナイザー
        blank_token_id: blankトークンのID
    
    Returns:
        デコードされたテキストのリスト
    """
    decoded_texts = []
    
    # トークンID→文字のマッピングを作成
    id2char = {v: k for k, v in tokenizer.get_vocab().items()}
    
    # バッチごとに処理
    for logit_seq in logits:  # shape: (seq_len, vocab_size)
        # 各フレームで最大確率のトークンを取得
        indices = torch.argmax(logit_seq, dim=-1)  # shape: (seq_len,)
        
        # 連続する重複を削除（CTCの標準的な処理）
        indices = torch.unique_consecutive(indices, dim=0)
        
        # blankトークンを除去
        indices = [i.item() for i in indices if i.item() != blank_token_id]
        
        # トークンIDを直接文字列に変換（tokenizer.decodeは使わない）
        chars = []
        for idx in indices:
            char = id2char.get(idx, '')
            if char == '|':  # '|'はスペース（単語境界）
                chars.append(' ')
            elif char and char not in ['<pad>', '<s>', '</s>', '<unk>']:  # 特殊トークンを除外
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


def load_checkpoint(
    checkpoint_dir: str,
    model: VisionConditionedASRv3,
    device: torch.device
):
    """チェックポイントからモデルを読み込み"""
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
    train_loss = checkpoint_state.get('train_loss', 0.0)
    val_loss = checkpoint_state.get('val_loss', 0.0)
    
    print(f"Epoch: {epoch + 1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return epoch + 1


def evaluate(
    model: VisionConditionedASRv3,
    dataloader: DataLoader,
    tokenizer,
    noise_augmenter: NoiseAugmenter,
    device: torch.device,
    config: TestConfig
):
    """モデルを評価"""
    model.eval()
    
    # Vision無効化のためのフック
    hook_handle = None
    if not config.use_image:
        print("\n[Evaluation] Image disabled (visual features set to zero)")
        
        def zero_vision_output_hook(module, input, output):
            return torch.zeros_like(output)
        
        hook_handle = model.vision_encoder.register_forward_hook(zero_vision_output_hook)
    
    all_references = []
    all_hypotheses = []
    all_samples = []
    
    print(f"\n{'='*60}")
    print("Starting Evaluation")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batches: {len(dataloader)}")
    print(f"Noise: {config.noise_type}")
    if config.noise_type != "none":
        print(f"SNR: {config.snr_db} dB")
    print(f"Use image: {config.use_image}")
    print(f"{'='*60}\n")
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    # ノイズ付加
                    if noise_augmenter.noise_type != "none":
                        batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]
                    
                    # モデル推論
                    logits = model(batch)
                    
                    # デコード（修正版：tokenizer.decodeを使わない）
                    hypotheses = decode_predictions(logits, tokenizer, blank_token_id=0)
                    references = batch["text"]
                    
                    all_references.extend(references)
                    all_hypotheses.extend(hypotheses)
                    
                    # サンプル保存（最初の100件）
                    if len(all_samples) < 100:
                        for ref, hyp in zip(references, hypotheses):
                            all_samples.append({'reference': ref, 'hypothesis': hyp})
                
                except Exception as e:
                    print(f"\n[Error] Batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    finally:
        # フックの削除
        if hook_handle is not None:
            hook_handle.remove()
            print("\n[Evaluation] Hook removed")
    
    # WER計算
    print(f"\n{'='*60}\nComputing WER...\n{'='*60}")
    wer_metrics = compute_wer(all_references, all_hypotheses)
    
    # 結果表示
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples: {len(all_references)}")
    print(f"Noise: {config.noise_type}")
    if config.noise_type != "none":
        print(f"SNR: {config.snr_db} dB")
    print(f"Use image: {config.use_image}")
    print(f"\nWER: {wer_metrics['wer']:.2f}%")
    print(f"MER: {wer_metrics['mer']:.2f}%")
    print(f"WIL: {wer_metrics['wil']:.2f}%")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {wer_metrics['substitutions']}")
    print(f"  Deletions:     {wer_metrics['deletions']}")
    print(f"  Insertions:    {wer_metrics['insertions']}")
    print(f"  Hits:          {wer_metrics['hits']}")
    print(f"{'='*60}\n")
    
    # サンプル表示
    print(f"{'='*60}\nSample Predictions (first 5):\n{'='*60}")
    for i, sample in enumerate(all_samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Ref:  {sample['reference']}")
        print(f"  Hyp:  {sample['hypothesis']}")
    print(f"{'='*60}\n")
    
    return {
        'wer_metrics': wer_metrics,
        'num_samples': len(all_references),
        'references': all_references,
        'hypotheses': all_hypotheses,
        'samples': all_samples
    }


def save_results(
    results: Dict,
    config: TestConfig,
    checkpoint_epoch: int
):
    """結果をファイルに保存"""
    os.makedirs(config.results_dir, exist_ok=True)
    
    # ファイル名: vision/novision_noise_epoch
    if config.use_image:
        prefix = "vision"
    else:
        prefix = "novision"
    
    base_filename = f"{prefix}_{config.noise_type}_epoch{checkpoint_epoch}"
    
    # テキストファイル
    results_file = os.path.join(config.results_dir, f"{base_filename}.txt")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("VisionConditionedASRv3 Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Checkpoint: {config.checkpoint_dir}\n")
        f.write(f"Epoch: {checkpoint_epoch}\n")
        f.write(f"Dataset: {config.val_json}\n")
        f.write(f"Use Image: {config.use_image}\n")
        f.write(f"Noise Type: {config.noise_type}\n")
        if config.noise_type != "none":
            f.write(f"SNR: {config.snr_db} dB\n")
        f.write(f"\nTotal samples: {results['num_samples']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Metrics:\n")
        f.write("="*60 + "\n")
        f.write(f"WER: {results['wer_metrics']['wer']:.2f}%\n")
        f.write(f"MER: {results['wer_metrics']['mer']:.2f}%\n")
        f.write(f"WIL: {results['wer_metrics']['wil']:.2f}%\n")
        f.write(f"\nError Breakdown:\n")
        f.write(f"  Substitutions: {results['wer_metrics']['substitutions']}\n")
        f.write(f"  Deletions:     {results['wer_metrics']['deletions']}\n")
        f.write(f"  Insertions:    {results['wer_metrics']['insertions']}\n")
        f.write(f"  Hits:          {results['wer_metrics']['hits']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Sample Predictions (first 20):\n")
        f.write("="*60 + "\n\n")
        
        for i, sample in enumerate(results['samples'][:20]):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  REF: {sample['reference']}\n")
            f.write(f"  HYP: {sample['hypothesis']}\n\n")
    
    print(f"[Results] Saved to: {results_file}")
    
    # CSVファイル
    csv_file = os.path.join(config.results_dir, f"{base_filename}.csv")
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Reference', 'Hypothesis'])
        for i, (ref, hyp) in enumerate(zip(results['references'], results['hypotheses'])):
            writer.writerow([i+1, ref, hyp])
    
    print(f"[Results] Predictions saved to: {csv_file}\n")


def main():
    """メイン実行関数"""
    config = TestConfig()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print("VisionConditionedASRv3 Test Script")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config.checkpoint_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Noise: {config.noise_type}")
    if config.noise_type != "none":
        print(f"SNR: {config.snr_db} dB")
    print(f"Use image: {config.use_image}")
    print(f"{'='*60}\n")
    
    # ノイズ増強器の初期化
    noise_augmenter = NoiseAugmenter(
        noise_type=config.noise_type,
        snr_db=config.snr_db,
        babble_path=config.babble_path,
        sample_rate=16000
    )
    
    # トークナイザーの読み込み
    print("\n[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルの初期化
    print("[Setup] Initializing model...")
    model = VisionConditionedASRv3(
        vocab_size=config.vocab_size,
        visual_tokens=config.visual_tokens,
        dropout=config.dropout,
        use_visual_pos_embed=config.use_visual_pos_embed,
        freeze_vision_encoder=True,
        device=device
    ).to(device)
    
    # チェックポイントの読み込み
    checkpoint_epoch = load_checkpoint(
        checkpoint_dir=config.checkpoint_dir,
        model=model,
        device=device
    )
    
    # データローダーの作成
    print("\n[Setup] Creating dataloader...")
    val_loader = create_dataloader(
        json_path=config.val_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
    # 評価実行
    results = evaluate(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        noise_augmenter=noise_augmenter,
        device=device,
        config=config
    )
    
    # 結果保存
    if config.save_results:
        save_results(
            results=results,
            config=config,
            checkpoint_epoch=checkpoint_epoch
        )
    
    print("="*60 + "\nEvaluation Completed!\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()