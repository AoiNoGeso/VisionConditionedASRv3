import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder
import jiwer
from safetensors.torch import load_file
import torchaudio
import librosa

from model import VisionConditionedASR
from finetune_noise import PureWav2Vec2ASR
from dataloader import create_dataloader
from train import TrainingConfig


@dataclass
class TestConfig:
    # checkpoint_dir: str = "../../Models/DINOv2_model/babble/epoch_20"
    checkpoint_dir: str = "../../Models/wav2vec2-finetune/babble/epoch_14"
    model_type: str = "pure"  # "pure" or "vision"
    
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    vocab_size: Optional[int] = None
    num_heads: int = 8
    dropout: float = 0.1
    
    batch_size: int = 16
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = False
    
    noise_type: str = "background"  # "none", "white", "pink", "background"
    snr_db: float = 0.0  # SNR in dB (全ノイズタイプ共通)
    background_path: Optional[str] = "../../Datasets/NOISEX92/babble/signal.wav"  # 背景音用
    
    use_beam_search: bool = True
    beam_width: int = 10
    use_image: bool = True
    
    device: str = "cuda:0"
    save_results: bool = True
    results_dir: str = "results/babble/"
    if noise_type != "none":
        results_dir += f"{noise_type}/snr_{snr_db}dB/"
    else:
        results_dir += "none/"
    
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}\n")

class NoiseAugmenter:
    """
    独自実装のノイズ付加クラス（dB単位でSNR制御）
    """
    def __init__(
        self,
        noise_type: str = "none",
        snr_db: float = 10.0,
        background_path: Optional[str] = None,  # 変更: noise_dir → background_path
        sample_rate: int = 16000
    ):
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.background_path = background_path  # 変更
        self.sample_rate = sample_rate
        self.background_audio = None  # 変更: 事前読み込み用
        
        print(f"\n[NoiseAugmenter] Type: {noise_type}")
        
        if noise_type == "none":
            print("  No noise augmentation")
        elif noise_type == "white":
            print(f"  White Noise - SNR: {snr_db} dB")
        elif noise_type == "pink":
            print(f"  Pink Noise - SNR: {snr_db} dB")
        elif noise_type == "background":
            if not background_path or not os.path.exists(background_path):  # 変更
                raise ValueError(f"Background noise file not found: {background_path}")  # 変更
            print(f"  Background Noise - SNR: {snr_db} dB")
            print(f"  Loading: {background_path}")  # 変更
            
            # 変更: librosaで事前に読み込み
            self.background_audio, sr = librosa.load(background_path, sr=self.sample_rate, mono=True)
            self.background_audio = self.background_audio.astype(np.float32)
            print(f"  Loaded: {len(self.background_audio)} samples ({len(self.background_audio)/self.sample_rate:.2f}s)")
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
    
    def _compute_rms(self, audio: np.ndarray) -> float:
        """RMS (Root Mean Square) を計算"""
        return np.sqrt(np.mean(audio ** 2))
    
    def _adjust_noise_level(self, signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """
        SNR(dB)に基づいてノイズレベルを調整
        
        SNR(dB) = 10 * log10(signal_power / noise_power)
        noise_power = signal_power / (10 ^ (SNR/10))
        """
        signal_rms = self._compute_rms(signal)
        noise_rms = self._compute_rms(noise)
        
        if noise_rms == 0:
            return noise
        
        # 目標ノイズRMSを計算
        snr_linear = 10 ** (snr_db / 10.0)
        target_noise_rms = signal_rms / np.sqrt(snr_linear)
        
        # ノイズをスケーリング
        noise_scaled = noise * (target_noise_rms / noise_rms)
        
        return noise_scaled
    
    def _generate_white_noise(self, length: int) -> np.ndarray:
        """ホワイトノイズを生成"""
        return np.random.randn(length).astype(np.float32)
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """
        ピンクノイズを生成（1/f特性）
        FFTを使用して周波数領域で生成
        """
        # ホワイトノイズから開始
        white = np.random.randn(length)
        
        # FFT
        fft = np.fft.rfft(white)
        
        # 周波数ビン（0は除外）
        freqs = np.arange(len(fft))
        freqs[0] = 1  # ゼロ除算回避
        
        # 1/f特性を適用
        fft = fft / np.sqrt(freqs)
        
        # 逆FFT
        pink = np.fft.irfft(fft, n=length)
        
        # 正規化
        pink = pink / np.max(np.abs(pink))
        
        return pink.astype(np.float32)
    
    def _load_background_noise(self, target_length: int) -> np.ndarray:
        """指定された背景音ファイルから切り出し（変更箇所）"""
        if self.background_audio is None:
            raise ValueError("Background audio not loaded")
        
        noise = self.background_audio
        
        # 長さ調整
        if len(noise) < target_length:
            # 短い場合は繰り返し
            num_repeats = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, num_repeats)[:target_length]
        else:
            # 長い場合はランダムな位置から切り出し
            start_idx = np.random.randint(0, len(noise) - target_length + 1)
            noise = noise[start_idx:start_idx + target_length]
        
        return noise.astype(np.float32)
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """
        音声にノイズを付加
        
        Args:
            audio: 入力音声 (np.ndarray)
        
        Returns:
            noisy_audio: ノイズ付加後の音声
        """
        if self.noise_type == "none":
            return audio
        
        # float32に変換
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        length = len(audio)
        
        # ノイズ生成
        if self.noise_type == "white":
            noise = self._generate_white_noise(length)
        elif self.noise_type == "pink":
            noise = self._generate_pink_noise(length)
        elif self.noise_type == "background":
            noise = self._load_background_noise(length)
        else:
            return audio
        
        # SNRに基づいてノイズレベルを調整
        noise_adjusted = self._adjust_noise_level(audio, noise, self.snr_db)
        
        # ノイズを加算
        noisy_audio = audio + noise_adjusted
        
        # クリッピング防止（-1.0 ~ 1.0の範囲に収める）
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        return noisy_audio


class CTCDecoder:
    def __init__(
        self,
        tokenizer,
        use_beam_search: bool = True,
        beam_width: int = 100
    ):
        self.tokenizer = tokenizer
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        
        vocab_list = []
        for i in range(tokenizer.vocab_size):
            token = tokenizer.convert_ids_to_tokens(i)
            if token is None:
                token = ""
            vocab_list.append(token)
        
        self.decoder = build_ctcdecoder(labels=vocab_list, kenlm_model_path=None)
        
        print(f"[CTCDecoder] Initialized")
        print(f"  Vocabulary size: {len(vocab_list)}")
        print(f"  Beam search: {use_beam_search}")
        if use_beam_search:
            print(f"  Beam width: {beam_width}")
    
    def decode(self, logits: torch.Tensor) -> List[str]:
        batch_size = logits.size(0)
        results = []
        
        for i in range(batch_size):
            logits_i = logits[i].cpu().numpy()
            if self.use_beam_search:
                text = self.decoder.decode(logits_i, beam_width=self.beam_width)
            else:
                text = self.decoder.decode(logits_i, beam_width=1)
            results.append(text)
        
        return results


def compute_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
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
    model: nn.Module,
    device: torch.device,
    model_type: str = "vision"
):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint files incomplete in: {checkpoint_dir}")
    
    print(f"\n[Loading] From: {checkpoint_dir}")
    print(f"[Loading] Model type: {model_type}")
    
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    
    epoch = checkpoint_state.get('epoch', -1)
    train_loss = checkpoint_state.get('train_loss', 0.0)
    val_loss = checkpoint_state.get('val_loss', 0.0)
    
    print(f"[Loading] Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return epoch + 1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    decoder: CTCDecoder,
    noise_augmenter: NoiseAugmenter,
    device: torch.device,
    config: TestConfig
):
    model.eval()
    
    hook_handle = None
    if config.model_type == "vision" and not config.use_image:
        print("\n[Evaluation] Image disabled (vision encoder output set to zero)")
        
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
    if config.model_type == "vision":
        print(f"Use image: {config.use_image}")
    print(f"{'='*60}\n")
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    # ノイズ付加
                    if noise_augmenter.noise_type != "none":
                        batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]
                    
                    logits = model(batch)
                    hypotheses = decoder.decode(logits)
                    references = batch["text"]
                    
                    all_references.extend(references)
                    all_hypotheses.extend(hypotheses)
                    
                    if len(all_samples) < 100:
                        for ref, hyp in zip(references, hypotheses):
                            all_samples.append({'reference': ref, 'hypothesis': hyp})
                except Exception as e:
                    print(f"\n[Error] Batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        if hook_handle is not None:
            hook_handle.remove()
            print("\n[Evaluation] Hook removed")
    
    print(f"\n{'='*60}\nComputing WER...\n{'='*60}")
    wer_metrics = compute_wer(all_references, all_hypotheses)
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples: {len(all_references)}")
    print(f"Noise: {config.noise_type}")
    if config.noise_type != "none":
        print(f"SNR: {config.snr_db} dB")
    if config.model_type == "vision":
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
    checkpoint_epoch: int,
    model_type: str
):
    os.makedirs(config.results_dir, exist_ok=True)
    
    # ファイル名のプレフィックスを決定
    if model_type == "pure":
        # wav2vec2単体のファインチューニングモデル
        prefix = "ft"
    elif model_type == "vision":
        if config.use_image:
            # VASRモデルでuse_imageがtrue
            prefix = "vision"
        else:
            # VASRモデルでuse_imageがfalse
            prefix = "novision"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # ファイル名: {prefix}_{noise_type}_epoch{epoch}
    base_filename = f"{prefix}_{config.noise_type}_epoch{checkpoint_epoch}"
    
    # テキストファイル
    results_file = os.path.join(config.results_dir, f"{base_filename}.txt")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"WER Evaluation Results ({model_type.upper()} Model)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model Type: {model_type}\n")
        if config.model_type == "vision":
            f.write(f"Use Image: {config.use_image}\n")
        f.write(f"Checkpoint: {config.checkpoint_dir}\n")
        f.write(f"Epoch: {checkpoint_epoch}\n")
        f.write(f"Dataset: {config.val_json}\n")
        f.write(f"Noise Type: {config.noise_type}\n")
        if config.noise_type != "none":
            f.write(f"SNR: {config.snr_db} dB\n")
        f.write(f"Beam Search: {config.use_beam_search}\n")
        if config.use_beam_search:
            f.write(f"Beam Width: {config.beam_width}\n")
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
    
    print(f"[Results] Saved to: {results_file}\n")
    
    # CSVファイル
    csv_file = os.path.join(config.results_dir, f"{base_filename}.csv")
    
    import csv
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Reference', 'Hypothesis'])
        for i, (ref, hyp) in enumerate(zip(results['references'], results['hypotheses'])):
            writer.writerow([i+1, ref, hyp])
    
    print(f"[Results] Predictions saved to: {csv_file}\n")


def main():
    config = TestConfig()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Test Configuration")
    print(f"{'='*60}")
    print(f"Model Type: {config.model_type}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config.checkpoint_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Noise: {config.noise_type}")
    if config.noise_type != "none":
        print(f"SNR: {config.snr_db} dB")
    if config.model_type == "vision":
        print(f"Use image: {config.use_image}")
    print(f"Beam search: {config.use_beam_search}")
    if config.use_beam_search:
        print(f"Beam width: {config.beam_width}")
    print(f"{'='*60}\n")
    
    noise_augmenter = NoiseAugmenter(
        noise_type=config.noise_type,
        snr_db=config.snr_db,
        background_path=config.background_path,
        sample_rate=16000
    )
    
    print("\n[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    print("[Setup] Initializing model...")
    if config.model_type == "pure":
        model = PureWav2Vec2ASR(device=device).to(device)
        print("[Model] Using PureWav2Vec2ASR")
    elif config.model_type == "vision":
        model = VisionConditionedASR(
            vocab_size=config.vocab_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            device=device
        ).to(device)
        print("[Model] Using VisionConditionedASR")
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    checkpoint_epoch = load_checkpoint(
        checkpoint_dir=config.checkpoint_dir,
        model=model,
        device=device,
        model_type=config.model_type
    )
    
    print("\n[Setup] Initializing decoder...")
    decoder = CTCDecoder(
        tokenizer=tokenizer,
        use_beam_search=config.use_beam_search,
        beam_width=config.beam_width
    )
    
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
    
    results = evaluate(
        model=model,
        dataloader=val_loader,
        decoder=decoder,
        noise_augmenter=noise_augmenter,
        device=device,
        config=config
    )
    
    if config.save_results:
        save_results(
            results=results,
            config=config,
            checkpoint_epoch=checkpoint_epoch,
            model_type=config.model_type
        )
    
    print("="*60 + "\nEvaluation Completed!\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()