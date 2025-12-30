import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from tqdm import tqdm
import wandb
from safetensors.torch import save_file, load_file
import librosa
import sys

from model import VisionConditionedASRv3
from dataloader import create_dataloader


@dataclass
class TrainingConfig:
    """VisionConditionedASRv3 学習設定"""
    # データセットパス
    train_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_train_fixed.json"
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定
    vocab_size: Optional[int] = None  # Noneの場合は32
    visual_tokens: int = 64  # Visual Tokensの数
    dropout: float = 0.1
    use_visual_pos_embed: bool = True  # Visual位置埋め込み
    
    # 学習設定
    batch_size: int = 20
    num_epochs: int = 15
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # データローダー設定
    num_workers: int = 8
    max_audio_length: float = 10.0
    validate_files: bool = True
    
    # レイヤーfreeze設定
    freeze_vision_encoder: bool = True  # DINOv2をfreeze
    freeze_audio_cnn: bool = True  # Wav2Vec2 CNNをfreeze (常にTrue)
    freeze_transformer_encoder: bool = False  # Transformer Encoderを学習
    audio_trainable_layers: int = 0  # Transformerの上位N層のみ学習（0=全層学習）
    
    # ノイズ設定（統一仕様）
    noise_type: str = "babble"  # "none", "white", "pink", "babble"
    snr_db: float = 0.0  # 全ノイズタイプ共通のSNR（dB）
    noise_prob: float = 0.5  # ノイズを付加する確率 (0.0-1.0)
    babble_path: str = "../../Datasets/NOISEX92/babble/signal.wav"
    
    # Mixed Precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    
    # チェックポイント
    checkpoint_dir: str = "../../Models/VisionConditionedASRv3/babble"
    save_epoch: int = 1
    resume_from: Optional[str] = "../../Models/VisionConditionedASRv3/clear/epoch_10"
    
    # デバイス
    device: str = "cuda:1"
    
    # ログ設定
    log_step: int = 100
    validate_epoch: int = 1
    use_wandb: bool = True
    wandb_project: str = "VisionConditionedASRv3-babble"


class NoiseAugmenter:
    """
    独自実装のノイズ付加クラス（dB単位でSNR制御、確率的付加）
    train.py準拠の実装
    """
    def __init__(
        self,
        noise_type: str = "none",
        snr_db: float = 10.0,
        noise_prob: float = 1.0,
        babble_path: Optional[str] = None,
        sample_rate: int = 16000
    ):
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.noise_prob = noise_prob
        self.babble_path = babble_path
        self.sample_rate = sample_rate
        self.babble_audio = None
        
        print(f"\n[NoiseAugmenter] Type: {noise_type}")
        print(f"  SNR: {snr_db} dB")
        print(f"  Noise Probability: {noise_prob * 100:.1f}%")
        
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
        if self.noise_type == "none" or np.random.random() > self.noise_prob:
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


def freeze_layers(model: VisionConditionedASRv3, config: TrainingConfig):
    """レイヤーのfreeze設定"""
    print("\n" + "="*60)
    print("Layer Freeze Configuration")
    print("="*60)
    
    # Vision Encoder
    if config.freeze_vision_encoder:
        for param in model.vision_encoder.model.parameters():
            param.requires_grad = False
        print("✓ Vision Encoder (DINOv2):  FROZEN")
    else:
        print("✓ Vision Encoder (DINOv2):  Trainable")
    
    # Vision Adapter（常に学習可能）
    print("✓ Visual Adapter:           Trainable")
    
    # Audio Encoder CNN（常にfreeze）
    for param in model.audio_encoder.feature_extractor.parameters():
        param.requires_grad = False
    print("✓ Audio Encoder (CNN):      FROZEN")
    
    # Audio Encoder Projection（常に学習可能）
    print("✓ Audio Encoder (Proj):     Trainable")
    
    # Transformer Encoder
    if config.freeze_transformer_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("✓ Transformer Encoder:      FROZEN")
    elif config.audio_trainable_layers > 0:
        # 上位N層のみ学習可能
        total_layers = len(model.encoder.layers)
        frozen_layers = total_layers - config.audio_trainable_layers
        
        for i, layer in enumerate(model.encoder.layers):
            if i < frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        print(f"✓ Transformer Encoder:      Last {config.audio_trainable_layers} layers trainable")
        print(f"                            First {frozen_layers} layers frozen")
    else:
        print("✓ Transformer Encoder:      Trainable (all layers)")
    
    # Classifier（常に学習可能）
    print("✓ Classifier:               Trainable")
    
    # パラメータ統計
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Frozen:    {total-trainable:,} ({100*(total-trainable)/total:.2f}%)")
    print(f"Total:     {total:,}")
    print(f"{'='*60}\n")


def decode_predictions(
    logits: torch.Tensor,
    tokenizer,
    blank_token_id: int = 0
) -> List[str]:
    """
    CTCの予測結果をテキストにデコード (torch.unique_consecutive使用)
    
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
        # 例: [A, A, B, B, B, C] -> [A, B, C]
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


def compute_ctc_loss(
    logits: torch.Tensor,
    texts: List[str],
    tokenizer,
    wav_lengths: torch.Tensor
) -> torch.Tensor:
    """CTC損失を計算"""
    batch_size = logits.size(0)
    device = logits.device
    
    target_ids = []
    target_lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        target_ids.extend(tokens)
        target_lengths.append(len(tokens))
    
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=device)
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    try:
        loss = ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
    except RuntimeError as e:
        print(f"\n[Warning] CTC error: {e}")
        loss = torch.tensor(1e6, device=device, requires_grad=True)
    
    return loss


def train_one_epoch(
    model: VisionConditionedASRv3,
    noise_augmenter: NoiseAugmenter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    tokenizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig
):
    """1エポック分の学習"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    
    print(f"\n{'='*60}\nEpoch {epoch+1}/{config.num_epochs} - Training\n{'='*60}")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train", total=num_batches)
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # ノイズ増強
            if noise_augmenter.noise_type != "none":
                batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]

            wav_lengths = batch["wav_lengths"].to(device)
            
            with autocast(device_type='cuda', enabled=config.use_amp, dtype=amp_dtype):
                logits = model(batch)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"\n🚨 Logits NaN/Inf at batch {batch_idx}, skipping")
                    continue
                loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🚨 Loss NaN/Inf at batch {batch_idx}, skipping")
                continue
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")
            
            if (batch_idx + 1) % config.log_step == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / (batch_idx + 1)
                if config.use_wandb:
                    wandb.log({
                        "train/loss_step": current_loss,
                        "train/avg_loss_step": avg_loss,
                        "train/scale": scaler.get_scale(),
                        "epoch": epoch
                    }, step=epoch * num_batches + batch_idx + 1)
            
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[Error] Batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if config.use_wandb:
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch})
    
    print(f"\n{'='*60}\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}\n{'='*60}\n")
    return avg_loss


def validate(
    model: VisionConditionedASRv3,
    noise_augmenter: NoiseAugmenter,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    num_examples: int = 3
):
    """検証"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_references = []
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    
    print(f"\n{'='*60}\nEpoch {epoch+1}/{config.num_epochs} - Validation\n{'='*60}")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Val", total=len(dataloader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                if noise_augmenter.noise_type != "none":
                    batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]

                wav_lengths = batch["wav_lengths"].to(device)
                
                with autocast(device_type='cuda', enabled=config.use_amp, dtype=amp_dtype):
                    logits = model(batch)
                    loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    current_loss = loss.item()
                    total_loss += current_loss
                    num_batches += 1
                    pbar.set_postfix(loss=f"{current_loss:.4f}")
                
                # logitsを直接デコード関数に渡す（argmaxしない）
                pred_texts = decode_predictions(logits, tokenizer, blank_token_id=0)
                all_predictions.extend(pred_texts)
                all_references.extend(batch["text"])
            except Exception as e:
                print(f"\n[Error] Val batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"\n{'='*60}\nPrediction Examples:\n{'='*60}")
    prediction_table = []
    for i in range(min(num_examples, len(all_predictions))):
        ref = all_references[i][:80]
        pred = all_predictions[i][:80]
        print(f"\nExample {i+1}:\n  Ref:  {ref}\n  Pred: {pred}")
        prediction_table.append([i+1, ref, pred])
        
    if config.use_wandb:
        wandb.log({
            "val/loss_epoch": avg_loss,
            "val/prediction_examples": wandb.Table(
                data=prediction_table,
                columns=["Example", "Reference", "Prediction"]
            ),
            "epoch": epoch
        })
    
    print(f"\n{'='*60}\nEpoch {epoch+1} Val Loss: {avg_loss:.4f}\n{'='*60}\n")
    return avg_loss


def save_checkpoint(
    model: VisionConditionedASRv3,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TrainingConfig
):
    """チェックポイントを保存"""
    epoch_dir = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    model_path = os.path.join(epoch_dir, f"model_epoch_{epoch+1}.safetensors")
    save_file(model.state_dict(), model_path)
    
    state_path = os.path.join(epoch_dir, f"checkpoint_epoch_{epoch+1}_state.pt")
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, state_path)
    
    print(f"[Checkpoint] Saved to: {model_path}")


def load_checkpoint(
    checkpoint_dir: str,
    model: VisionConditionedASRv3,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
):
    """チェックポイントから学習を再開"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint files incomplete in: {checkpoint_dir}")
    
    print(f"\n{'='*60}\nResuming from: {checkpoint_dir}\n{'='*60}")
    
    model.load_state_dict(load_file(model_path, device=str(device)))
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint_state:
        scaler.load_state_dict(checkpoint_state['scaler_state_dict'])
    
    start_epoch = checkpoint_state['epoch'] + 1
    print(f"[Resume] Epoch {start_epoch}, Loss: {checkpoint_state.get('val_loss', 0):.4f}\n{'='*60}\n")
    return start_epoch, checkpoint_state.get('val_loss', 0.0)


def main():
    """メイン学習関数"""
    config = TrainingConfig()
    
    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=config.__dict__)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nDevice: {device} | Batch: {config.batch_size} | LR: {config.learning_rate} | "
          f"Epochs: {config.num_epochs}\nVisual Tokens: {config.visual_tokens} | "
          f"Noise: {config.noise_type} | SNR: {config.snr_db}dB | "
          f"Noise Prob: {config.noise_prob*100:.1f}% | AMP: {config.use_amp}\n{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデル初期化
    model = VisionConditionedASRv3(
        vocab_size=config.vocab_size,
        visual_tokens=config.visual_tokens,
        dropout=config.dropout,
        use_visual_pos_embed=config.use_visual_pos_embed,
        freeze_vision_encoder=config.freeze_vision_encoder,
        device=device
    ).to(device)
    
    freeze_layers(model, config)
    scaler = GradScaler(enabled=config.use_amp)
    
    # DataLoader作成
    train_loader = create_dataloader(
        json_path=config.train_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Noise Augmenter
    noise_augmenter = NoiseAugmenter(
        noise_type=config.noise_type,
        snr_db=config.snr_db,
        noise_prob=config.noise_prob,
        babble_path=config.babble_path
    )
    
    print("\n" + "="*60 + "\nStarting Training\n" + "="*60 + "\n")
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    if config.resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_dir=config.resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )
    
    for epoch in range(start_epoch, config.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            noise_augmenter=noise_augmenter,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            config=config
        )
        
        val_loss = 0.0
        if (epoch + 1) % config.validate_epoch == 0:
            val_loss = validate(
                model=model,
                noise_augmenter=noise_augmenter,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                epoch=epoch,
                config=config
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n✨ New best: {best_val_loss:.4f}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    config=config
                )
        
        if (epoch + 1) % config.save_epoch == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config
            )
    
    print("\n" + "="*60 + "\nTraining Completed!\n" + "="*60 + "\n")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()