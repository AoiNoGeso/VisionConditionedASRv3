from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoImageProcessor, AutoModel
import torch
import torch.nn as nn
from PIL import Image
import torchaudio
import numpy as np
import os

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        super().__init__()
        self.model_name = model_name
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.freeze_feature_encoder()

        if self.model.masked_spec_embed is not None:
            nn.init.uniform_(self.model.masked_spec_embed.data, a=-0.01, b=0.01)
                
        self._device = device
        
    def forward(self, data):
        wav = data["wav"]
        
        if not wav or len(wav) == 0:
            raise ValueError("Empty audio input received")
        
        for i, w in enumerate(wav):
            if len(w) == 0:
                raise ValueError(f"Audio sample {i} has zero length")
        
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device
        
        processed = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = processed.input_values.to(device)
        attention_mask = processed.attention_mask.to(device) if hasattr(processed, 'attention_mask') else None
        
        if torch.isnan(input_values).any() or torch.isinf(input_values).any():
            raise RuntimeError("Input values contain NaN or Inf after processing")
        
        audio_outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        audio_features = audio_outputs.last_hidden_state
        
        if torch.isnan(audio_features).any():
            raise RuntimeError("Audio features contain NaN values")
        if torch.isinf(audio_features).any():
            raise RuntimeError("Audio features contain Inf values")
        
        return audio_features


class VisionEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self._device = device
        
    def forward(self, data):
        """
        画像特徴を抽出（全パッチトークンを返す）
        
        Returns:
            vision_features: [batch, num_patches, dim]
                            DINOv2-base の場合 [B, 197, 768]
                            (1 [CLS] token + 196 patch tokens)
        """
        images = data["image"]
        
        if not images or len(images) == 0:
            raise ValueError("Empty image input received")
        
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(f"Image {i} is not a PIL.Image object: {type(img)}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Image {i} has invalid size: {img.size}")
        
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # DINOv2の全トークンを取得（[CLS] + パッチトークン）
        outputs = self.model(**inputs)
        image_features = outputs.last_hidden_state  # [B, 197, 768]
        
        if torch.isnan(image_features).any():
            raise RuntimeError("Image features contain NaN values")
        if torch.isinf(image_features).any():
            raise RuntimeError("Image features contain Inf values")
        
        return image_features


class CrossAttention(nn.Module):
    """
    次元削減なしのシンプルなCross Attention
    
    Multiple Vision Tokensに対応し、音声の各時刻が画像の全パッチにattentionを行う。
    768次元を維持することで情報損失を最小化。
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Multi-head Attention（次元変換なし）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Post-attention処理
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, audio_features, vision_features):
        """
        Args:
            audio_features: [batch, seq_len_audio, dim]
                           例: [B, 300, 768]
            vision_features: [batch, num_patches, dim]
                            例: [B, 257, 768]
        
        Returns:
            enhanced_audio_features: [batch, seq_len_audio, dim]
                                     例: [B, 300, 768]
        """
        # Multi-head Attention
        # MultiheadAttention内部でQ, K, V投影が行われる
        attn_output, attn_weights = self.multihead_attn(
            query=audio_features,      # [B, T_audio, 768]
            key=vision_features,       # [B, N_patches, 768]
            value=vision_features,     # [B, N_patches, 768]
            need_weights=True,
            average_attn_weights=True  # 全ヘッドの平均を返す
        )
        # attn_output: [B, T_audio, 768]
        # attn_weights: [B, T_audio, N_patches]
        
        # Residual connection + Layer Normalization
        output = self.layer_norm(audio_features + self.dropout(attn_output))
        
        return output


class VisionConditionedASR(nn.Module):
    def __init__(self, vocab_size=None, num_heads=8, dropout=0.1, device=None):
        """
        Vision-Conditioned ASR モデル（Ver3: シンプル設計版）
        
        Args:
            vocab_size: 語彙サイズ（Noneの場合は32）
            num_heads: Multi-head Attentionのヘッド数（推奨: 8 or 12）
            dropout: ドロップアウト率
            device: 計算デバイス
        """
        super().__init__()
        self._device = device
        
        # Audio Encoder: Wav2Vec2-base (768 dim)
        self.audio_encoder = AudioEncoder(device=device)
        
        # Vision Encoder: DINOv2-base (768 dim)
        self.vision_encoder = VisionEncoder(device=device)
        
        # Wav2Vec2-base-960h の語彙サイズは32（英語文字ベース）
        self.vocab_size = vocab_size if vocab_size is not None else 32
        
        # Cross Attention: 次元削減なし（768次元を維持）
        self.cross_attention = CrossAttention(
            dim=768,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classifier: 768次元から語彙サイズへ
        self.classifier = nn.Linear(768, self.vocab_size)
        
    def forward(self, data):
        """
        Args:
            data: dict with keys:
                - "wav": List[np.ndarray] - 音声データ
                - "image": List[PIL.Image] - 画像データ
        
        Returns:
            output_logits: [batch, seq_len, vocab_size]
        """
        # Audio features: [B, T_audio, 768]
        audio_features = self.audio_encoder(data)
        
        # Vision features: [B, N_patches, 768]
        # DINOv2-base の場合 [B, 257, 768]
        vision_features = self.vision_encoder(data)
        
        # Cross Attention: 音声が画像の全パッチに attention
        # [B, T_audio, 768]
        enhanced_audio = self.cross_attention(audio_features, vision_features)
        
        # メモリ解放
        del audio_features, vision_features
        
        # Classifier: [B, T_audio, vocab_size]
        output_logits = self.classifier(enhanced_audio)
        
        return output_logits
