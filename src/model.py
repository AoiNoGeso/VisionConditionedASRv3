import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np


class VisualAdapter(nn.Module):
    """
    DINOv2の出力を固定長のVisual Tokensに圧縮
    """
    def __init__(
        self,
        visual_tokens: int = 32,
        dim: int = 768,
        dropout: float = 0.1,
        use_pos_embed: bool = True
    ):
        super().__init__()
        self.visual_tokens = visual_tokens
        self.dim = dim
        self.use_pos_embed = use_pos_embed
        
        # Pre-pooling MLP
        self.pre_pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Adaptive Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(visual_tokens)
        
        # Post-pooling normalization
        self.layer_norm = nn.LayerNorm(dim)
        
        # Visual Tokens専用の位置埋め込み
        if use_pos_embed:
            self.visual_pos_embed = nn.Parameter(
                torch.zeros(1, visual_tokens, dim)
            )
        else:
            self.visual_pos_embed = None
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        if self.visual_pos_embed is not None:
            nn.init.normal_(self.visual_pos_embed, mean=0.0, std=0.02)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, 257, 768]
        Returns:
            visual_tokens: [B, visual_tokens, 768]
        """
        x = self.pre_pool(vision_features)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        
        if self.visual_pos_embed is not None:
            x = x + self.visual_pos_embed
        
        return x


class VisionEncoder(nn.Module):
    """DINOv2 + Visual Adapter"""
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        visual_tokens: int = 32,
        dropout: float = 0.1,
        use_pos_embed: bool = True,
        freeze: bool = True,
        device=None
    ):
        super().__init__()
        self.model_name = model_name
        self.visual_tokens = visual_tokens
        self._device = device
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.adapter = VisualAdapter(
            visual_tokens=visual_tokens,
            dim=768,
            dropout=dropout,
            use_pos_embed=use_pos_embed
        )
    
    def forward(self, data: dict) -> torch.Tensor:
        """
        Args:
            data: dict with key "image"
        Returns:
            visual_tokens: [B, visual_tokens, 768]
        """
        images = data["image"]
        
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        with torch.no_grad() if not self.model.training else torch.enable_grad():
            outputs = self.model(**inputs)
            vision_features = outputs.last_hidden_state
        
        if torch.isnan(vision_features).any() or torch.isinf(vision_features).any():
            raise RuntimeError("Vision features contain NaN/Inf")
        
        visual_tokens = self.adapter(vision_features)
        
        if torch.isnan(visual_tokens).any() or torch.isinf(visual_tokens).any():
            raise RuntimeError("Visual tokens contain NaN/Inf")
        
        return visual_tokens


class AudioEncoder(nn.Module):
    """Wav2Vec2を分解したAudio Encoder"""
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        super().__init__()
        self.model_name = model_name
        self._device = device
        
        wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Feature Extractorをfreeze
        wav2vec2_model.freeze_feature_encoder()
        
        self.feature_extractor = wav2vec2_model.feature_extractor
        self.feature_projection = wav2vec2_model.feature_projection
        self.masked_spec_embed = wav2vec2_model.masked_spec_embed
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        if self.masked_spec_embed is not None:
            nn.init.uniform_(self.masked_spec_embed.data, a=-0.01, b=0.01)
    
    def forward(self, data: dict) -> tuple:
        """
        Args:
            data: dict with key "wav"
        Returns:
            audio_embeddings: [B, T, 768]
            attention_mask: [B, T]
        """
        wav = data["wav"]
        
        if not wav or len(wav) == 0:
            raise ValueError("Empty audio input")
        
        if self._device is not None:
            device = self._device
        else:
            device = next(self.parameters()).device
        
        processed = self.processor(
            wav,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        input_values = processed.input_values.to(device)
        attention_mask = processed.attention_mask.to(device) if hasattr(processed, 'attention_mask') else None
        
        if torch.isnan(input_values).any() or torch.isinf(input_values).any():
            raise RuntimeError("Input values contain NaN/Inf")
        
        # CNN特徴抽出
        extract_features = self.feature_extractor(input_values)
        
        if torch.isnan(extract_features).any() or torch.isinf(extract_features).any():
            raise RuntimeError("CNN features contain NaN/Inf")
        
        # 特徴投影
        extract_features = extract_features.transpose(1, 2)
        hidden_states, _ = self.feature_projection(extract_features)
        
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            raise RuntimeError("Audio embeddings contain NaN/Inf")
        
        # Attention maskを調整
        if attention_mask is not None:
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            attention_mask_new = torch.ones(batch_size, seq_len, device=device)
        else:
            attention_mask_new = None
        
        return hidden_states, attention_mask_new


class VisionConditionedASRv3(nn.Module):
    """
    Vision-Conditioned ASR v3 (Prefix Tuning方式)
    
    Visual Tokensを音声系列の先頭に結合し、
    Transformer Encoderで処理する
    """
    
    def __init__(
        self,
        vocab_size: int = None,
        visual_tokens: int = 32,
        dropout: float = 0.1,
        use_visual_pos_embed: bool = True,
        freeze_vision_encoder: bool = True,
        device=None
    ):
        super().__init__()
        self._device = device
        self.visual_tokens = visual_tokens
        
        # Vision Encoder
        self.vision_encoder = VisionEncoder(
            model_name="facebook/dinov2-base",
            visual_tokens=visual_tokens,
            dropout=dropout,
            use_pos_embed=use_visual_pos_embed,
            freeze=freeze_vision_encoder,
            device=device
        )
        
        # Audio Encoder
        self.audio_encoder = AudioEncoder(
            model_name="facebook/wav2vec2-base-960h",
            device=device
        )
        
        # Transformer Encoder (Wav2Vec2から取得)
        wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder = wav2vec2_model.encoder
        
        # Vocab size
        self.vocab_size = vocab_size if vocab_size is not None else 32
        
        # Classifier
        self.classifier = nn.Linear(768, self.vocab_size)
        
        print(f"\n[VisionConditionedASRv3] Initialized")
        print(f"  Visual Tokens: {visual_tokens}")
        print(f"  Vocab Size: {self.vocab_size}")
        print(f"  Visual Position Embed: {use_visual_pos_embed}")
        print(f"  Vision Encoder Frozen: {freeze_vision_encoder}")
        
        self._print_parameters()
    
    def _print_parameters(self):
        """パラメータ数の表示"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        vision_total = sum(p.numel() for p in self.vision_encoder.parameters())
        vision_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        
        audio_total = sum(p.numel() for p in self.audio_encoder.parameters())
        audio_trainable = sum(p.numel() for p in self.audio_encoder.parameters() if p.requires_grad)
        
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        
        classifier_total = sum(p.numel() for p in self.classifier.parameters())
        
        print(f"\n{'='*70}")
        print("Model Parameters")
        print(f"{'='*70}")
        print(f"{'Component':<30} {'Total':>15} {'Trainable':>15}")
        print(f"{'-'*70}")
        print(f"{'Vision Encoder':<30} {vision_total:>15,} {vision_trainable:>15,}")
        print(f"{'Audio Encoder':<30} {audio_total:>15,} {audio_trainable:>15,}")
        print(f"{'Transformer Encoder':<30} {encoder_total:>15,} {encoder_trainable:>15,}")
        print(f"{'Classifier':<30} {classifier_total:>15,} {classifier_total:>15,}")
        print(f"{'-'*70}")
        print(f"{'Total':<30} {total:>15,} {trainable:>15,}")
        print(f"{'Trainable Ratio':<30} {'':<15} {100*trainable/total:>14.2f}%")
        print(f"{'='*70}\n")
    
    def forward(self, data: dict) -> torch.Tensor:
        """
        Args:
            data: dict with keys:
                - "image": List[PIL.Image]
                - "wav": List[np.ndarray]
        
        Returns:
            logits: [B, T, vocab_size] - 音声部分のみのCTC logits
        """
        # Visual Tokens: [B, visual_tokens, 768]
        visual_tokens = self.vision_encoder(data)
        
        # Audio Embeddings: [B, T, 768]
        audio_embeddings, attention_mask = self.audio_encoder(data)
        
        batch_size = audio_embeddings.size(0)
        audio_seq_len = audio_embeddings.size(1)
        
        # Visual + Audio を結合: [B, visual_tokens + T, 768]
        combined = torch.cat([visual_tokens, audio_embeddings], dim=1)
        
        # Attention Maskを拡張
        if attention_mask is not None:
            # Visual部分は常に有効
            visual_mask = torch.ones(batch_size, self.visual_tokens, device=combined.device)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Transformer Encoderで処理
        encoder_output = self.encoder(
            combined,
            attention_mask=combined_mask
        )
        
        # encoder_outputはBaseModelOutputなので.last_hidden_stateを取得
        if hasattr(encoder_output, 'last_hidden_state'):
            encoder_hidden = encoder_output.last_hidden_state
        else:
            encoder_hidden = encoder_output[0]
        
        # Visual Tokens部分を削除: [B, T, 768]
        audio_output = encoder_hidden[:, self.visual_tokens:, :]
        
        # NaN/Infチェック
        if torch.isnan(audio_output).any() or torch.isinf(audio_output).any():
            raise RuntimeError("Audio output contains NaN/Inf")
        
        # Classifier: [B, T, vocab_size]
        logits = self.classifier(audio_output)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("Logits contain NaN/Inf")
        
        return logits


def test_model_initialization():
    """モデル初期化のテスト"""
    print("="*70)
    print("VisionConditionedASRv3 - Initialization Test")
    print("="*70)
    
    print("\n[Test 1] Model initialization")
    print("-"*70)
    
    model = VisionConditionedASRv3(
        vocab_size=32,
        visual_tokens=32,
        dropout=0.1,
        use_visual_pos_embed=True,
        freeze_vision_encoder=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    model.eval()
    
    print("✓ Model initialized successfully")
    
    print("\n" + "="*70)
    print("✓ Initialization test passed!")
    print("="*70)


def test_forward_pass():
    """Forward passのテスト"""
    print("\n" + "="*70)
    print("Forward Pass Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VisionConditionedASRv3(
        vocab_size=32,
        visual_tokens=32,
        device=device
    )
    model.eval()
    
    print("\n[Test 2] Forward pass with dummy data")
    print("-"*70)
    
    # ダミーデータ作成
    batch_size = 2
    
    # 画像
    images = [Image.new('RGB', (224, 224), color='red') for _ in range(batch_size)]
    
    # 音声（1秒と2秒）
    wavs = [
        np.random.randn(16000).astype(np.float32),
        np.random.randn(32000).astype(np.float32)
    ]
    
    data = {
        "image": images,
        "wav": wavs
    }
    
    print(f"Input:")
    print(f"  Images: {len(images)} samples")
    print(f"  Audio:  {len(wavs)} samples")
    print(f"    - Audio 1: {len(wavs[0])} samples ({len(wavs[0])/16000:.2f}s)")
    print(f"    - Audio 2: {len(wavs[1])} samples ({len(wavs[1])/16000:.2f}s)")
    
    # Forward pass
    print(f"\n[Running forward pass...]")
    with torch.no_grad():
        logits = model(data)
    
    print(f"\nOutput:")
    print(f"  Shape: {logits.shape}")
    print(f"  Expected: [Batch={batch_size}, T=variable, Vocab=32]")
    
    assert logits.size(0) == batch_size, "Batch size mismatch!"
    assert logits.size(2) == 32, "Vocab size mismatch!"
    print("✓ Shape check passed")
    
    # NaN/Infチェック
    assert not torch.isnan(logits).any(), "NaN detected!"
    assert not torch.isinf(logits).any(), "Inf detected!"
    print("✓ NaN/Inf check passed")
    
    # 統計情報
    print(f"\nLogits statistics:")
    print(f"  Mean: {logits.mean().item():.6f}")
    print(f"  Std:  {logits.std().item():.6f}")
    print(f"  Min:  {logits.min().item():.6f}")
    print(f"  Max:  {logits.max().item():.6f}")
    
    print("\n" + "="*70)
    print("✓ Forward pass test passed!")
    print("="*70)


def test_gradient_flow():
    """Gradient flowのテスト"""
    print("\n" + "="*70)
    print("Gradient Flow Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VisionConditionedASRv3(
        vocab_size=32,
        visual_tokens=32,
        freeze_vision_encoder=True,
        device=device
    )
    model.train()
    
    # ダミーデータ
    images = [Image.new('RGB', (224, 224), color='blue') for _ in range(2)]
    wavs = [np.random.randn(16000).astype(np.float32) for _ in range(2)]
    data = {"image": images, "wav": wavs}
    
    print("\n[Running forward + backward...]")
    logits = model(data)
    loss = logits.sum()
    loss.backward()
    
    # Gradientチェック
    print("\n[Checking gradients...]")
    
    # Vision Encoder
    vision_dinov2_has_grad = any(
        p.grad is not None for p in model.vision_encoder.model.parameters()
    )
    vision_adapter_has_grad = any(
        p.grad is not None for p in model.vision_encoder.adapter.parameters()
    )
    
    # Audio Encoder
    audio_cnn_has_grad = any(
        p.grad is not None for p in model.audio_encoder.feature_extractor.parameters()
    )
    audio_proj_has_grad = any(
        p.grad is not None for p in model.audio_encoder.feature_projection.parameters()
    )
    
    # Transformer Encoder
    encoder_has_grad = any(
        p.grad is not None for p in model.encoder.parameters()
    )
    
    # Classifier
    classifier_has_grad = any(
        p.grad is not None for p in model.classifier.parameters()
    )
    
    print(f"  Vision DINOv2:           {vision_dinov2_has_grad} (should be False)")
    print(f"  Vision Adapter:          {vision_adapter_has_grad} (should be True)")
    print(f"  Audio CNN:               {audio_cnn_has_grad} (should be False)")
    print(f"  Audio Projection:        {audio_proj_has_grad} (should be True)")
    print(f"  Transformer Encoder:     {encoder_has_grad} (should be True)")
    print(f"  Classifier:              {classifier_has_grad} (should be True)")
    
    assert not vision_dinov2_has_grad, "Vision DINOv2 should be frozen!"
    assert vision_adapter_has_grad, "Vision Adapter should be trainable!"
    assert not audio_cnn_has_grad, "Audio CNN should be frozen!"
    assert audio_proj_has_grad, "Audio Projection should be trainable!"
    assert encoder_has_grad, "Transformer Encoder should be trainable!"
    assert classifier_has_grad, "Classifier should be trainable!"
    
    print("✓ All gradient flows are correct")
    
    print("\n" + "="*70)
    print("✓ Gradient flow test passed!")
    print("="*70)


def test_different_batch_sizes():
    """様々なバッチサイズのテスト"""
    print("\n" + "="*70)
    print("Different Batch Sizes Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VisionConditionedASRv3(
        vocab_size=32,
        visual_tokens=32,
        device=device
    )
    model.eval()
    
    print("\n[Testing various batch sizes...]")
    print("-"*70)
    
    for batch_size in [1, 2, 4, 8]:
        images = [Image.new('RGB', (224, 224), color='green') for _ in range(batch_size)]
        wavs = [np.random.randn(np.random.randint(16000, 48000)).astype(np.float32) 
                for _ in range(batch_size)]
        data = {"image": images, "wav": wavs}
        
        with torch.no_grad():
            logits = model(data)
        
        print(f"  Batch {batch_size}: Output shape {logits.shape}")
        assert logits.size(0) == batch_size, f"Batch size mismatch for {batch_size}!"
        assert logits.size(2) == 32, "Vocab size mismatch!"
    
    print("✓ All batch sizes handled correctly")
    
    print("\n" + "="*70)
    print("✓ Batch sizes test passed!")
    print("="*70)


def test_visual_tokens_removal():
    """Visual Tokensが正しく削除されているかテスト"""
    print("\n" + "="*70)
    print("Visual Tokens Removal Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for visual_tokens in [16, 32, 64]:
        print(f"\n[Testing with visual_tokens={visual_tokens}]")
        print("-"*70)
        
        model = VisionConditionedASRv3(
            vocab_size=32,
            visual_tokens=visual_tokens,
            device=device
        )
        model.eval()
        
        # 固定長の音声（1秒 = 16000サンプル → 約50フレーム）
        images = [Image.new('RGB', (224, 224), color='blue')]
        wavs = [np.random.randn(16000).astype(np.float32)]
        data = {"image": images, "wav": wavs}
        
        with torch.no_grad():
            logits = model(data)
        
        audio_seq_len = logits.size(1)
        expected_len = 50  # 約50フレーム
        
        print(f"  Visual tokens: {visual_tokens}")
        print(f"  Audio seq len: {audio_seq_len} (expected ~{expected_len})")
        print(f"  Output shape:  {logits.shape}")
        
        # Visual Tokensが削除されているので、audio_seq_lenは約50のはず
        assert 40 <= audio_seq_len <= 60, f"Unexpected seq len: {audio_seq_len}"
        print(f"✓ Visual tokens correctly removed")
    
    print("\n" + "="*70)
    print("✓ Visual tokens removal test passed!")
    print("="*70)


if __name__ == "__main__":
    try:
        test_model_initialization()
        test_forward_pass()
        test_gradient_flow()
        test_different_batch_sizes()
        test_visual_tokens_removal()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nVisionConditionedASRv3 is ready for training!")
        print("\n次のステップ:")
        print("  1. train_v3.py の実装")
        print("  2. 学習の実行")
        print("  3. test_v3.py での評価")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()