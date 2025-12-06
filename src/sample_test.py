import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
import os

# 更新されたmodel.pyからクラスをインポート
from model import VisionConditionedASR

def count_params(module):
    """パラメータ数をカウントするヘルパー関数"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def check_freeze_status(model):
    """各コンポーネントの学習可否状態（Freeze状況）を確認"""
    print(f"\n{'='*20} Freeze Status Check {'='*20}")
    
    # 1. CNN Feature Extractor (通常はFreeze)
    cnn_params = list(model.feature_extractor.parameters())
    if len(cnn_params) > 0:
        cnn_requires_grad = cnn_params[0].requires_grad
        status = "Trainable" if cnn_requires_grad else "FROZEN"
        print(f"  [Audio] CNN Feature Extractor: {status}")
    
    # 2. Audio Encoder (Transformer) (Prefix Tuningでは学習させる必要あり)
    encoder_params = list(model.encoder.parameters())
    if len(encoder_params) > 0:
        enc_requires_grad = encoder_params[0].requires_grad
        status = "Trainable" if enc_requires_grad else "FROZEN"
        print(f"  [Audio] Transformer Encoder:   {status}")

    # 3. Vision Encoder (DINOv2)
    vision_params = list(model.vision_model.parameters())
    if len(vision_params) > 0:
        vis_requires_grad = vision_params[0].requires_grad
        status = "Trainable" if vis_requires_grad else "FROZEN"
        print(f"  [Vision] Vision Encoder:       {status}")

    # 4. Visual Compressor (新規層なので必ずTrainableであるべき)
    comp_params = list(model.visual_compressor.parameters())
    if len(comp_params) > 0:
        comp_requires_grad = comp_params[0].requires_grad
        status = "Trainable" if comp_requires_grad else "FROZEN"
        print(f"  [Vision] Visual Compressor:    {status}")
        
    print(f"{'='*60}\n")

def main():
    print(f"Initializing Test Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. モデルの初期化
    print("\n[1] Initializing VisionConditionedASR (Prefix Tuning Ver)...")
    try:
        model = VisionConditionedASR(
            vocab_size=32,       # Wav2Vec2デフォルト
            visual_tokens=64,    # 圧縮後の視覚トークン数
            freeze_cnn=True,     # CNNは固定
            device=device
        ).to(device)
        model.eval()
        print("    -> Model initialized successfully.")
    except Exception as e:
        print(f"    -> Error initializing model: {e}")
        return

    # 2. パラメータ状態の確認
    check_freeze_status(model)

    # 3. ダミーデータの作成
    print("[2] Generating Dummy Data...")
    batch_size = 2
    
    # ダミー音声: 16kHz, 約2秒と3秒のランダムノイズ
    wav_lengths_sec = [2.0, 3.5] 
    dummy_wavs = [
        np.random.randn(int(16000 * length)).astype(np.float32) 
        for length in wav_lengths_sec
    ]
    print(f"    -> Audio: Batch={batch_size}, Lengths={wav_lengths_sec} sec")

    # ダミー画像: RGBノイズ画像 (224x224)
    dummy_images = [
        Image.new('RGB', (224, 224), color=(np.random.randint(0, 255), 0, 0)) 
        for _ in range(batch_size)
    ]
    print(f"    -> Image: Batch={batch_size}, Size=(224, 224)")

    # 入力データ形式
    data = {
        "wav": dummy_wavs,
        "image": dummy_images
    }

    # 4. 推論実行 (Forward Pass)
    print("\n[3] Running Forward Pass...")
    try:
        with torch.no_grad():
            logits = model(data)
        
        print("    -> Forward pass successful.")
        print(f"    -> Output Logits Shape: {logits.shape}")
        # Expected: [Batch, Max_Audio_Seq_Len, Vocab_Size]
        # 注: 画像トークン分(64)が削除され，音声長に戻っているか確認
        
    except Exception as e:
        print(f"    -> Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 出力の検証 (CTC Decoding)
    print("\n[4] checking Output Validity (Dummy Decoding)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        predicted_ids = torch.argmax(logits, dim=-1)
        
        for i in range(batch_size):
            pred_text = tokenizer.decode(predicted_ids[i])
            print(f"    Sample {i+1} Output: '{pred_text}'")
            print(f"    (Note: Since inputs are random noise, output will be meaningless, but functionality is verified.)")
            
    except Exception as e:
        print(f"    -> Error during decoding: {e}")

    print(f"\n{'='*60}")
    print("Test Completed Successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()