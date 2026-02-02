import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from tqdm import tqdm
import time
import csv
from safetensors.torch import load_file

from model import VisionConditionedASRv3
from finetune_noise import PureWav2Vec2ASR
from dataloader import create_dataloader


@dataclass
class BenchmarkConfig:
    """推論速度計測設定"""
    # データセット
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # チェックポイント
    wav2vec2_checkpoint: str = "../../Models/wav2vec2-finetune/pink/epoch_15"
    vasr_checkpoint: str = "../../Models/VisionConditionedASRv3/pink/epoch_15"
    
    # VASR設定
    visual_tokens: int = 64
    use_visual_pos_embed: bool = True
    
    # 計測設定
    num_samples: int = 100  # 計測するサンプル数
    batch_size: int = 1     # リアルタイム推論を想定
    num_workers: int = 0    # 速度計測のため0推奨
    max_audio_length: float = 10.0
    validate_files: bool = False
    
    # デバイス
    # device: str = "cuda:1"
    device: str = "cpu"

    # 結果保存
    save_results: bool = True
    results_dir: str = "results/speed/pink/cpu"


class SpeedBenchmark:
    """推論速度計測クラス"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*70}")
        print("Speed Benchmark Initialization")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Number of samples: {config.num_samples}")
        print(f"Batch size: {config.batch_size}")
        print(f"{'='*70}\n")
        
        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        
        # データローダー
        print("[Setup] Creating dataloader...")
        self.dataloader = create_dataloader(
            json_path=config.val_json,
            audio_dir=config.audio_dir,
            image_dir=config.image_dir,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            max_audio_length=config.max_audio_length,
            validate_files=config.validate_files
        )
        
        # 結果格納
        self.results = {}
    
    def load_wav2vec2_model(self) -> PureWav2Vec2ASR:
        """Wav2Vec2モデルをロード"""
        print("\n[Loading] Wav2Vec2 model...")
        
        model = PureWav2Vec2ASR(
            model_name="facebook/wav2vec2-base-960h",
            device=self.device
        ).to(self.device)
        
        # チェックポイントから読み込み
        checkpoint_dir = self.config.wav2vec2_checkpoint
        if os.path.exists(checkpoint_dir):
            epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
            
            if os.path.exists(model_path):
                state_dict = load_file(model_path, device=str(self.device))
                model.load_state_dict(state_dict)
                print(f"  Loaded from: {model_path}")
            else:
                print(f"  Warning: Checkpoint not found, using pretrained model")
        else:
            print(f"  Warning: Checkpoint directory not found, using pretrained model")
        
        model.eval()
        return model
    
    def load_vasr_model(self) -> VisionConditionedASRv3:
        """VASRモデルをロード"""
        print("\n[Loading] VASR model...")
        
        model = VisionConditionedASRv3(
            vocab_size=None,
            visual_tokens=self.config.visual_tokens,
            dropout=0.1,
            use_visual_pos_embed=self.config.use_visual_pos_embed,
            freeze_vision_encoder=True,
            device=self.device
        ).to(self.device)
        
        # チェックポイントから読み込み
        checkpoint_dir = self.config.vasr_checkpoint
        if os.path.exists(checkpoint_dir):
            epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
            
            if os.path.exists(model_path):
                state_dict = load_file(model_path, device=str(self.device))
                model.load_state_dict(state_dict)
                print(f"  Loaded from: {model_path}")
            else:
                raise FileNotFoundError(f"VASR checkpoint not found: {model_path}")
        else:
            raise FileNotFoundError(f"VASR checkpoint directory not found: {checkpoint_dir}")
        
        model.eval()
        return model
    
    def measure_model_speed(
        self,
        model: nn.Module,
        model_name: str,
        measure_components: bool = False
    ) -> Dict:
        """
        モデルの推論速度を計測
        
        Args:
            model: 計測対象モデル
            model_name: モデル名（"Wav2Vec2" or "VASR"）
            measure_components: コンポーネント別計測を行うか
        
        Returns:
            計測結果の辞書
        """
        print(f"\n{'='*70}")
        print(f"Measuring {model_name} Speed")
        print(f"{'='*70}\n")
        
        total_inference_time = 0.0
        total_audio_duration = 0.0
        sample_count = 0
        
        # コンポーネント別時間（VASR用）
        vision_time = 0.0
        audio_time = 0.0
        encoder_time = 0.0
        
        # ウォームアップ
        print("[Warmup] Running warmup iterations...")
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= 5:  # 5バッチでウォームアップ
                    break
                _ = model(batch)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        print(f"[Benchmark] Measuring speed on {self.config.num_samples} samples...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"{model_name}")):
                if sample_count >= self.config.num_samples:
                    break
                
                try:
                    # 音声長を計算（秒）
                    audio_lengths = batch["wav_lengths"].numpy()  # サンプル数
                    batch_audio_duration = audio_lengths.sum() / 16000.0  # 秒に変換
                    
                    # 推論時間計測
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    
                    if measure_components and model_name == "VASR":
                        # VASRのコンポーネント別計測
                        # Vision Encoder
                        vision_start = time.time()
                        visual_tokens = model.vision_encoder(batch)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        vision_time += time.time() - vision_start
                        
                        # Audio Encoder
                        audio_start = time.time()
                        audio_embeddings, attention_mask = model.audio_encoder(batch)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        audio_time += time.time() - audio_start
                        
                        # Combine and Encoder
                        encoder_start = time.time()
                        combined = torch.cat([visual_tokens, audio_embeddings], dim=1)
                        batch_size = audio_embeddings.size(0)
                        if attention_mask is not None:
                            visual_mask = torch.ones(batch_size, model.visual_tokens, device=combined.device)
                            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
                        else:
                            combined_mask = None
                        encoder_output = model.encoder(combined, attention_mask=combined_mask)
                        if hasattr(encoder_output, 'last_hidden_state'):
                            encoder_hidden = encoder_output.last_hidden_state
                        else:
                            encoder_hidden = encoder_output[0]
                        audio_output = encoder_hidden[:, model.visual_tokens:, :]
                        logits = model.classifier(audio_output)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        encoder_time += time.time() - encoder_start
                    else:
                        # 通常の推論
                        logits = model(batch)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    
                    total_inference_time += inference_time
                    total_audio_duration += batch_audio_duration
                    sample_count += len(batch["wav"])
                
                except Exception as e:
                    print(f"\n[Error] Batch {batch_idx}: {e}")
                    continue
        
        # 統計計算
        avg_inference_time = total_inference_time / sample_count
        avg_audio_duration = total_audio_duration / sample_count
        rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0
        
        results = {
            'model_name': model_name,
            'num_samples': sample_count,
            'total_inference_time': total_inference_time,
            'total_audio_duration': total_audio_duration,
            'avg_inference_time': avg_inference_time,
            'avg_audio_duration': avg_audio_duration,
            'rtf': rtf
        }
        
        if measure_components and model_name == "VASR":
            results.update({
                'vision_time': vision_time,
                'audio_time': audio_time,
                'encoder_time': encoder_time,
                'avg_vision_time': vision_time / sample_count,
                'avg_audio_time': audio_time / sample_count,
                'avg_encoder_time': encoder_time / sample_count
            })
        
        # 結果表示
        print(f"\n{'='*70}")
        print(f"{model_name} Speed Benchmark Results")
        print(f"{'='*70}")
        print(f"Samples:                {sample_count}")
        print(f"Total audio duration:   {total_audio_duration:.2f}s")
        print(f"Total inference time:   {total_inference_time:.2f}s")
        print(f"Avg audio duration:     {avg_audio_duration:.3f}s")
        print(f"Avg inference time:     {avg_inference_time:.3f}s")
        print(f"RTF:                    {rtf:.4f}")
        
        if measure_components and model_name == "VASR":
            print(f"\nComponent Breakdown:")
            print(f"  Vision Encoder:       {vision_time:.2f}s ({vision_time/total_inference_time*100:.1f}%)")
            print(f"  Audio Encoder:        {audio_time:.2f}s ({audio_time/total_inference_time*100:.1f}%)")
            print(f"  Transformer Encoder:  {encoder_time:.2f}s ({encoder_time/total_inference_time*100:.1f}%)")
        
        print(f"{'='*70}\n")
        
        return results
    
    def run_benchmark(self):
        """ベンチマーク実行"""
        print(f"\n{'='*70}")
        print("Starting Speed Benchmark")
        print(f"{'='*70}\n")
        
        # Wav2Vec2計測
        wav2vec2_model = self.load_wav2vec2_model()
        self.results['wav2vec2'] = self.measure_model_speed(
            model=wav2vec2_model,
            model_name="Wav2Vec2",
            measure_components=False
        )
        
        # メモリ解放
        del wav2vec2_model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # VASR計測
        vasr_model = self.load_vasr_model()
        self.results['vasr'] = self.measure_model_speed(
            model=vasr_model,
            model_name="VASR",
            measure_components=True
        )
        
        # メモリ解放
        del vasr_model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 比較表示
        self.print_comparison()
        
        # 結果保存
        if self.config.save_results:
            self.save_results()
    
    def print_comparison(self):
        """比較結果を表示"""
        print(f"\n{'='*70}")
        print("Model Comparison")
        print(f"{'='*70}")
        
        wav2vec2 = self.results['wav2vec2']
        vasr = self.results['vasr']
        
        print(f"\n{'Model':<15} {'Samples':<10} {'Avg Audio':<12} {'Avg Inference':<15} {'RTF':<10}")
        print(f"{'-'*70}")
        print(f"{'Wav2Vec2':<15} {wav2vec2['num_samples']:<10} "
              f"{wav2vec2['avg_audio_duration']:<12.3f} "
              f"{wav2vec2['avg_inference_time']:<15.3f} "
              f"{wav2vec2['rtf']:<10.4f}")
        print(f"{'VASR':<15} {vasr['num_samples']:<10} "
              f"{vasr['avg_audio_duration']:<12.3f} "
              f"{vasr['avg_inference_time']:<15.3f} "
              f"{vasr['rtf']:<10.4f}")
        
        # 比較
        slowdown = vasr['rtf'] / wav2vec2['rtf'] if wav2vec2['rtf'] > 0 else 0
        print(f"\nVASR is {slowdown:.2f}x slower than Wav2Vec2")
        
        print(f"{'='*70}\n")
    
    def save_results(self):
        """結果を保存"""
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # テキストファイル
        txt_file = os.path.join(self.config.results_dir, "speed_benchmark.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Speed Benchmark Results\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Device: {self.device}\n")
            f.write(f"  Batch size: {self.config.batch_size}\n")
            f.write(f"  Number of samples: {self.config.num_samples}\n\n")
            
            for model_key in ['wav2vec2', 'vasr']:
                result = self.results[model_key]
                f.write(f"\n{result['model_name']} Results:\n")
                f.write(f"{'-'*70}\n")
                f.write(f"  Samples:              {result['num_samples']}\n")
                f.write(f"  Total audio duration: {result['total_audio_duration']:.2f}s\n")
                f.write(f"  Total inference time: {result['total_inference_time']:.2f}s\n")
                f.write(f"  Avg audio duration:   {result['avg_audio_duration']:.3f}s\n")
                f.write(f"  Avg inference time:   {result['avg_inference_time']:.3f}s\n")
                f.write(f"  RTF:                  {result['rtf']:.4f}\n")
                
                if 'vision_time' in result:
                    f.write(f"\n  Component Breakdown:\n")
                    f.write(f"    Vision Encoder:       {result['vision_time']:.2f}s "
                           f"({result['vision_time']/result['total_inference_time']*100:.1f}%)\n")
                    f.write(f"    Audio Encoder:        {result['audio_time']:.2f}s "
                           f"({result['audio_time']/result['total_inference_time']*100:.1f}%)\n")
                    f.write(f"    Transformer Encoder:  {result['encoder_time']:.2f}s "
                           f"({result['encoder_time']/result['total_inference_time']*100:.1f}%)\n")
            
            # 比較
            wav2vec2 = self.results['wav2vec2']
            vasr = self.results['vasr']
            slowdown = vasr['rtf'] / wav2vec2['rtf'] if wav2vec2['rtf'] > 0 else 0
            
            f.write(f"\n{'='*70}\n")
            f.write(f"Comparison:\n")
            f.write(f"  VASR is {slowdown:.2f}x slower than Wav2Vec2\n")
            f.write(f"{'='*70}\n")
        
        print(f"[Results] Saved to: {txt_file}")
        
        # CSVファイル
        csv_file = os.path.join(self.config.results_dir, "speed_benchmark.csv")
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Samples', 'Total Audio (s)', 'Total Inference (s)', 
                           'Avg Audio (s)', 'Avg Inference (s)', 'RTF'])
            
            for model_key in ['wav2vec2', 'vasr']:
                result = self.results[model_key]
                writer.writerow([
                    result['model_name'],
                    result['num_samples'],
                    f"{result['total_audio_duration']:.2f}",
                    f"{result['total_inference_time']:.2f}",
                    f"{result['avg_audio_duration']:.3f}",
                    f"{result['avg_inference_time']:.3f}",
                    f"{result['rtf']:.4f}"
                ])
        
        print(f"[Results] CSV saved to: {csv_file}\n")


def main():
    """メイン実行関数"""
    config = BenchmarkConfig()
    
    benchmark = SpeedBenchmark(config)
    benchmark.run_benchmark()
    
    print("\n" + "="*70)
    print("Benchmark Completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()