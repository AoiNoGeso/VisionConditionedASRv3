import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torchaudio
import os
import numpy as np
from typing import List, Dict, Any


class SpokenCOCODataset(Dataset):
    """
    SpokenCOCOデータセット
    
    音声キャプションと対応する画像のペアを提供します。
    
    Args:
        json_path: データセットのJSONファイルパス
        audio_dir: 音声ファイルのルートディレクトリ
        image_dir: 画像ファイルのルートディレクトリ
        sample_rate: 音声のサンプリングレート（デフォルト: 16000Hz）
        max_audio_length: 音声の最大長（秒）。Noneの場合は制限なし
        validate_files: ファイルの存在確認を行うか
    """
    
    def __init__(
        self, 
        json_path: str, 
        audio_dir: str = None, 
        image_dir: str = None, 
        sample_rate: int = 16000,
        max_audio_length: float = None,
        validate_files: bool = True
    ):
        self.audio_dir = audio_dir if audio_dir is not None else ""
        self.image_dir = image_dir if image_dir is not None else ""
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # JSONファイルの読み込み
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # データの展開とバリデーション
        self.data = []
        
        if validate_files:
            self._load_with_validation(raw_data)
        else:
            self._load_without_validation(raw_data)
        
        if len(self.data) == 0:
            raise ValueError("No valid data found. Please check file paths and data availability.")
    
    def _load_with_validation(self, raw_data: dict):
        """ファイル存在確認付きでデータを読み込む"""
        total_items = 0
        missing_audio = 0
        missing_image = 0
        
        print(f"\n[Dataset] Loading data with file validation...")
        
        for item in raw_data["data"]:
            image_path = item["image"]
            full_image_path = os.path.join(self.image_dir, image_path)
            
            # 画像の存在確認
            if not os.path.exists(full_image_path):
                missing_image += 1
                continue
            
            # 各キャプションについて処理
            for cap in item["captions"]:
                total_items += 1
                wav_path = cap["wav"]
                full_wav_path = os.path.join(self.audio_dir, wav_path)
                
                # 音声の存在確認
                if not os.path.exists(full_wav_path):
                    missing_audio += 1
                    continue
                
                # 両方のファイルが存在する場合のみ追加
                self.data.append({
                    "image": full_image_path,
                    "wav": full_wav_path,
                    "text": cap["text"]
                })
        
        # 統計情報を表示
        valid_items = len(self.data)
        print(f"\n[Dataset] Loading statistics:")
        print(f"  Total items in JSON:     {total_items}")
        print(f"  Missing image files:     {missing_image}")
        print(f"  Missing audio files:     {missing_audio}")
        print(f"  Valid items loaded:      {valid_items}")
        print(f"  Success rate:            {valid_items/total_items*100:.2f}%")
    
    def _load_without_validation(self, raw_data: dict):
        """ファイル存在確認なしでデータを読み込む（高速）"""
        print(f"\n[Dataset] Loading data without file validation (fast mode)...")
        
        for item in raw_data["data"]:
            image_path = item["image"]
            full_image_path = os.path.join(self.image_dir, image_path)
            
            for cap in item["captions"]:
                wav_path = cap["wav"]
                full_wav_path = os.path.join(self.audio_dir, wav_path)
                
                self.data.append({
                    "image": full_image_path,
                    "wav": full_wav_path,
                    "text": cap["text"]
                })
        
        print(f"[Dataset] Loaded {len(self.data)} items")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        インデックスに対応するデータを取得
        
        Returns:
            dict with keys:
                - "wav": Tensor[T] - 音声波形（1次元）
                - "image": PIL.Image - RGB画像
                - "text": str - テキストキャプション
        """
        item = self.data[idx]
        
        try:
            # 音声読み込み
            wav, sr = torchaudio.load(item['wav'])  # (channels, T)
            
            # リサンプリング
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = resampler(wav)
            
            # モノラル化
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # 1次元に変換
            wav = wav.squeeze(0)  # (T,)
            
            # 最大長の制限（指定されている場合）
            if self.max_audio_length is not None:
                max_samples = int(self.max_audio_length * self.sample_rate)
                if wav.size(0) > max_samples:
                    wav = wav[:max_samples]
            
            # 音声が空でないことを確認
            if wav.size(0) == 0:
                raise ValueError(f"Empty audio file: {item['wav']}")
            
            # 画像読み込み
            image = Image.open(item['image']).convert("RGB")
            
            # 画像サイズの確認
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image size: {item['image']}")
            
            # テキスト
            text = item['text']
            
            return {
                "wav": wav,
                "image": image,
                "text": text
            }
        
        except Exception as e:
            print(f"\n[Warning] Error loading item {idx}: {e}")
            print(f"  Audio path: {item['wav']}")
            print(f"  Image path: {item['image']}")
            
            # エラーが発生した場合、次のインデックスを試す
            # 無限ループを防ぐため、データセットサイズの半分まで試行
            for offset in range(1, len(self.data) // 2):
                try:
                    next_idx = (idx + offset) % len(self.data)
                    return self.__getitem__(next_idx)
                except Exception:
                    continue
            
            # それでも失敗する場合はダミーデータを返す
            print(f"[Error] Failed to load any valid data, returning dummy data")
            return self._get_dummy_data()
    
    def _get_dummy_data(self) -> Dict[str, Any]:
        """エラー時のダミーデータを生成"""
        dummy_wav = torch.zeros(16000)  # 1秒の無音
        dummy_image = Image.new('RGB', (224, 224), color='black')
        dummy_text = ""
        
        return {
            "wav": dummy_wav,
            "image": dummy_image,
            "text": dummy_text
        }


def spokenCOCO_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    バッチをまとめるcollate関数
    
    Args:
        batch: list of dict:
            - "wav": Tensor[T]
            - "image": PIL.Image
            - "text": str
    
    Returns:
        dict:
            - "wav": List[np.ndarray] - AutoProcessorに渡す形式
            - "wav_lengths": Tensor[B] - 各音声の長さ
            - "image": List[PIL.Image] - AutoProcessorに渡す形式
            - "text": List[str] - テキストのリスト
    """
    # 音声: Tensorからnumpy配列のリストに変換
    waveforms = [b["wav"].numpy() for b in batch]
    
    # 各音声の長さを記録
    lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    
    # 画像: PIL Imageのリストとして保持
    images = [b["image"] for b in batch]
    
    # テキスト: リストとして保持
    texts = [b["text"] for b in batch]
    
    return {
        "wav": waveforms,           # List[np.ndarray] - Wav2Vec2Processorが期待する形式
        "wav_lengths": lengths,     # Tensor[B] - オプション情報
        "image": images,            # List[PIL.Image] - CLIPProcessorが期待する形式
        "text": texts               # List[str] - tokenizerが期待する形式
    }


def create_dataloader(
    json_path: str,
    audio_dir: str,
    image_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_audio_length: float = None,
    validate_files: bool = True
) -> DataLoader:
    """
    DataLoaderを作成するヘルパー関数
    
    Args:
        json_path: データセットのJSONファイルパス
        audio_dir: 音声ファイルのルートディレクトリ
        image_dir: 画像ファイルのルートディレクトリ
        batch_size: バッチサイズ
        shuffle: データをシャッフルするか
        num_workers: データローダーのワーカー数
        sample_rate: 音声のサンプリングレート
        max_audio_length: 音声の最大長（秒）
        validate_files: ファイルの存在確認を行うか
    
    Returns:
        DataLoader
    """
    dataset = SpokenCOCODataset(
        json_path=json_path,
        audio_dir=audio_dir,
        image_dir=image_dir,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
        validate_files=validate_files
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=spokenCOCO_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader