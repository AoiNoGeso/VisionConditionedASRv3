# VisionConditionedASRv3 (PrefixVASR)

視覚情報（画像）をPrefix Tuningの手法を用いて音声認識モデル（Wav2Vec2）に統合し，雑音環境下での認識精度向上を目指したマルチモーダルASRシステムの実験実装です．

## 概要
* **Vision Encoder**: DINOv2 (Frozen) + Adapter
* **Audio Encoder**: Wav2Vec2 (Feature Extractor Frozen)
* **Fusion**: 視覚トークンを音声埋め込みの先頭に結合（Prefix）してTransformerに入力
* **Dataset**: SpokenCOCO

## ファイル構成とスクリプト説明

`src/` ディレクトリ内の主要スクリプトの役割は以下の通りです．

### データ・モデル定義
* **`dataloader.py`**
    * SpokenCOCOデータセット（音声・画像・キャプション）を読み込むためのDatasetクラスおよびCollate関数を定義しています．
* **`model.py`**
    * **`VisionConditionedASRv3`**: DINOv2とWav2Vec2を統合したメインモデルの定義．
    * **`VisualAdapter`**: 画像特徴量を固定長のVisual Tokensに変換するアダプター．
    * **`PureWav2Vec2ASR`**: 比較実験用の音声のみのベースラインモデル．

### 学習
* **`train.py`**
    * 提案モデル（VisionConditionedASRv3）の学習スクリプト．
    * WandBによるログ管理，Mixed Precision (AMP) 学習，ノイズ付加（White/Pink/Babble）に対応しています．
* **`finetune_noise.py`**
    * ベースラインモデル（音声のみ）のファインチューニング用スクリプト．提案手法との公平な比較のために使用します．

### 評価・テスト
* **`test.py`**
    * 提案モデル（画像あり/なし）とベースラインモデルのWER（単語誤り率）を一括で評価・比較するスクリプト．
* **`new_test.py`**
    * `test.py` の改良版．評価対象モデル（baseline, vision, novision）を引数で柔軟に選択可能です．
* **`unrelated_test.py`**
    * 「無関係な画像」を入力した場合と「画像なし（Zero Vision）」の場合を比較し，視覚情報の意味的有用性を検証するスクリプト．

### 分析・ベンチマーク
* **`attention_analysis.py`**
    * Transformer内部のAttention重みを抽出し，ノイズレベル（SNR）に応じた視覚情報への依存度（Attention Ratio）を可視化・分析します．
* **`benchmark_speed.py`**
    * モデルの推論速度（RTF: Real Time Factor）を計測し，視覚エンコーダ追加によるオーバーヘッドを算出します．

## セットアップ

### 必要ライブラリのインストール
```bash
pip install -r requirements.txt
# または
pip install .