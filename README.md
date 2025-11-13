# リアルタイム物体検出アプリ

FlaskとYOLOv8nを活用したリアルタイム物体検出アプリケーションのプロジェクト。

## 概要

このプロジェクトは、Webカメラや画像を使用してリアルタイムで物体検出を行うアプリケーションを開発することが目的。
PBLの授業の一貫です。

## 必要な環境

- Python
- 仮想環境（推奨）

## セットアップ

1. リポジトリをクローンまたはダウンロード

2. 仮想環境を作成して有効化：
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Macの場合
# または
.venv\Scripts\activate  # Windowsの場合
```

3. 必要なライブラリをインストールします：
```bash
pip install -r requirements.txt
```

## 実行方法

仮想環境を有効化した状態で、以下のコマンドを実行：
pythonがv2の人は,pythonコマンドをv3の人はpython3でコマンドを実行
```bash
python app.py
python3 app.py
```

ブラウザで `http://127.0.0.1:5000` にアクセスしてアプリケーションを使用できる

## プロジェクト構造

```
pbl_ObjectDetection/
├── app.py              # Flaskアプリケーションのメインファイル
├── requirements.txt    # プロジェクトの依存関係
├── .gitignore         # Gitで無視するファイル・ディレクトリ
├── image/             # 画像ファイル用ディレクトリ
└── output/            # 出力ファイル用ディレクトリ
```

## 開発状況

Flaskアプリケーションの基本的な構造が完了
今後、YOLOv8nを使用した物体検出機能の実装
