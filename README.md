# 写真で集める図鑑アプリ / Photo Collection Encyclopedia App

FlaskとLLMを活用した図鑑自動生成アプリケーション。

An automatic encyclopedia generation application using Flask and LLM.

---

## 📋 目次 / Table of Contents

- [概要 / Overview](#概要--overview)
- [完了した作業 / Completed Work](#完了した作業--completed-work)
- [機能一覧 / Features](#機能一覧--features)
- [システム構成図 / System Architecture](#システム構成図--system-architecture)
- [必要な環境 / Requirements](#必要な環境--requirements)
- [セットアップ / Setup](#セットアップ--setup)
- [実行方法 / How to Run](#実行方法--how-to-run)
- [プロジェクト構造 / Project Structure](#プロジェクト構造--project-structure)

---

## 概要 / Overview

### 日本語
このプロジェクトは、LLM（大規模言語モデル）を活用して自動的に図鑑を生成するアプリケーションです。場所を入力すると、LLMがWeb検索で情報を収集し、その場所で見つけやすい対象の図鑑を自動生成します。PBLの授業の一環として開発しています。

### English
This project is an application that automatically generates an encyclopedia using LLM (Large Language Model). When you enter a place, LLM collects information through web search and automatically generates an encyclopedia of objects that are easy to find at that location. It is being developed as part of a PBL course.

---

## 完了した作業 / Completed Work

### 日本語
- ✅ **概念図の作成**: Miroで概念図を作成（大まかなもの）
- ✅ **図鑑生成機能の実装**:
  - 場所の記録機能
  - LLMがWeb検索で場所を確認し、図鑑を自動生成

### English
- ✅ **Conceptual Diagram**: Created a conceptual diagram in Miro (rough version)
- ✅ **Encyclopedia Generation Function**:
  - Place recording functionality
  - LLM checks the place via web search and generates encyclopedia automatically

---

## 機能一覧 / Features

### 1. 図鑑生成 / Generate Encyclopedia

#### 日本語
- 生成された図鑑は`Encyclopedia.csv`に保存されます
- LLMがWeb検索を通じて場所の情報を収集し、図鑑の内容を自動生成します

#### English
- Generated encyclopedia is saved to `Encyclopedia.csv`
- LLM collects place information through web search and automatically generates encyclopedia content

---

### 2. カメラ機能 / Camera Function

#### 日本語
**ワークフロー:**
1. 📷 **撮影** - カメラで写真を撮影
2. ⏳ **ローディング** - 写真をLLMに送信し、出力を待機
3. ✅ **図鑑確認** - 生成された図鑑の内容を確認

**UI構成:**
- カメラ画面
- ローディング画面
- 図鑑確認画面

**セキュリティ機能:**
- YOLOを使用して写真に人物が写っていないかチェック
- 入力画像 → 推論 → 人物が含まれているかの判定

**データ保存:**
- 図鑑確認時、現在の位置情報（緯度・経度）を取得
- 位置情報は`LocationInformation.csv`に保存

#### English
**Workflow:**
1. 📷 **Take a picture** - Capture photo with camera
2. ⏳ **Loading** - Send picture to LLM and wait for output
3. ✅ **Confirm encyclopedia** - Review the generated encyclopedia content

**UI Components:**
- Camera screen
- Loading screen
- Encyclopedia confirmation screen

**Security Feature:**
- YOLO checks whether there are people in the picture
- Input picture → Inference → Determine if person is included or not

**Data Storage:**
- When confirming encyclopedia, get current location information ("lat", "lon")
- Location data is saved to `LocationInformation.csv`

---

### 3. 図鑑表示 / Encyclopedia Display

#### 日本語
**UI構成:**
- 図鑑一覧画面
- 詳細表示画面

**機能:**
- 特定の「場所」に対する名前のリストを取得
- `Encyclopedia.csv`から特定の名前に対応する「画像」と「説明」を取得

#### English
**UI Components:**
- Encyclopedia list screen
- Detail display screen

**Functions:**
- Get a list of names for a specific "place"
- Get "image" and "description" of a specific name from `Encyclopedia.csv`

---

### 4. 地図UI / Map UI

#### 日本語
**UI構成:**
- 地図表示
- ピン表示

**機能:**
- `LocationInformation.csv`から位置情報（緯度・経度）を取得し、地図上にピンを配置
- ピンを選択すると、`LocationInformation.csv`から対応する写真を取得して表示

#### English
**UI Components:**
- Map display
- Pin markers

**Functions:**
- Get location information ("lat", "lon") from `LocationInformation.csv` and place pins on the map
- When selecting a pin, get the corresponding picture from `LocationInformation.csv`

---

## システム構成図 / System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│   Camera    │   Loading   │  Encyclopedia│    Map     │  List  │
│    📷      │     ⏳      │     📖      │    🗺️      │   📋   │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Flask Backend (main.py)                     │
├──────────────────┬───────────────────┬───────────────────────────┤
│   YOLO (v8n)     │       LLM         │      Data Management      │
│  Person Check    │  Web Search &     │   CSV Read/Write          │
│                  │  Encyclopedia Gen │                           │
└────────┬─────────┴─────────┬─────────┴─────────┬─────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌───────────────┐  ┌──────────────────────────┐
│   yolov8n.pt    │  │   OpenAI API  │  │         data/            │
│                 │  │               │  │  ├── Encyclopedia.csv    │
│                 │  │               │  │  └── LocationInfo.csv    │
└─────────────────┘  └───────────────┘  └──────────────────────────┘
```

---

## 必要な環境 / Requirements

- Python 3.x
- 仮想環境（推奨） / Virtual environment (recommended)
- Webカメラ / Webcam (for camera function)
- OpenAI API Key

---

## セットアップ / Setup

### 1. リポジトリをクローンまたはダウンロード / Clone or download the repository

### 2. 仮想環境を作成して有効化 / Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Macの場合 / For Linux/Mac
# または / or
.venv\Scripts\activate  # Windowsの場合 / For Windows
```

### 3. 必要なライブラリをインストール / Install required libraries

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定(やらなくていい) / Set environment variables(not doing)

`.env`ファイルを作成し、OpenAI APIキーを設定してください。
Create a `.env` file and set your OpenAI API key.

```
OPENAI_API_KEY=your_api_key_here
```

---

## 実行方法 / How to Run

仮想環境を有効化した状態で、以下のコマンドを実行してください。
Run the following command with the virtual environment activated.

```bash
# Python 2.x の場合 / For Python 2.x
python main.py

# Python 3.x の場合 / For Python 3.x
python3 main.py
```

ブラウザで `http://127.0.0.1:5000` にアクセスしてアプリケーションを使用できます。
Access `http://127.0.0.1:5000` in your browser to use the application.

---

## プロジェクト構造 / Project Structure

```
pbl_ObjectDetection/
├── main.py                 # Flaskアプリケーションのメインファイル
│                           # Main Flask application file
├── requirements.txt        # プロジェクトの依存関係 / Project dependencies
├── yolov8n.pt             # YOLOv8nモデルファイル / YOLOv8n model file
├── .gitignore             # Gitで無視するファイル / Git ignore file
├── .env                   # 環境変数（API キーなど）/ Environment variables
├── README.md              # このファイル / This file
├── data/
│   ├── Encyclopedia.csv   # 図鑑データ / Encyclopedia data
│   └── LocationInformation.csv  # 位置情報データ / Location data
└── templates/
    ├── base.html          # ベーステンプレート / Base template
    ├── camera.html        # カメラページ / Camera page
    ├── encyclopedia.html  # 図鑑表示ページ / Encyclopedia display page
    ├── create_encyclopedia.html  # 図鑑作成ページ / Encyclopedia creation page
    └── map.html           # 地図ページ / Map page
```

---

## API エンドポイント / API Endpoints

### ページルート / Page Routes

| エンドポイント / Endpoint | メソッド / Method | 説明 / Description |
|--------------------------|------------------|-------------------|
| `/` | GET | 図鑑作成ページ / Encyclopedia creation page |
| `/camera` | GET | カメラページ / Camera page |
| `/encyclopedia` | GET | 図鑑一覧ページ / Encyclopedia list page |
| `/map` | GET | 地図ページ / Map page |

### API ルート / API Routes

| エンドポイント / Endpoint | メソッド / Method | 説明 / Description |
|--------------------------|------------------|-------------------|
| `/api/encyclopedia` | POST | 図鑑生成API / Encyclopedia generation API |

---

## データ形式 / Data Format

### Encyclopedia.csv

| カラム / Column | 説明 / Description |
|----------------|-------------------|
| place | 場所名 / Place name |
| name | 項目名 / Item name |
| image | 画像パス / Image path |
| description | 説明文 / Description |

### LocationInformation.csv

| カラム / Column | 説明 / Description |
|----------------|-------------------|
| lat | 緯度 / Latitude |
| lon | 経度 / Longitude |
| image | 画像パス / Image path |

---