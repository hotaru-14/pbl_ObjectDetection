# リアルタイム物体検出アプリ

FlaskとYOLOv8を活用したリアルタイム物体検出アプリケーション。Webカメラから取得した映像をリアルタイムで物体検出し、検出された物体をクリックすると、子供向けの分かりやすい説明文を表示します。

## 主な機能

- **リアルタイム物体検出**: Webカメラから取得した映像をリアルタイムで物体検出
- **高精度検出**: YOLOv8sモデルを使用した高精度な物体検出
- **子供向け説明文**: 検出された物体をクリックすると、漢字にひらがなが振られた分かりやすい説明文を表示
- **APIキー不要**: ローカル説明文生成機能により、OpenAI APIキーなしで動作

## 必要な環境

- Python 3.8以上
- Webカメラ（または内蔵カメラ）
- 仮想環境（推奨）

## セットアップ

### 1. リポジトリをクローンまたはダウンロード

```bash
git clone <repository-url>
cd pbl_ObjectDetection
```

### 2. 仮想環境を作成して有効化

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (コマンドプロンプト):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 必要なライブラリをインストール

#### 方法1: 自動インストールスクリプトを使用（推奨）

**Windows:**
`install_with_fix.bat` をダブルクリックして実行

#### 方法2: 手動でインストール

```bash
pip install -r requirements.txt
```

インストールが完了したら、以下で確認：
```bash
python check_installation.py
```

## 実行方法

仮想環境を有効化した状態で、以下のコマンドを実行：

```bash
python app.py
```

または

```bash
python3 app.py
```

ブラウザで `http://127.0.0.1:5000` または `http://localhost:5000` にアクセス

## 使用方法

1. **カメラを開始**: 「カメラを開始」ボタンをクリック
   - ブラウザからカメラへのアクセス許可が必要です

2. **物体検出を開始**: 「物体検出を開始」ボタンをクリック
   - 検出された物体に緑色のバウンディングボックスが表示されます

3. **説明文を表示**: 緑色のバウンディングボックスをクリック
   - 画面下部に、その物体の説明文が表示されます
   - 説明文は漢字にひらがなが振られており、子供にも分かりやすくなっています

## プロジェクト構造

```
pbl_ObjectDetection/
├── app.py                      # Flaskアプリケーションのメインファイル
├── yolo_detection.py           # YOLOv8物体検出機能
├── local_description.py         # ローカル説明文生成（APIキー不要）
├── gpt_detection.py            # GPT-4o-mini説明文生成（オプション）
├── check_installation.py       # ライブラリインストール確認スクリプト
├── requirements.txt            # プロジェクトの依存関係
├── install_with_fix.bat       # 自動インストールスクリプト（Windows）
├── .gitignore                  # Gitで無視するファイル・ディレクトリ
├── README.md                   # このファイル
├── image/                      # 画像ファイル用ディレクトリ
│   └── elephant.png
├── static/                     # 静的ファイル（CSS等）
│   └── styles.css
└── templates/                  # HTMLテンプレート
    └── camera.html
```

## 技術スタック

- **バックエンド**: Flask
- **物体検出**: YOLOv8s (Ultralytics)
- **画像処理**: PIL (Pillow), NumPy
- **フロントエンド**: HTML5, CSS3, JavaScript (Vanilla)

## トラブルシューティング

### カメラが起動しない

- ブラウザの設定でカメラへのアクセス許可を確認してください
- HTTPS環境での実行を推奨します（localhostは通常問題ありません）

### 物体検出が動作しない

- 必要なライブラリがインストールされているか確認：
  ```bash
  python check_installation.py
  ```
- 初回実行時、YOLOv8sモデルが自動的にダウンロードされます（約22MB）

### 説明文が表示されない

- バウンディングボックスをクリックしているか確認してください
- ブラウザのコンソール（F12キー）でエラーを確認してください

### ライブラリのインストールエラー

- Visual Studio関連のエラーが出る場合：
  - `install_with_fix.bat` を使用してください
  - または、`opencv-python-headless` を個別にインストール：
    ```bash
    pip install opencv-python-headless --no-cache-dir
    ```

## 開発状況

- ✅ リアルタイム物体検出機能
- ✅ 子供向け説明文表示機能
- ✅ バウンディングボックスクリック機能
- ✅ 高精度検出（YOLOv8sモデル）
- ✅ APIキー不要のローカル説明文生成

## ライセンス

このプロジェクトはPBL授業の一環として作成されました。

## 参考URL

- [Ultralytics YOLOv8公式ドキュメント](https://docs.ultralytics.com/)
- [Flask公式ドキュメント](https://flask.palletsprojects.com/)
- [YOLOv8 COCOデータセット](https://cocodataset.org/)
