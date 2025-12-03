import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

class Config:
    # アプリケーション設定
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # モデル設定
    MODEL_PATH = 'yolov8n.pt'  # 初回実行時に自動ダウンロード
    CONFIDENCE_THRESHOLD = 0.5  # 信頼度の閾値（0.0〜1.0）
    
    # アップロード設定
    UPLOAD_FOLDER = 'uploads'  # アップロードファイルの保存先
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # 許可するファイル形式
    
    # 説明文のデフォルト設定
    DEFAULT_DESCRIPTION = "物体が検出されました。"
    
    # LLM API設定
    LLM_API_KEY = os.getenv('LLM_API_KEY', None)  # LLM APIキー（環境変数から取得）
    
    # その他の設定が必要な場合は以下に追加
    # 例: APIキー、接続タイムアウト、リトライ回数など
