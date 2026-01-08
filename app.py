from flask import Flask, render_template, request, jsonify
import sys

app = Flask(__name__)

# 必要なライブラリがインストールされているかチェック
# 起動時にエラーを検出して、ユーザーに明確なメッセージを表示するため
def check_dependencies():
    """必要なライブラリがインストールされているか確認する関数"""
    missing_modules = []
    
    try:
        import numpy
    except ImportError:
        missing_modules.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing_modules.append("Pillow")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing_modules.append("ultralytics")
    
    if missing_modules:
        print("\n" + "="*60)
        print("エラー: 必要なライブラリがインストールされていません")
        print("="*60)
        print(f"\n以下のライブラリが不足しています: {', '.join(missing_modules)}")
        print("\n対処方法:")
        print("1. 仮想環境に入っているか確認してください")
        print("   Windows (PowerShell): .\\venv\\Scripts\\Activate.ps1")
        print("   Windows (CMD): venv\\Scripts\\activate.bat")
        print("\n2. 以下のコマンドでライブラリをインストールしてください:")
        print("   pip install -r requirements.txt")
        print("\n3. インストール後、再度アプリケーションを起動してください")
        print("="*60 + "\n")
        sys.exit(1)

# 起動時に依存関係をチェック
check_dependencies()

# 依存関係が確認できた後にインポート
# これにより、エラーメッセージがより明確になる
from yolo_detection import preprocess_image_from_base64, run_inference, results_to_json

# ローカル説明文生成用の関数をインポート
# APIキー不要で動作する、クラス名ベースの説明文生成を使用
from local_description import generate_description_with_context

# GPT-4o-mini用の関数をインポート（オプショナル）
# APIキーが設定されている場合のみ使用可能
try:
    from gpt_detection import generate_description_from_base64
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    print("情報: gpt_detectionモジュールは使用できません。ローカル説明文生成を使用します。")


@app.get("/")
def show_camera():
    """
    カメラUIページを表示するルート

    Flaskのテンプレート機能を使うことで、フロントエンドのHTML/CSS/JSを
    分離して管理しやすくするためこの構成にしている。
    """
    return render_template("camera.html")


@app.post("/detect")
def detect_objects():
    """
    物体検出APIエンドポイント
    
    フロントエンドから送信されたbase64エンコードされた画像を受け取り、
    YOLOv8nで物体検出を実行して結果をJSON形式で返す。
    
    リクエストボディ:
        {
            "image": "data:image/png;base64,..."  # base64エンコードされた画像データ
        }
    
    レスポンス:
        {
            "detections": [
                {
                    "box": [x1, y1, x2, y2],
                    "score": 0.95,
                    "class_id": 0,
                    "class_name": "person"
                },
                ...
            ]
        }
    """
    try:
        # リクエストからJSONデータを取得
        data = request.get_json()
        
        # 画像データが含まれているか確認
        if not data or 'image' not in data:
            return jsonify({'error': '画像データが提供されていません'}), 400
        
        image_base64 = data['image']
        
        # base64エンコードされた画像を前処理
        # フロントエンドから送信された画像をYOLOv8が処理できる形式に変換
        preprocessed_image = preprocess_image_from_base64(image_base64)
        
        # 信頼度閾値を取得（デフォルトは0.3、より高い値にすると精度が上がる）
        conf_threshold = float(data.get('conf_threshold', 0.3))
        model_size = data.get('model_size', 's')  # 's' (small)が精度と速度のバランスが良い
        
        # YOLOv8で推論を実行
        # 前処理された画像で物体検出を行う
        # 's' (small)モデルを使用して精度と速度のバランスを取る
        results = run_inference(preprocessed_image, conf_threshold=conf_threshold, model_size=model_size)
        
        # 推論結果をJSON形式に変換
        # フロントエンドで表示しやすいように辞書形式に変換
        # 最小信頼度スコアも設定（デフォルトは0.3）
        min_score = float(data.get('min_score', 0.3))
        detections = results_to_json(results, min_score=min_score)
        
        # 検出結果をJSON形式で返す
        return jsonify({'detections': detections})
        
    except ImportError as e:
        # インポートエラーの場合は、より詳細なメッセージを返す
        # ユーザーに必要なライブラリのインストールを促す
        error_msg = f"必要なライブラリがインストールされていません: {str(e)}\n"
        error_msg += "仮想環境に入って 'pip install -r requirements.txt' を実行してください。"
        print(f"ImportError: {error_msg}")
        return jsonify({'error': error_msg}), 500
    except Exception as e:
        # その他のエラーが発生した場合はエラーメッセージを返す
        # デバッグ時にエラー内容を確認できるようにする
        print(f"Error in detect_objects: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post("/describe")
def describe_image():
    """
    画像説明生成APIエンドポイント
    
    フロントエンドから送信された検出情報を受け取り、
    ローカル説明文生成（APIキー不要）またはGPT-4o-miniで説明文を生成して返す。
    
    リクエストボディ:
        {
            "class_name": "person",  # 検出された物体のクラス名
            "score": 0.95,  # 検出の信頼度スコア
            "box": [x1, y1, x2, y2],  # バウンディングボックスの座標（オプショナル）
            "image": "data:image/png;base64,...",  # base64エンコードされた画像データ（GPT使用時のみ）
            "use_gpt": false  # GPT-4o-miniを使用するかどうか（デフォルト: false）
        }
    
    レスポンス:
        {
            "description": "生成された説明文"
        }
    """
    try:
        # リクエストからJSONデータを取得
        data = request.get_json()
        
        # クラス名が含まれているか確認
        if not data or 'class_name' not in data:
            return jsonify({'error': 'クラス名が提供されていません'}), 400
        
        class_name = data['class_name']
        score = data.get('score', 1.0)
        box = data.get('box')
        use_gpt = data.get('use_gpt', False)  # デフォルトはローカル説明文生成を使用
        
        # GPT-4o-miniを使用する場合（APIキーが必要）
        if use_gpt and GPT_AVAILABLE:
            if 'image' not in data:
                return jsonify({'error': 'GPT使用時は画像データが必要です'}), 400
            
            image_base64 = data['image']
            if not image_base64 or len(image_base64) < 100:
                return jsonify({'error': '画像データが無効です'}), 400
            
            prompt = data.get('prompt', '画像について小学生にも理解できるように、簡単な言葉を使って3〜4文で説明してください。')
            
            try:
                description = generate_description_from_base64(image_base64, prompt)
                if not description:
                    raise ValueError('説明文が生成されませんでした')
            except Exception as e:
                # GPTでエラーが発生した場合は、ローカル説明文生成にフォールバック
                print(f"GPT error, falling back to local description: {str(e)}")
                description = generate_description_with_context(class_name, score, box)
        else:
            # ローカル説明文生成を使用（APIキー不要）
            # クラス名から事前定義された説明文を生成
            description = generate_description_with_context(class_name, score, box)
        
        # 説明文をJSON形式で返す
        return jsonify({'description': description})
        
    except Exception as e:
        # エラーが発生した場合はエラーメッセージを返す
        error_msg = str(e)
        print(f"Error in describe_image: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

# スクリプトが直接実行された場合にFlaskアプリケーションを起動
# これがないと、python3 app.pyを実行してもサーバーが起動しない
if __name__ == "__main__":
    # デバッグモードを有効にして、コード変更時に自動リロードされるようにする
    # host='0.0.0.0'を指定することで、ローカルネットワークからもアクセス可能にする
    # port=5000はFlaskのデフォルトポート
    print("\n" + "="*50)
    print("Flaskアプリケーションを起動しています...")
    print("="*50)
    print("\nブラウザで以下のURLを開いてください:")
    print("  → http://127.0.0.1:5000/")
    print("  → http://localhost:5000/")
    print("\nサーバーを停止するには Ctrl+C を押してください")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
