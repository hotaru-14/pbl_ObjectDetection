from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
from typing import Tuple, Optional

# Import modularized components
from detection import run_inference, preprocess_image
from description import generate_description_with_fallback, build_fallback_description
from config import Config

app = Flask(__name__)

# カメラ機能を実装したメインページ
# videoタグを使用してブラウザからカメラにアクセスし、リアルタイムで映像を表示
@app.route("/")
def index():
    """
    メインページを表示
    
    templates/index.htmlテンプレートを使用してリアルタイムカメラページを表示
    """
    return render_template('index.html')


@app.route("/object-detection", methods=['POST'])
def detect():
    """
    物体検出エンドポイント
    
    クライアントから送信されたbase64エンコードされた画像を受け取り、
    YOLOv8nで物体検出を実行して結果をJSON形式で返す。
    
    参考URL:
    - Flask request: https://flask.palletsprojects.com/en/2.3.x/api/#flask.Request
    - Flask jsonify: https://flask.palletsprojects.com/en/2.3.x/api/#flask.json.jsonify
    """
    try:
        # リクエストからJSONデータを取得
        data = request.get_json()
        
        # 画像データが含まれていない場合はエラーを返す
        if 'image' not in data:
            return jsonify({'error': '画像データが含まれていません'}), 400
        
        # base64エンコードされた画像データを取得
        base64_image = data['image']
        
        # base64文字列をバイナリデータにデコード
        # 参考: https://docs.python.org/ja/3/library/base64.html
        image_bytes = base64.b64decode(base64_image)
        
        # バイナリデータをnumpy配列に変換
        # np.frombuffer()でバイナリデータをnumpy配列に変換
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # OpenCVを使用して画像をデコード
        # cv2.imdecode()でnumpy配列から画像（BGR形式）に変換
        # OpenCVは画像をnumpy配列として読み込むため、YOLOv8nがそのまま処理できる
        # 前処理は不要（YOLOv8nがリサイズや正規化を自動で行う）
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 画像のデコードに失敗した場合
        if image is None:
            return jsonify({'error': '画像のデコードに失敗しました'}), 400
        
        # YOLOv8nで推論を実行
        # OpenCVで読み込んだ画像は既にnumpy配列なので、そのままrun_inference()に渡せる
        # 前処理関数を通す必要はない（OpenCVが既にnumpy形式で返すため）
        result = run_inference(image)
        
        # 結果をJSON形式で返す
        return jsonify(result)
        
    except Exception as e:
        # 予期しないエラーが発生した場合のエラーハンドリング
        print(f"エラー: 物体検出エンドポイントで問題が発生しました: {e}")
        return jsonify({
            'error': 'サーバーエラーが発生しました',
            'message': str(e)
        }), 500


@app.route('/description', methods=['POST'])
def description():
    """
    物体説明エンドポイント
    
    バウンディングボックス選択時のフレームを受け取り、LLMに渡すための
    事前処理を行ったうえで解説文を返す。APIキーが無い場合でもUXを損なわないよう
    フォールバック説明を必ず返す。
    """
    data = {}
    try:
        data = request.get_json() or {}
        if 'image' not in data:
            return jsonify({'error': '画像データが含まれていません'}), 400
        
        base64_image = data['image']
        detection = data.get('detection', {})
        
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        description, source = generate_description_with_fallback(image, detection)
        return jsonify({
            'description': description,
            'source': source
        })
    except Exception as e:
        fallback = build_fallback_description(
            data.get('detection', {}),
            error_message=str(e)
        )
        return jsonify({
            'description': fallback,
            'source': 'fallback',
            'error': str(e)
        }), 500

# Render が期待するポートで待ち受けさせる
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render がくれる PORT 環境変数を使う
    app.run(
        host="0.0.0.0",  # ここが超重要！127.0.0.1 ではなく 0.0.0.0
        port=port,
        debug=False      # 本番なので False 推奨
    )
