"""
YOLOv8を使用した物体検出のための前処理と推論関数
このモジュールは、画像の前処理とYOLOv8モデルによる推論を行う機能を提供します。
デフォルトではYOLOv8s (small)モデルを使用し、精度と速度のバランスが良い設定になっています。
"""

import base64
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


def preprocess_image(image_path):
    """
    YOLOv8nに入力するための前処理関数
    
    Args:
        image_path (str): 入力画像のパス（例: 'image/elephant.png'）
    
    Returns:
        numpy.ndarray: 前処理された画像のnumpy配列（RGB形式、uint8型）
    
    処理内容:
    - 画像ファイルを読み込む（PIL.Imageを使用）
    - RGB形式に変換（PILは自動的にRGBで読み込むため、そのまま使用）
    - numpy配列に変換して返す
    """
    # 画像ファイルを読み込む
    # PIL.Imageを使用することで、様々な画像形式（PNG、JPEG等）に対応できる
    image = Image.open(image_path)
    
    # RGB形式に変換（RGBAやグレースケール画像もRGBに統一）
    # YOLOv8nはRGB形式の画像を期待しているため、この変換が必要
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # numpy配列に変換
    # YOLOv8nはnumpy配列を直接受け取ることができるため、この形式で返す
    # shapeは (height, width, channels) の形式になる
    image_array = np.array(image)
    
    return image_array


def preprocess_image_from_base64(base64_string):
    """
    base64エンコードされた画像をYOLOv8nに入力するための前処理関数
    
    Args:
        base64_string (str): base64エンコードされた画像データ（data URI形式の場合は自動的に処理）
    
    Returns:
        numpy.ndarray: 前処理された画像のnumpy配列（RGB形式、uint8型）
    
    処理内容:
    - base64文字列から画像データをデコード
    - PIL.Imageで画像を読み込む
    - RGB形式に変換
    - numpy配列に変換して返す
    """
    # data URI形式の場合は、base64部分のみを抽出
    # 例: "data:image/png;base64,iVBORw0KG..." -> "iVBORw0KG..."
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # base64文字列をバイト列にデコード
    image_bytes = base64.b64decode(base64_string)
    
    # バイト列からPIL Imageオブジェクトを作成
    image = Image.open(BytesIO(image_bytes))
    
    # RGB形式に変換（RGBAやグレースケール画像もRGBに統一）
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # numpy配列に変換
    image_array = np.array(image)
    
    return image_array


# モデルをグローバル変数として保持（複数回の推論で再利用）
# これにより、モデルのロード時間を短縮できる
_model = None

def get_model(model_size='s'):
    """
    YOLOv8モデルを取得する関数（シングルトンパターン）
    
    Args:
        model_size (str): モデルのサイズ 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
                          's' (small)が精度と速度のバランスが良いためデフォルト
    
    Returns:
        YOLO: YOLOv8モデルオブジェクト
    """
    global _model
    
    if _model is None:
        # モデルサイズに応じたファイル名を決定
        # 's' (small)は精度と速度のバランスが良いため推奨
        model_file = f'yolov8{model_size}.pt'
        
        # YOLOv8モデルをロード
        # 初回実行時は自動的にモデルファイルがダウンロードされる
        # 'yolov8s.pt'はYOLOv8のsmall版（精度と速度のバランスが良い）
        print(f"YOLOv8モデルをロード中: {model_file}")
        _model = YOLO(model_file)
        print("モデルのロードが完了しました")
    
    return _model


def run_inference(preprocessed_image, conf_threshold=0.25, model_size='s'):
    """
    YOLOv8で推論した結果を返す関数
    
    Args:
        preprocessed_image (numpy.ndarray): 前処理された画像のnumpy配列
        conf_threshold (float): 信頼度の閾値（0.0-1.0）。この値以上の検出のみを返す
                               デフォルトは0.25。値を上げると精度が上がるが検出数が減る
        model_size (str): モデルのサイズ 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    
    Returns:
        ultralytics.engine.results.Results: YOLOv8の推論結果オブジェクト
    
    処理内容:
    - YOLOv8モデルをロード（初回実行時に自動的にダウンロードされる）
    - 前処理された画像で推論を実行
    - 信頼度閾値を適用して検出結果をフィルタリング
    - 推論結果を返す
    
    推論結果には以下の情報が含まれます:
    - boxes: 検出された物体のバウンディングボックス
    - scores: 各検出の信頼度スコア
    - class_ids: 検出された物体のクラスID
    
    モデルサイズの比較:
    - 'n' (nano): 最も軽量で高速、精度は低め
    - 's' (small): 精度と速度のバランスが良い（推奨）
    - 'm' (medium): より高精度だが処理が遅い
    - 'l' (large): 高精度だが処理が遅い
    - 'x' (xlarge): 最高精度だが処理が最も遅い
    """
    # モデルを取得（シングルトンパターンで再利用）
    model = get_model(model_size)
    
    # 前処理された画像で推論を実行
    # YOLOv8はnumpy配列を直接受け取ることができ、自動的に適切な前処理（リサイズ、正規化等）を行う
    # confパラメータで信頼度閾値を設定（この値以上の検出のみを返す）
    # 推論結果はResultsオブジェクトとして返される
    results = model(preprocessed_image, conf=conf_threshold)
    
    return results


def results_to_json(results, min_score=0.3):
    """
    YOLOv8の推論結果をJSON形式に変換する関数
    
    Args:
        results (ultralytics.engine.results.Results): YOLOv8の推論結果オブジェクト
        min_score (float): 最小信頼度スコア（0.0-1.0）。この値以上の検出のみを返す
                          デフォルトは0.3。値を上げると精度が上がるが検出数が減る
    
    Returns:
        list: 検出結果のリスト。各要素は以下のキーを持つ辞書:
            - box: バウンディングボックスの座標 [x1, y1, x2, y2]
            - score: 信頼度スコア（0.0-1.0）
            - class_id: クラスID（整数）
            - class_name: クラス名（文字列）
    
    処理内容:
    - 推論結果から検出された物体の情報を抽出
    - バウンディングボックス、信頼度、クラスID、クラス名を取得
    - 最小信頼度スコアでフィルタリング
    - JSON形式で返す
    """
    detections = []
    
    # 推論結果から検出情報を取得
    # results[0]は最初の画像の結果（通常は1枚の画像を処理するため）
    result = results[0]
    
    # 検出された物体の数だけループ
    if result.boxes is not None and len(result.boxes) > 0:
        # バウンディングボックスの座標を取得（xyxy形式: 左上と右下の座標）
        boxes = result.boxes.xyxy.cpu().numpy()
        # 信頼度スコアを取得
        scores = result.boxes.conf.cpu().numpy()
        # クラスIDを取得
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        # クラス名を取得（YOLOv8のデフォルトクラス名）
        class_names = result.names
        
        # 各検出結果を辞書形式で保存（最小信頼度スコアでフィルタリング）
        for box, score, class_id in zip(boxes, scores, class_ids):
            # 最小信頼度スコア以上の検出のみを追加
            if float(score) >= min_score:
                detections.append({
                    'box': box.tolist(),  # numpy配列をリストに変換
                    'score': float(score),  # numpyのfloat型をPythonのfloat型に変換
                    'class_id': int(class_id),
                    'class_name': class_names[class_id]  # クラスIDからクラス名を取得
                })
    
    return detections


# 参考URL:
# - Ultralytics YOLOv8公式ドキュメント: https://docs.ultralytics.com/
# - PIL Image ドキュメント: https://pillow.readthedocs.io/
# - NumPy ドキュメント: https://numpy.org/doc/



