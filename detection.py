import cv2
import numpy as np
from ultralytics import YOLO
from config import Config
import os
from typing import Optional

def load_model(model_path=Config.MODEL_PATH):
    """YOLOモデルをロードする
    
    Returns:
        YOLO: ロードされたYOLOモデル
    """
    return YOLO(model_path)

def detect_objects(image, model=None):
    """画像から物体を検出する
    
    Args:
        image (numpy.ndarray): 入力画像（BGR形式のnumpy配列）
        model (YOLO, optional): ロード済みのYOLOモデル。
                              Noneの場合は新規にモデルをロードします。
        
    Returns:
        list: 検出結果のリスト。各要素は以下のキーを持つ辞書:
            - 'class' (str): 検出されたクラス名
            - 'confidence' (float): 信頼度スコア (0.0 〜 1.0)
            - 'bbox' (list): バウンディングボックス座標 [x1, y1, x2, y2]
    """
    if model is None:
        model = load_model()
    
    # 物体検出を実行
    results = model(image)
    
    # 検出結果を整形
    detections = []
    for result in results:
        for box in result.boxes:
            # バウンディングボックス座標を整数に変換
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])  # 信頼度
            cls = int(box.cls[0])      # クラスID
            
            # 信頼度が閾値以上の検出結果のみを保持
            if conf >= Config.CONFIDENCE_THRESHOLD:
                detections.append({
                    'class': result.names[cls],  # クラス名を取得
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]    # [x1, y1, x2, y2] 形式のバウンディングボックス
                })
    
    return detections

def preprocess_image(image_path: str) -> np.ndarray:
    """
    image/elephant.pngをyolov8nで推論するための前処理を行う関数
    
    YOLOv8nは画像を行列(numpy配列)で受け取る必要があるため、
    OpenCVを使用して画像を読み込み、numpy配列に変換する。
    
    Args:
        image_path (str): 画像ファイルのパス（例: "image/elephant.png"）
    
    Returns:
        np.ndarray: 画像データ（BGR形式のnumpy配列）
        エラー時はNoneを返す
    
    参考URL:
    - OpenCV imread: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    - NumPy配列: https://numpy.org/doc/stable/reference/arrays.html
    """
    try:
        # OpenCVを使用して画像を読み込む
        # cv2.imread()は画像をBGR形式のnumpy配列として読み込む
        # これがYOLOv8nが処理できる形式
        image = cv2.imread(image_path)
        
        # 画像が存在しない場合、cv2.imread()はNoneを返す
        if image is None:
            print(f"エラー: 画像ファイルが見つかりません: {image_path}")
            return None
        
        # 読み込んだ画像をnumpy配列として返す
        # YOLOv8nは画像のリサイズや正規化を自動で行うため、
        # 基本的な読み込みだけでOK
        return image
        
    except Exception as e:
        # 予期しないエラーが発生した場合のエラーハンドリング
        print(f"エラー: 画像の読み込み中に問題が発生しました: {e}")
        return None


def run_inference(image: np.ndarray, model_path: str = "yolov8n.pt") -> dict:
    """
    yolov8nで推論した結果を返す関数
    
    前処理された画像（numpy配列）を入力として受け取り、
    YOLOv8nモデルで推論を実行して検出結果を返す。
    
    Args:
        image (np.ndarray): 前処理済みの画像データ（numpy配列）
        model_path (str, optional): YOLOv8nモデルのパス。デフォルトは"yolov8n.pt"
            初回実行時に自動でダウンロードされる
    
    Returns:
        dict: 以下の形式の辞書
            {
                "detections": [
                    {
                        "class": "elephant",      # クラス名
                        "confidence": 0.95,       # 信頼度（0.0～1.0）
                        "bbox": [x1, y1, x2, y2]  # バウンディングボックス座標
                    },
                    # ... 他の検出結果
                ],
                "count": 1  # 検出された物体の数
            }
        検出されなかった場合は空のリストを返す
    
    参考URL:
    - Ultralytics YOLO: https://docs.ultralytics.com/
    - YOLOv8推論: https://docs.ultralytics.com/modes/predict/
    """
    try:
        # 入力画像がNoneの場合はエラーを返す
        if image is None:
            return {
                "detections": [],
                "count": 0
            }
        
        # ultralyticsのYOLOクラスを使用してモデルをロード
        # 初回実行時は自動でyolov8n.ptがダウンロードされる
        # device='cpu'を指定することで、CPUで推論を実行することを明示
        # GPUが利用可能でもCPUで動作するように設定
        model = YOLO(model_path)
        
        # 画像に対して推論を実行
        # model(image, device='cpu')を呼ぶことで、CPUで推論を実行
        # YOLOv8nは画像のリサイズや正規化を自動で行うため、前処理は最小限でOK
        # device='cpu'を指定することで、CPU想定で動作することを保証
        results = model(image, device='cpu')
        
        # 推論結果から検出情報を抽出
        detections = []
        
        # results[0]は最初の画像の推論結果（この場合は1枚の画像なので[0]）
        # boxes属性から検出された物体の情報を取得
        boxes = results[0].boxes
        
        # 信頼度の閾値（0.5以上のみを検出結果として採用）
        confidence_threshold = 0.5
        
        # 各検出結果を処理
        for box in boxes:
            # 信頼度を取得
            confidence = float(box.conf[0])
            
            # 信頼度が閾値以上の検出結果のみを採用
            if confidence >= confidence_threshold:
                # クラス名を取得（クラスIDから名前を取得）
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # バウンディングボックスの座標を取得
                # xyxy形式（左上x, 左上y, 右下x, 右下y）で取得
                # CPUで実行しているため、.cpu()は不要だが、互換性のため残している
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # 検出結果を辞書形式で追加
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        # 結果を辞書形式で返す
        return {
            "detections": detections,
            "count": len(detections)
        }
        
    except Exception as e:
        # 予期しないエラーが発生した場合のエラーハンドリング
        print(f"エラー: 推論中に問題が発生しました: {e}")
        return {
            "detections": [],
            "count": 0
        }
