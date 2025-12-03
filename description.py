from config import Config
import numpy as np
from typing import Tuple, Optional

def generate_description(detection):
    """検出結果に基づいて説明文を生成する
    
    Args:
        detection (dict): 検出結果の辞書。以下のキーを含む:
            - 'class' (str): 検出された物体のクラス名
            - 'confidence' (float): 信頼度スコア (0.0 〜 1.0)
            - 'bbox' (list): バウンディングボックス座標 [x1, y1, x2, y2]
            
    Returns:
        str: 生成された説明文。検出結果がない場合は「検出された物体はありません。」を返す。
    """
    if not detection:
        return "検出された物体はありません。"
    
    class_name = detection.get('class', '物体')
    confidence = detection.get('confidence', 0) * 100  # パーセンテージに変換
    
    return f"{class_name}を検出しました（信頼度: {confidence:.1f}%）。{Config.DEFAULT_DESCRIPTION}"

def get_detailed_description(detection):
    """より詳細な説明を生成する（将来的に拡張用）
    
    Args:
        detection (dict): 検出結果の辞書（generate_description と同様）
        
    Returns:
        str: 生成された詳細説明文
        
    Note:
        将来的にはLLM連携などを実装して、より詳細な説明を生成できるように拡張予定。
    """
    # 将来的にはLLM連携などを実装
    return generate_description(detection)


def generate_description_with_fallback(image: np.ndarray, detection: dict) -> Tuple[str, str]:
    """
    LLMからの応答を試み、失敗した場合は説明文を自前で生成する。
    こうすることでAPIキーが無い環境でもUIフローを統一できる。
    """
    try:
        description = request_multimodal_caption(image, detection)
        return description, "llm"
    except Exception as error:
        description = build_fallback_description(detection, error_message=str(error))
        return description, "fallback"


def request_multimodal_caption(image: np.ndarray, detection: dict) -> str:
    """
    マルチモーダルLLMへ画像を送信してキャプションを生成する。
    実際のAPI連携は後続のタスクで行えるよう、APIキーの存在のみチェック。
    """
    import os
    
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM APIキーが設定されていません")
    
    # TODO: 実際のLLMリクエスト。未実装のため例外を投げてフォールバックさせる。
    raise NotImplementedError("LLM連携は未実装です")


def build_fallback_description(detection: Optional[dict], error_message: str = "") -> str:
    """
    LLMを使えない場合でもユーザーへ状況を伝えるための簡易説明を合成する。
    既存の検出結果を利用して、それなりに文脈のあるメッセージを返す。
    """
    detection = detection or {}
    label = detection.get("class", "不明な物体")
    confidence = detection.get("confidence")
    bbox = detection.get("bbox", [])
    
    confidence_text = ""
    if confidence is not None:
        confidence_text = f"（信頼度{confidence * 100:.1f}%）"
    
    bbox_text = ""
    if isinstance(bbox, list) and len(bbox) == 4:
        width = max(bbox[2] - bbox[0], 1)
        height = max(bbox[3] - bbox[1], 1)
        bbox_text = f" おおよそ{int(width)}×{int(height)}ピクセルのサイズ感です。"
    
    description = (
        f"{label}{confidence_text}が映っています。"
        f"{bbox_text} LLM APIキーが未設定のため簡易説明を表示しています。"
    )
    
    if error_message:
        description += f" 詳細: {error_message}"
    
    return description
