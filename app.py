import argparse  # コマンドライン引数で動作モードを切り替えるために使用
import cv2       # カメラ映像の取得・表示のために使用
from ultralytics import YOLO  # YOLOv8nモデルを用いた物体検出のために使用
import time  # FPS計算用


def run_object_detection(show_fps: bool = True, confidence_threshold: float = 0.5) -> None:
    """YOLOv8n を用いたリアルタイム物体検出カメラ機能。
    
    カメラからの映像をリアルタイムで処理し、物体検出結果を表示します。
    FPS表示と信頼度スコアの閾値設定が可能です。

    Args:
        show_fps (bool, optional): FPSを表示するかどうか. デフォルトは True.
        confidence_threshold (float, optional): 検出の信頼度の閾値 (0.0 ～ 1.0). デフォルトは 0.5.
    
    参考:
        - Ultralytics YOLO Python Usage
          https://docs.ultralytics.com/usage/python/
        - Ultralytics YOLOv8 モデル概要
          https://docs.ultralytics.com/models/yolov8/
    """
    # モデルの読み込み
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print("YOLOモデル 'yolov8n.pt' の読み込みに失敗しました。ファイルの有無を確認してください。")
        print(f"詳細: {e}")
        return

    # カメラデバイスをオープン
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("カメラが見つかりません。")
        return

    print("物体検出カメラを起動しました。")
    print("終了するには 'q' キーを押してください。")
    print("モード切り替え: 's' キーで表示/非表示切り替え")
    print("信頼度閾値: '+' キーで上げる, '-' キーで下げる")

    # FPS計測用の変数
    prev_time = 0
    current_fps = 0
    
    # 表示モード (True: 検出結果表示, False: 通常カメラ表示)
    show_detection = True
    
    while True:
        # フレームの取得
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。終了します。")
            break
            
        # 現在時刻を取得（FPS計算用）
        current_time = time.time()
        
        # 物体検出モードの場合
        if show_detection:
            # 推論を実行
            results = model(frame, verbose=False, conf=confidence_threshold)
            
            # 検出結果をフレームに描画
            frame = results[0].plot()
            
            # 信頼度閾値を表示
            cv2.putText(frame, f'Confidence: {confidence_threshold:.2f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPSを計算して表示
        if show_fps:
            current_fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time
            cv2.putText(frame, f'FPS: {current_fps:.1f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 表示モードを表示
        mode_text = "検出モード" if show_detection else "通常モード"
        cv2.putText(frame, f'Mode: {mode_text}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # フレームを表示
        cv2.imshow('リアルタイム物体検出', frame)
        
        # キー入力を処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' キーで終了
            break
        elif key == ord('s'):  # 's' キーで表示モード切り替え
            show_detection = not show_detection
        elif key == ord('+') and confidence_threshold < 0.9:  # 信頼度閾値を上げる
            confidence_threshold += 0.1
        elif key == ord('-') and confidence_threshold > 0.1:  # 信頼度閾値を下げる
            confidence_threshold -= 0.1
    
    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。
    
    以下のオプションをサポート:
    - `--no-fps`: FPS表示を無効化
    - `--conf`: 物体検出の信頼度閾値 (デフォルト: 0.5)
    
    参考:
        - Python argparse チュートリアル
          https://docs.python.org/3/howto/argparse.html
    """
    parser = argparse.ArgumentParser(
        description="リアルタイム物体検出カメラアプリケーション"
    )
    
    parser.add_argument(
        "--no-fps",
        action="store_false",
        dest="show_fps",
        help="FPS表示を無効化"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="物体検出の信頼度閾値 (0.0 ～ 1.0)",
        metavar="THRESHOLD"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # メインエントリポイント
    args = parse_args()
    
    # 信頼度閾値が有効な範囲内かチェック
    args.conf = max(0.0, min(1.0, args.conf))  # 0.0 ～ 1.0 の範囲に収める
    
    try:
        # 物体検出機能を実行
        run_object_detection(show_fps=args.show_fps, confidence_threshold=args.conf)
    except KeyboardInterrupt:
        print("\nアプリケーションを終了します。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        # 確実にリソースを解放
        cv2.destroyAllWindows()
