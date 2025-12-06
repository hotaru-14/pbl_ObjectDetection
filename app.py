import argparse  # コマンドライン引数で動作モード(simple / yolo)を切り替えるために使用
import cv2       # カメラ映像の取得・表示のために使用
from ultralytics import YOLO  # YOLOv8nモデルを用いた物体検出のために使用


def simple_camera() -> None:
    """OpenCVのみを用いたシンプルなカメラ表示機能。

    GUIライブラリ(Tkinter等)は使わず、OpenCVの標準ウィンドウ機能だけで
    カメラ映像をリアルタイム表示する最小限の実装にしている。

    参考:
        - OpenCV Python VideoCapture チュートリアル
          https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
        - OpenCV HighGUI (imshow, waitKey)
          https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
    """

    # 0 は「デフォルトのカメラデバイス」を表す
    # 一般的なPCではノートPC内蔵カメラや最初に接続されたUSBカメラが 0 になることが多い
    cap = cv2.VideoCapture(0)

    # カメラが取得できなかった場合は、無限ループに入らないよう即座に終了する
    if not cap.isOpened():
        print("カメラが見つかりません。")
        return

    print("[Simple] 終了するには 'q' キーを押してください。")

    while True:
        # 1フレームずつカメラから画像を取得する
        # ret が False の場合はカメラストリームの終了やエラーを意味する
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。終了します。")
            break

        # 取得したフレームをそのままウィンドウに表示
        # ウィンドウ名はわかりやすく 'Simple Offline Camera' としている
        cv2.imshow('Simple Offline Camera', frame)

        # waitKey(1) で約1ms待機しつつ、キーボード入力を確認する
        # 'q' キーが押されたらループを抜け、リソースを解放して終了する
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ループを抜けたらカメラリソースとウィンドウを必ず解放する
    # これを行わないとカメラが別アプリから利用できなくなる場合があるため
    cap.release()
    cv2.destroyAllWindows()


def yolo_camera() -> None:
    """YOLOv8n を用いたリアルタイム物体検出カメラ機能。

    - `yolov8n.pt` を同一ディレクトリに配置しておけば完全オフラインで動作可能
    - Ultralytics YOLO の Python API を利用して推論し、結果をフレームに描画して表示する

    参考:
        - Ultralytics YOLO Python Usage
          https://docs.ultralytics.com/usage/python/
        - Ultralytics YOLOv8 モデル概要
          https://docs.ultralytics.com/models/yolov8/
    """

    # モデルの読み込み
    # 文字列 'yolov8n.pt' を指定することで、カレントディレクトリのファイルを読み込む。
    # ファイルが存在しない場合は Ultralytics が自動的にダウンロードを試みるが、
    # オフライン環境を想定しているため、事前に配置しておくことを前提としている。
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        # モデルが見つからない / 読み込めない場合は、スタックトレースではなく
        # ユーザーが原因を推測しやすいメッセージのみを出して終了する
        print("YOLOモデル 'yolov8n.pt' の読み込みに失敗しました。ファイルの有無を確認してください。")
        print(f"詳細: {e}")
        return

    # カメラデバイスをオープン
    cap = cv2.VideoCapture(0)

    # カメラが開けなかった場合は、物体検出処理に進んでも意味がないので直ちに終了
    if not cap.isOpened():
        print("カメラが見つかりません。")
        return

    print("[YOLO] 終了するには 'q' キーを押してください。")

    while True:
        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。終了します。")
            break

        # YOLOv8n による物体検出を実行
        # verbose=False にすることで、毎フレームのログ出力を抑制し、
        # コンソール出力が大量にならないようにしている
        results = model(frame, verbose=False)

        # Ultralytics の Result オブジェクトには plot() メソッドが用意されており、
        # 検出結果(バウンディングボックス、ラベル、スコア)を描画済みの画像を返してくれる
        annotated_frame = results[0].plot()

        # 推論結果を描画したフレームを表示
        cv2.imshow('YOLOv8 Offline Detection', annotated_frame)

        # 'q' キーが押されたら推論ループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 使用後は必ずカメラとウィンドウリソースを解放
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析して、実行モードを指定できるようにする。

    - `--mode simple` : OpenCVのみのシンプルカメラ表示
    - `--mode yolo`   : YOLOv8nを用いた物体検出カメラ

    モードを引数として渡せるようにすることで、
    同じスクリプトから用途に応じて動作を切り替えられるようにしている。

    参考:
        - Python argparse チュートリアル
          https://docs.python.org/3/howto/argparse.html
    """

    parser = argparse.ArgumentParser(
        description="オフライン物体検出カメラアプリケーション (simple / yolo)"
    )

    parser.add_argument(
        "--mode",
        choices=["simple", "yolo"],
        default="simple",
        help="実行モードを指定 (simple: シンプルカメラ, yolo: YOLOv8n物体検出)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # メインエントリポイント
    # コマンドライン引数からモードを取得し、対応する機能を呼び出す。
    args = parse_args()

    if args.mode == "simple":
        simple_camera()
    elif args.mode == "yolo":
        yolo_camera()
