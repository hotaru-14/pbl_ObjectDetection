# オフライン環境で YOLOv8n を CPU でリアルタイム動作させるためのまとめ

## ゴール

- **完全オフライン**で  
- **CPU のみ**を使って  
- **（それなりに）リアルタイム**に  
- **YOLOv8n による物体検出**を行う

現在は Flask のエンドポイントで実装している前提で、構成とチューニング方法をまとめます。

---

## 1. オフライン環境の準備

### 1.1 モデルと依存関係をローカルに固める

オンライン環境（開発マシン）で：

- YOLOv8 のライブラリをインストール：
  - `pip install ultralytics`
- モデルを取得：
  - `yolov8n.pt` をダウンロード  
    （例：`from ultralytics import YOLO; YOLO("yolov8n.pt")` で利用）

そのうえで、以下をオフライン環境にコピーする：

- `yolov8n.pt`（モデルファイル）
- 推論用・Flask 用の Python スクリプト
- `requirements.txt` と必要であれば wheel ファイル群

オフライン側では、ローカルに保存した wheel からインストール：

```bash
pip install --no-index --find-links=./wheels -r requirements.txt
```

### 1.2 オンライン依存がないかの確認

- `from ultralytics import YOLO` を使うだけなら OK
- `model.train()` や hub 連携、オンライン API 呼び出しなどを  
  オフライン環境で行っていないか確認する

---

## 2. Flask と推論処理の分離

### 2.1 なぜ分離するか

Flask のエンドポイント内でそのまま `model.predict()` を呼ぶ構成だと：

- リクエストのたびに CPU を占有し、処理が詰まりやすい
- 複数クライアントからのアクセスでレスポンスが遅くなる
- カメラ映像など「リアルタイム配信」には向かない

### 2.2 推奨構成

1. **バックグラウンドで動く推論スレッド／プロセス**
   - カメラや動画ソースからフレームを取り続ける
   - ループ内で YOLO 推論を実行し、「最新の推論結果」を共有変数に保存

2. **Flask アプリ**
   - `/video` などのエンドポイントで「最新フレーム（検出結果付き）」を配信
   - `/detections` などでバウンディングボックス情報だけ返すことも可能

### 2.3 コード構成イメージ

```python
# detector.py
from ultralytics import YOLO
import cv2
import threading

model = YOLO("yolov8n.pt")

latest_frame = None
latest_lock = threading.Lock()

def detection_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)  # カメラ

    frame_count = 0
    SKIP = 3  # 3フレームに1回だけ推論する例

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 解像度を落として高速化
        frame_small = cv2.resize(frame, (640, 360))

        if frame_count % SKIP == 0:
            results = model.predict(
                frame_small,
                imgsz=640,
                device="cpu",
                conf=0.4,
                iou=0.5,
                verbose=False
            )
            annotated = results[0].plot()
            with latest_lock:
                latest_frame = annotated

        frame_count += 1
```

```python
# app.py
from flask import Flask, Response
import cv2
from detector import latest_frame, latest_lock, detection_loop
import threading

app = Flask(__name__)

@app.route("/video")
def video():
    def gen():
        while True:
            with latest_lock:
                frame = latest_frame
            if frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
```

ポイント：

- モデルは **起動時に1回だけロード**する
- カメラも **ループの外で1回だけオープン**する
- Flask は「最後に処理したフレームを返すだけ」にして軽くする

---

## 3. CPU でも「リアルタイムっぽく」動かす工夫

### 3.1 解像度と FPS の調整

CPU では、入力が重いとすぐに FPS が落ちるため：

- `imgsz` を小さめにする（例：640 → 480 や 416）
- `cv2.VideoCapture` 側の解像度を落とす
- すべてのフレームで推論せず、**N フレームに 1 回**推論する

```python
frame_count = 0
SKIP = 3  # 3フレームに1回推論

if frame_count % SKIP == 0:
    results = model.predict(...)
    # 結果を保存
frame_count += 1
```

### 3.2 モデルパラメータのチューニング

- `yolov8n` はシリーズ中で一番軽量なので、CPU向けに適している
- さらに速度を稼ぐために：
  - `conf` 閾値を上げて弱い検出を捨てる
  - `iou` を調整して NMS の負荷を抑える
  - 小さすぎる物体は無視する方針にする

例：

```python
results = model.predict(
    frame_small,
    imgsz=480,
    conf=0.4,
    iou=0.5,
    device="cpu"
)
```

---

## 4. ランタイムを変えて高速化（ONNX / OpenVINO / 量子化）

さらに CPU を有効活用したい場合は、YOLOv8 モデルを **別ランタイムにエクスポート**すると高速になることが多い。

### 4.1 ONNX + ONNX Runtime

#### エクスポート

オンライン環境で：

```bash
yolo export model=yolov8n.pt format=onnx
```

#### 推論例

```python
import onnxruntime as ort
import numpy as np
import cv2

sess = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

def infer(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    outputs = sess.run(None, {input_name: img})
    # outputs に対して自前で後処理（スケーリング・NMSなど）を行う
```

- ONNX Runtime は CPU 向け最適化がされており、PyTorch より速い場合がある

### 4.2 OpenVINO（Intel CPU に特に有効）

#### エクスポート

```bash
yolo export model=yolov8n.pt format=openvino
```

- エクスポートされた IR（.xml / .bin）を OpenVINO の API で読み込んで推論
- INT8 量子化なども利用可能で、CPU での推論速度がさらに向上する

---

## 5. 完全オフライン構成の整理

- **クラウド・外部 API への依存なし**
  - モデルファイル（`.pt` / `.onnx` / OpenVINO IR）はすべてローカル
  - Python パッケージもローカルの wheel などからインストール
- **Flask は UI / API のフロントだけを担当**
  - 推論はバックグラウンドスレッドまたは別プロセスが担当
- **CPU 最適化のポイント**
  - 解像度を縮小
  - フレーム間引き（すべてのフレームで推論しない）
  - YOLO のパラメータ（`imgsz`, `conf`, `iou`）調整
  - 必要に応じて ONNX / OpenVINO へのエクスポート＋量子化

---

## 6. 既存 Flask 実装を見直すチェックリスト

現在の Flask 実装をそのまま改善したい場合は、以下を確認する：

1. **モデルロード**
   - `model = YOLO("yolov8n.pt")` をアプリ起動時に 1 回だけ呼んでいるか
   - エンドポイントごとに毎回ロードしていないか

2. **デバイス指定**
   - `device="cpu"` を明示しているか

3. **カメラの扱い**
   - エンドポイント内で毎回 `cv2.VideoCapture` を開いていないか
   - 起動時にカメラを開き、ループで使い回しているか

4. **解像度・imgsz**
   - 無駄に大きな解像度で推論していないか
   - `imgsz` を小さくしても許容できるか

---

以上が、「YOLOv8n をオフライン環境で CPU だけでリアルタイム物体検出に使う」ための整理です。

