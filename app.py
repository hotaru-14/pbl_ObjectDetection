from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from typing import Tuple, Optional

app = Flask(__name__)

# カメラ機能を実装したメインページ
# videoタグを使用してブラウザからカメラにアクセスし、リアルタイムで映像を表示
@app.route("/")
def index():
    # HTMLテンプレートを文字列として定義
    # videoタグを使用することで、ブラウザのネイティブ機能でカメラ映像を表示できる
    html_template = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>リアルタイムカメラ</title>
        <style>
            /* 画面全体表示のためのリセット */
            /* margin: 0 と padding: 0 でブラウザのデフォルトマージンを削除 */
            body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #000;
            }
            /* ビデオコンテナを画面全体に配置 */
            /* position: fixed で画面に固定し、width/height 100%で全画面表示 */
            .video-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
            }
            /* ビデオ要素を画面全体に表示 */
            /* object-fit: cover でアスペクト比を保ちながら画面全体を埋める */
            #video {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            /* バウンディングボックス用のcanvasをビデオの上に重ねる */
            #canvas-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: auto; /* バウンディングボックスをタップできるようにマウスイベントを許可 */
                cursor: pointer; /* クリック可能であることをユーザーに伝える */
            }
            /* 画像キャプチャ用のcanvasは非表示 */
            #canvas-capture {
                display: none;
            }
            /* 解説文を常時表示するパネルを下部に固定 */
            .description-panel {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                width: min(90vw, 600px);
                background-color: rgba(0, 0, 0, 0.75);
                color: #fff;
                padding: 12px 16px;
                border-radius: 10px;
                font-size: 14px;
                line-height: 1.6;
                z-index: 10;
                backdrop-filter: blur(4px); /* 背景と分離して可読性を上げる */
            }
            /* コントロールボタンを画面の右上に絶対配置 */
            /* position: fixed で画面の右上に固定表示し、コンパクトなサイズに */
            .controls {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10;
                display: flex;
                gap: 10px;
            }
            button {
                padding: 8px 16px;
                font-size: 14px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            #startBtn {
                background-color: #4CAF50;
                color: white;
            }
            #startBtn:hover {
                background-color: #45a049;
            }
            #stopBtn {
                background-color: #f44336;
                color: white;
            }
            #stopBtn:hover {
                background-color: #da190b;
            }
            #stopBtn:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            /* ステータス表示をコントロールの下（右上）に配置 */
            .status {
                position: fixed;
                top: 70px;
                right: 20px;
                color: white;
                background-color: rgba(0, 0, 0, 0.7);
                padding: 8px 12px;
                border-radius: 5px;
                z-index: 10;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <!-- videoタグを使用してカメラ映像を表示 -->
        <!-- autoplay属性により、ストリームが開始されると自動的に再生される -->
        <!-- playsinline属性により、モバイルデバイスでもインライン再生が可能 -->
        <div class="video-container">
            <video id="video" autoplay playsinline></video>
            <!-- バウンディングボックスを描画するためのcanvas -->
            <!-- position: absoluteでvideoの上に重ねて表示 -->
            <canvas id="canvas-overlay"></canvas>
        </div>
        <!-- 画像キャプチャ用のcanvas（非表示） -->
        <!-- videoから画像を取得するために使用 -->
        <canvas id="canvas-capture"></canvas>
        
        <!-- コントロールボタンを画面の上部に配置 -->
        <div class="controls">
            <button id="startBtn" onclick="startCamera()">カメラ開始</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>カメラ停止</button>
        </div>
        <!-- ステータス表示をコントロールの下に配置 -->
        <div class="status" id="status">カメラを開始するには「カメラ開始」ボタンをクリックしてください</div>
        <!-- バウンディングボックスをタップした際の解説文を表示 -->
        <div class="description-panel" id="descriptionPanel">
            <span id="descriptionText">バウンディングボックスをタップすると解説文が表示されます</span>
        </div>

        <script>
            let stream = null; // カメラストリームを保持する変数
            let detectionInterval = null; // 自動物体検出のインターバルIDを保持する変数
            let isDetecting = false; // 物体検出の実行中フラグ（重複実行を防ぐため）
            let previousDetections = []; // 前回の検出結果を保持（滑らかなアニメーションのため）
            let currentDetections = []; // 現在の検出結果を保持
            let animationStartTime = null; // アニメーション開始時刻
            let animationFrameId = null; // アニメーションフレームID（requestAnimationFrame用）
            const ANIMATION_DURATION = 1000; // アニメーション時間（1秒、次の検出結果が来るまでの時間）
            let isDescribing = false; // 同時に複数の解説リクエストが走らないよう制御するフラグ
            
            // DOM構築後にバウンディングボックスのタップイベントをセット
            // ここで設定しておくことで、ユーザー操作に即座に反応できる
            document.addEventListener('DOMContentLoaded', () => {
                const canvasOverlay = document.getElementById('canvas-overlay');
                if (!canvasOverlay) {
                    return;
                }
                canvasOverlay.addEventListener('click', handleCanvasClick);
            });
            
            // カメラを開始する関数
            // getUserMedia APIを使用してブラウザからカメラにアクセス
            // 参考: https://developer.mozilla.org/ja/docs/Web/API/MediaDevices/getUserMedia
            async function startCamera() {
                try {
                    // ステータスを更新
                    document.getElementById('status').textContent = 'カメラにアクセス中...';
                    
                    // getUserMedia APIでカメラとマイクにアクセス
                    // video: true でカメラ、audio: false でマイクは無効化
                    // これにより、ブラウザの許可を求めるダイアログが表示される
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: true, 
                        audio: false 
                    });
                    
                    // videoタグのsrcObjectプロパティにストリームを設定
                    // これにより、videoタグにカメラ映像が表示される
                    const videoElement = document.getElementById('video');
                    videoElement.srcObject = stream;
                    
                    // videoのメタデータが読み込まれたら、canvasのサイズを設定
                    // これにより、canvasとvideoのサイズを一致させる
                    videoElement.addEventListener('loadedmetadata', () => {
                        const canvasOverlay = document.getElementById('canvas-overlay');
                        const canvasCapture = document.getElementById('canvas-capture');
                        canvasOverlay.width = videoElement.videoWidth;
                        canvasOverlay.height = videoElement.videoHeight;
                        canvasCapture.width = videoElement.videoWidth;
                        canvasCapture.height = videoElement.videoHeight;
                    });
                    
                    // ボタンの状態を更新
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').textContent = 'カメラが起動しました';
                    
                    // 1秒間隔で自動的に物体検出を実行
                    // setInterval()を使用して、指定した間隔で関数を繰り返し実行
                    // 参考: https://developer.mozilla.org/ja/docs/Web/API/setInterval
                    // 1000ミリ秒（1秒）ごとにdetectObjects()を呼び出す
                    // これにより、リアルタイムで物体検出が行われる
                    detectionInterval = setInterval(detectObjects, 1000);
                    
                } catch (error) {
                    // エラーハンドリング
                    // ユーザーがカメラへのアクセスを拒否した場合や、カメラが利用できない場合など
                    console.error('カメラへのアクセスに失敗しました:', error);
                    document.getElementById('status').textContent = 
                        'エラー: カメラにアクセスできませんでした。' + 
                        (error.name === 'NotAllowedError' ? ' カメラへのアクセスが拒否されました。' : 
                         error.name === 'NotFoundError' ? ' カメラが見つかりませんでした。' : 
                         ' ブラウザがカメラをサポートしていない可能性があります。');
                }
            }
            
            // カメラを停止する関数
            // ストリームの各トラックを停止することで、カメラへのアクセスを解放
            function stopCamera() {
                if (stream) {
                    // 自動物体検出のインターバルをクリア
                    // clearInterval()を使用して、setInterval()で設定した繰り返し処理を停止
                    // 参考: https://developer.mozilla.org/ja/docs/Web/API/clearInterval
                    if (detectionInterval) {
                        clearInterval(detectionInterval);
                        detectionInterval = null;
                    }
                    
                    // ストリーム内のすべてのトラック（この場合はビデオトラック）を停止
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                    
                    // videoタグのsrcObjectをnullに設定して映像をクリア
                    const videoElement = document.getElementById('video');
                    videoElement.srcObject = null;
                    
                    // アニメーションフレームをキャンセル（もしあれば）
                    if (animationFrameId) {
                        cancelAnimationFrame(animationFrameId);
                        animationFrameId = null;
                    }
                    
                    // canvasをクリア
                    const canvasOverlay = document.getElementById('canvas-overlay');
                    const ctx = canvasOverlay.getContext('2d');
                    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
                    
                    // 検出結果をリセット
                    previousDetections = [];
                    currentDetections = [];
                    
                    // ボタンの状態を更新
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('status').textContent = 'カメラが停止しました';
                    // 検出中フラグをリセット
                    isDetecting = false;
                }
            }
            
            // 線形補間関数（lerp: linear interpolation）
            // 前回の値から新しい値へ滑らかに補間するために使用
            // 参考: https://en.wikipedia.org/wiki/Linear_interpolation
            function lerp(start, end, t) {
                return start + (end - start) * t;
            }
            
            // バウンディングボックスを滑らかに描画する関数
            // requestAnimationFrameを使用して、前回の検出結果から新しい検出結果へ滑らかにアニメーション
            // 参考: https://developer.mozilla.org/ja/docs/Web/API/window/requestAnimationFrame
            function animateDetections() {
                const canvasOverlay = document.getElementById('canvas-overlay');
                const ctxOverlay = canvasOverlay.getContext('2d');
                
                // canvasをクリア
                ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
                
                // 現在時刻を取得
                const currentTime = Date.now();
                
                // アニメーションの進行度を計算（0.0～1.0）
                const elapsed = currentTime - animationStartTime;
                const progress = Math.min(elapsed / ANIMATION_DURATION, 1.0);
                
                // イージング関数（ease-in-out）を適用してより滑らかな動きにする
                // 参考: https://easings.net/#easeInOutQuad
                const easedProgress = progress < 0.5 
                    ? 2 * progress * progress 
                    : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                
                // 現在の検出結果と前回の検出結果の数が同じ場合、補間して描画
                if (currentDetections.length > 0 && previousDetections.length === currentDetections.length) {
                    currentDetections.forEach((current, index) => {
                        const previous = previousDetections[index];
                        
                        // バウンディングボックスの座標を補間
                        const x1 = lerp(previous.bbox[0], current.bbox[0], easedProgress);
                        const y1 = lerp(previous.bbox[1], current.bbox[1], easedProgress);
                        const x2 = lerp(previous.bbox[2], current.bbox[2], easedProgress);
                        const y2 = lerp(previous.bbox[3], current.bbox[3], easedProgress);
                        
                        // バウンディングボックスを描画
                        ctxOverlay.strokeStyle = '#00FF00';
                        ctxOverlay.lineWidth = 2;
                        ctxOverlay.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        
                        // クラス名と信頼度を表示（現在の値を表示）
                        const label = `${current.class} ${(current.confidence * 100).toFixed(1)}%`;
                        ctxOverlay.font = '16px Arial';
                        const textMetrics = ctxOverlay.measureText(label);
                        const textWidth = textMetrics.width;
                        const textHeight = 20;
                        
                        // テキストの背景を描画
                        ctxOverlay.fillStyle = 'rgba(0, 255, 0, 0.7)';
                        ctxOverlay.fillRect(x1, y1 - textHeight, textWidth + 10, textHeight);
                        
                        // テキストを描画
                        ctxOverlay.fillStyle = '#000000';
                        ctxOverlay.fillText(label, x1 + 5, y1 - 5);
                    });
                } else if (currentDetections.length > 0) {
                    // 検出結果の数が変わった場合は、補間せずに直接描画
                    currentDetections.forEach((detection) => {
                        const [x1, y1, x2, y2] = detection.bbox;
                        
                        // バウンディングボックスを描画
                        ctxOverlay.strokeStyle = '#00FF00';
                        ctxOverlay.lineWidth = 2;
                        ctxOverlay.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        
                        // クラス名と信頼度を表示
                        const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
                        ctxOverlay.font = '16px Arial';
                        const textMetrics = ctxOverlay.measureText(label);
                        const textWidth = textMetrics.width;
                        const textHeight = 20;
                        
                        // テキストの背景を描画
                        ctxOverlay.fillStyle = 'rgba(0, 255, 0, 0.7)';
                        ctxOverlay.fillRect(x1, y1 - textHeight, textWidth + 10, textHeight);
                        
                        // テキストを描画
                        ctxOverlay.fillStyle = '#000000';
                        ctxOverlay.fillText(label, x1 + 5, y1 - 5);
                    });
                } else if (previousDetections.length > 0) {
                    // 検出結果が0個でも、前回の結果があれば表示を維持
                    previousDetections.forEach((detection) => {
                        const [x1, y1, x2, y2] = detection.bbox;
                        
                        // バウンディングボックスを描画
                        ctxOverlay.strokeStyle = '#00FF00';
                        ctxOverlay.lineWidth = 2;
                        ctxOverlay.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        
                        // クラス名と信頼度を表示
                        const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
                        ctxOverlay.font = '16px Arial';
                        const textMetrics = ctxOverlay.measureText(label);
                        const textWidth = textMetrics.width;
                        const textHeight = 20;
                        
                        // テキストの背景を描画
                        ctxOverlay.fillStyle = 'rgba(0, 255, 0, 0.7)';
                        ctxOverlay.fillRect(x1, y1 - textHeight, textWidth + 10, textHeight);
                        
                        // テキストを描画
                        ctxOverlay.fillStyle = '#000000';
                        ctxOverlay.fillText(label, x1 + 5, y1 - 5);
                    });
                }
                
                // アニメーションが完了していない場合は、次のフレームをリクエスト
                if (progress < 1.0) {
                    animationFrameId = requestAnimationFrame(animateDetections);
                }
            }
            
            // 物体検出を実行する関数
            // videoタグの現在のフレームをcanvasに描画し、base64エンコードしてサーバーに送信
            // 1秒間隔で自動的に呼び出される（リアルタイム物体検出）
            async function detectObjects() {
                // 既に検出中の場合は処理をスキップ（重複実行を防ぐ）
                // これにより、前回の検出が完了する前に次の検出が開始されることを防ぐ
                // サーバーへの負荷を軽減し、エラーを防ぐため
                if (isDetecting) {
                    return;
                }
                
                try {
                    const video = document.getElementById('video');
                    const canvasCapture = document.getElementById('canvas-capture');
                    const canvasOverlay = document.getElementById('canvas-overlay');
                    
                    // カメラが起動していない場合は処理を中断
                    if (!video.srcObject || video.readyState !== video.HAVE_ENOUGH_DATA) {
                        return;
                    }
                    
                    // 検出中フラグを設定
                    isDetecting = true;
                    
                    // canvasのサイズをvideoのサイズに合わせる
                    canvasCapture.width = video.videoWidth;
                    canvasCapture.height = video.videoHeight;
                    canvasOverlay.width = video.videoWidth;
                    canvasOverlay.height = video.videoHeight;
                    
                    // videoの現在のフレームをcanvasに描画
                    // drawImage()を使用してvideo要素の現在の画像をcanvasにコピー
                    const ctxCapture = canvasCapture.getContext('2d');
                    ctxCapture.drawImage(video, 0, 0, canvasCapture.width, canvasCapture.height);
                    
                    // canvasからbase64エンコードされた画像データを取得
                    // toDataURL()でPNG形式のbase64文字列を取得
                    // 参考: https://developer.mozilla.org/ja/docs/Web/API/HTMLCanvasElement/toDataURL
                    const imageData = canvasCapture.toDataURL('image/png');
                    
                    // base64のプレフィックス（data:image/png;base64,）を除去
                    const base64Data = imageData.split(',')[1];
                    
                    // サーバーに画像データを送信
                    // fetch APIを使用してPOSTリクエストを送信
                    // 参考: https://developer.mozilla.org/ja/docs/Web/API/Fetch_API
                    const response = await fetch('/object-detection', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ image: base64Data })
                    });
                    
                    // サーバーからの応答をJSON形式で取得
                    const result = await response.json();
                    
                    // 前回の検出結果を保存（滑らかなアニメーションのため）
                    // 現在の検出結果を前回の検出結果としてコピー
                    previousDetections = currentDetections.map(detection => ({
                        class: detection.class,
                        confidence: detection.confidence,
                        bbox: [...detection.bbox] // 配列をコピー
                    }));
                    
                    // 新しい検出結果を現在の検出結果として保存
                    currentDetections = result.detections.map(detection => ({
                        class: detection.class,
                        confidence: detection.confidence,
                        bbox: [...detection.bbox] // 配列をコピー
                    }));
                    
                    // アニメーションを開始
                    // 既存のアニメーションフレームをキャンセル（もしあれば）
                    if (animationFrameId) {
                        cancelAnimationFrame(animationFrameId);
                    }
                    
                    // アニメーション開始時刻を記録
                    animationStartTime = Date.now();
                    
                    // アニメーションループを開始
                    // requestAnimationFrameを使用して滑らかに描画
                    animationFrameId = requestAnimationFrame(animateDetections);
                    
                    // ステータスを更新
                    if (result.count > 0) {
                        document.getElementById('status').textContent = 
                            `${result.count}個の物体を検出`;
                    } else {
                        document.getElementById('status').textContent = '物体なし';
                    }
                    
                } catch (error) {
                    // エラーハンドリング
                    console.error('物体検出に失敗しました:', error);
                    document.getElementById('status').textContent = 
                        '物体検出エラー: ' + error.message;
                } finally {
                    // 検出中フラグをリセット
                    // finallyブロックを使用することで、エラーが発生しても必ずフラグをリセット
                    // これにより、次の検出が正常に実行できるようになる
                    isDetecting = false;
                }
            }
            
            // ページが閉じられる際にカメラを停止
            // これにより、カメラリソースが適切に解放される
            window.addEventListener('beforeunload', stopCamera);
            
            // カメラの現在フレームをbase64として取得
            // toDataURLを活用しサーバー側で再エンコードしなくて済むようにする
            function captureFrameAsBase64() {
                const video = document.getElementById('video');
                const canvasCapture = document.getElementById('canvas-capture');
                
                if (!video.srcObject || video.readyState !== video.HAVE_ENOUGH_DATA) {
                    return null;
                }
                
                canvasCapture.width = video.videoWidth;
                canvasCapture.height = video.videoHeight;
                const ctxCapture = canvasCapture.getContext('2d');
                ctxCapture.drawImage(video, 0, 0, canvasCapture.width, canvasCapture.height);
                const imageData = canvasCapture.toDataURL('image/png');
                return imageData.split(',')[1];
            }
            
            // 描画済みバウンディングボックスとタップ座標の突き合わせ
            // マルチメディアUIではクリック点をキャンバス座標に変換する必要がある
            function handleCanvasClick(event) {
                if (currentDetections.length === 0) {
                    updateDescription('検出結果がないため解説を生成できません');
                    return;
                }
                
                const canvas = event.target;
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const x = (event.clientX - rect.left) * scaleX;
                const y = (event.clientY - rect.top) * scaleY;
                
                const tappedDetection = currentDetections.find((detection) => {
                    const [x1, y1, x2, y2] = detection.bbox;
                    return x >= x1 && x <= x2 && y >= y1 && y <= y2;
                });
                
                if (!tappedDetection) {
                    updateDescription('バウンディングボックス内をタップしてください');
                    return;
                }
                
                describeDetection(tappedDetection);
            }
            
            // サーバーへフレームと検出情報を送り、LLM解説をリクエスト
            // APIキー未設定時もUXを損なわないためにここで状態管理する
            async function describeDetection(detection) {
                if (isDescribing) {
                    return;
                }
                
                const base64Frame = captureFrameAsBase64();
                if (!base64Frame) {
                    updateDescription('画像を取得できませんでした');
                    return;
                }
                
                isDescribing = true;
                updateDescription('解説文を生成しています...');
                
                try {
                    const response = await fetch('/describe', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            image: base64Frame,
                            detection
                        })
                    });
                    
                    const result = await response.json();
                    if (response.ok && result.description) {
                        updateDescription(result.description);
                    } else if (result.description) {
                        updateDescription(result.description);
                    } else {
                        updateDescription('解説文の取得に失敗しました');
                    }
                } catch (error) {
                    console.error('解説生成に失敗しました:', error);
                    updateDescription('解説文の取得に失敗しました');
                } finally {
                    isDescribing = false;
                }
            }
            
            // 解説テキストの更新を1か所で管理するヘルパー
            function updateDescription(message) {
                const descriptionElement = document.getElementById('descriptionText');
                descriptionElement.textContent = message;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


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


@app.route('/describe', methods=['POST'])
def describe():
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


# スクリプトが直接実行された場合にFlaskアプリケーションを起動
# これがないと、python3 app.pyを実行してもサーバーが起動しない
if __name__ == "__main__":
    app.run(debug=True)
