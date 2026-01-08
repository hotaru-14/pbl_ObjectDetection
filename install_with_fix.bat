@echo off
REM 必要なライブラリを確実にインストールするバッチファイル
REM Visual Studio関連のエラーを回避する方法を含む

echo ============================================================
echo 必要なライブラリのインストールを開始します
echo ============================================================
echo.

REM 仮想環境を有効化
echo 仮想環境を有効化しています...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo エラー: 仮想環境の有効化に失敗しました
    pause
    exit /b 1
)

echo 仮想環境が有効化されました
echo.

REM pipをアップグレード
echo pipをアップグレードしています...
python -m pip install --upgrade pip setuptools wheel

echo.
echo ============================================================
echo 基本的なライブラリをインストール中...
echo ============================================================
echo.

REM まず基本的なライブラリをインストール
echo [1/5] numpy をインストール中...
python -m pip install numpy --no-cache-dir
if errorlevel 1 (
    echo エラー: numpy のインストールに失敗しました
    pause
    exit /b 1
)

echo [2/5] Pillow をインストール中...
python -m pip install Pillow --no-cache-dir
if errorlevel 1 (
    echo エラー: Pillow のインストールに失敗しました
    pause
    exit /b 1
)

echo [3/5] flask をインストール中...
python -m pip install flask --no-cache-dir
if errorlevel 1 (
    echo エラー: flask のインストールに失敗しました
    pause
    exit /b 1
)

echo [4/5] openai をインストール中...
python -m pip install openai --no-cache-dir
if errorlevel 1 (
    echo エラー: openai のインストールに失敗しました
    pause
    exit /b 1
)

echo [5/5] python-dotenv をインストール中...
python -m pip install python-dotenv --no-cache-dir
if errorlevel 1 (
    echo エラー: python-dotenv のインストールに失敗しました
    pause
    exit /b 1
)

echo.
echo ============================================================
echo OpenCVをインストール中（GUI機能なし版を使用）...
echo ============================================================
echo.

REM opencv-python-headlessを使用（Visual Studio不要）
echo opencv-python-headless をインストール中...
python -m pip install opencv-python-headless --no-cache-dir
if errorlevel 1 (
    echo 警告: opencv-python-headless のインストールに失敗しました
    echo opencv-python を試します...
    python -m pip install opencv-python --no-cache-dir
    if errorlevel 1 (
        echo エラー: OpenCV のインストールに失敗しました
        echo 手動でインストールしてください: pip install opencv-python-headless
    )
)

echo.
echo ============================================================
echo ultralytics をインストール中...
echo ============================================================
echo.

REM ultralyticsをインストール（依存関係も含む）
echo ultralytics をインストール中（時間がかかる場合があります）...
python -m pip install ultralytics --no-cache-dir --no-build-isolation
if errorlevel 1 (
    echo 警告: ultralytics のインストールに失敗しました
    echo 別の方法を試します...
    python -m pip install ultralytics --no-cache-dir
    if errorlevel 1 (
        echo エラー: ultralytics のインストールに失敗しました
        echo.
        echo 対処方法:
        echo 1. インターネット接続を確認してください
        echo 2. 管理者権限で実行してみてください
        echo 3. 手動でインストール: pip install ultralytics
    )
)

echo.
echo ============================================================
echo インストール状況を確認します...
echo ============================================================
echo.

python check_installation.py

echo.
echo ============================================================
echo インストール処理が完了しました
echo ============================================================
echo.
pause






