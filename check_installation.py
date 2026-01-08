"""
必要なライブラリがインストールされているか確認するスクリプト
このスクリプトを実行することで、不足しているライブラリを特定できます。
"""

import sys

def check_module(module_name, import_name=None):
    """モジュールがインストールされているか確認する関数"""
    if import_name is None:
        import_name = module_name
    
    try:
        __import__(import_name)
        print(f"✓ {module_name} はインストールされています")
        return True
    except ImportError:
        print(f"✗ {module_name} がインストールされていません")
        return False

def main():
    """メイン関数"""
    print("="*60)
    print("必要なライブラリのインストール状況を確認しています...")
    print("="*60)
    print()
    
    # Pythonのバージョンとパスを表示
    print(f"Pythonバージョン: {sys.version}")
    print(f"Pythonパス: {sys.executable}")
    print()
    
    # 必要なライブラリをチェック
    # opencv-python-headlessもcv2としてインポートできるため、cv2でチェック
    modules = [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("ultralytics", "ultralytics"),
        ("flask", "flask"),
        ("opencv-python-headless", "cv2"),  # opencv-python-headlessもcv2として使用可能
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
    ]
    
    missing = []
    installed = []
    
    for package_name, import_name in modules:
        if check_module(package_name, import_name):
            installed.append(package_name)
        else:
            missing.append(package_name)
    
    print()
    print("="*60)
    if missing:
        print(f"不足しているライブラリ: {', '.join(missing)}")
        print()
        print("以下のコマンドで個別にインストールしてください:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print()
        print("または、一括でインストール:")
        print("  pip install -r requirements.txt")
    else:
        print("すべてのライブラリがインストールされています！")
    print("="*60)

if __name__ == "__main__":
    main()

