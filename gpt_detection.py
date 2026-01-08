"""
GPT-4o-miniを使用した画像説明生成のための前処理と推論関数
このモジュールは、画像の前処理とGPT-4o-miniモデルによる説明文生成を行う機能を提供します。
"""

import base64
from PIL import Image
from openai import OpenAI
import os
from dotenv import load_dotenv

# 環境変数からAPIキーを読み込む
# .envファイルにOPENAI_API_KEYを設定することで、セキュリティを確保
load_dotenv()


def preprocess_image_for_gpt(image_path):
    """
    GPT-4o-miniに画像を与えるための画像前処理関数
    
    Args:
        image_path (str): 入力画像のパス（例: 'image/elephant.png'）
    
    Returns:
        str: base64エンコードされた画像データ（data URI形式）
    
    処理内容:
    - 画像ファイルを読み込む（PIL.Imageを使用）
    - RGB形式に変換（GPT-4o-miniはRGB形式の画像を期待）
    - base64エンコードしてdata URI形式に変換
    - OpenAI APIが受け取れる形式で返す
    
    data URI形式: data:image/png;base64,<base64_string>
    """
    # 画像ファイルを読み込む
    # PIL.Imageを使用することで、様々な画像形式（PNG、JPEG等）に対応できる
    image = Image.open(image_path)
    
    # RGB形式に変換（RGBAやグレースケール画像もRGBに統一）
    # GPT-4o-miniはRGB形式の画像を期待しているため、この変換が必要
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 画像をバイト列に変換
    # BytesIOを使用してメモリ上で画像をPNG形式で保存
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    # base64エンコード
    # OpenAI APIは画像をbase64エンコードされた形式で受け取る
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # data URI形式に変換
    # OpenAI APIはdata URI形式（data:image/png;base64,<base64_string>）で画像を受け取る
    data_uri = f"data:image/png;base64,{image_base64}"
    
    return data_uri


def generate_description_with_gpt(image_path, prompt):
    """
    GPT-4o-miniで推論した結果を返す関数
    
    Args:
        image_path (str): 入力画像のパス（例: 'image/elephant.png'）
        prompt (str): 画像に対する説明を求めるプロンプト
    
    Returns:
        str: GPT-4o-miniが生成した説明文
    
    処理内容:
    - 画像を前処理してbase64エンコード形式に変換
    - OpenAI APIを使用してGPT-4o-miniに画像とプロンプトを送信
    - 生成された説明文を返す
    
    注意:
    - 環境変数OPENAI_API_KEYにAPIキーを設定する必要があります
    - .envファイルにOPENAI_API_KEY=your_api_keyを記述してください
    """
    # 画像を前処理してbase64エンコード形式に変換
    # GPT-4o-miniは画像をbase64エンコードされたdata URI形式で受け取る
    image_data_uri = preprocess_image_for_gpt(image_path)
    
    # OpenAIクライアントを初期化
    # 環境変数OPENAI_API_KEYからAPIキーを自動的に読み込む
    # APIキーは.envファイルに設定するか、環境変数として設定してください
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # GPT-4o-miniに画像とプロンプトを送信して推論を実行
    # gpt-4o-miniはマルチモーダルモデルで、テキストと画像の両方を処理できる
    # messagesのcontentに画像（data URI形式）とテキストプロンプトを含める
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ]
            }
        ],
        max_tokens=500  # 生成される説明文の最大トークン数
    )
    
    # 推論結果から説明文を取得
    # response.choices[0].message.contentに生成された説明文が含まれる
    description = response.choices[0].message.content
    
    return description


def generate_description_from_base64(base64_image, prompt):
    """
    base64エンコードされた画像からGPT-4o-miniで説明文を生成する関数
    
    Args:
        base64_image (str): base64エンコードされた画像データ（data URI形式）
        prompt (str): 画像に対する説明を求めるプロンプト
    
    Returns:
        str: GPT-4o-miniが生成した説明文
    
    処理内容:
    - base64エンコードされた画像をそのまま使用（data URI形式）
    - OpenAI APIを使用してGPT-4o-miniに画像とプロンプトを送信
    - 生成された説明文を返す
    
    注意:
    - 環境変数OPENAI_API_KEYにAPIキーを設定する必要があります
    - .envファイルにOPENAI_API_KEY=your_api_keyを記述してください
    """
    # OpenAIクライアントを初期化
    # 環境変数OPENAI_API_KEYからAPIキーを自動的に読み込む
    api_key = os.getenv('OPENAI_API_KEY')
    
    # APIキーが設定されているか確認
    if not api_key:
        raise ValueError('OPENAI_API_KEYが設定されていません。.envファイルにOPENAI_API_KEY=your_api_keyを記述してください。')
    
    client = OpenAI(api_key=api_key)
    
    # data URI形式の画像をそのまま使用
    # フロントエンドから送信されたbase64画像は既にdata URI形式になっている
    image_data_uri = base64_image
    
    # GPT-4o-miniに画像とプロンプトを送信して推論を実行
    # gpt-4o-miniはマルチモーダルモデルで、テキストと画像の両方を処理できる
    # messagesのcontentに画像（data URI形式）とテキストプロンプトを含める
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ]
            }
        ],
        max_tokens=500  # 生成される説明文の最大トークン数
    )
    
    # 推論結果から説明文を取得
    # response.choices[0].message.contentに生成された説明文が含まれる
    description = response.choices[0].message.content
    
    return description


# 参考URL:
# - OpenAI API ドキュメント: https://platform.openai.com/docs/guides/vision
# - OpenAI Python SDK: https://github.com/openai/openai-python
# - PIL Image ドキュメント: https://pillow.readthedocs.io/
# - Base64 エンコーディング: https://docs.python.org/3/library/base64.html





