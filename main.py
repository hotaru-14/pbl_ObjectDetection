from flask import Flask, render_template, render_template_string, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os, json
import pandas as pd

# TODO: promptを工夫する

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.2-2025-12-11"

app = Flask(__name__)

# 場所の一覧取得
def read_places(path="data/Encyclopedia.csv"):
    df = pd.read_csv(path)
    return df["place"].unique().tolist()

@app.route("/", methods=["GET"])
def index():
    return render_template("create_encyclopedia.html")

# helloworldのメインページにしておく(ヘッダーとフッターとメインコンテンツにhelloworldを表示)
@app.route("/camera", methods=["GET"])
def camera():
    return render_template("camera.html")

# helloworldのメインページにしておく(ヘッダーとフッターとメインコンテンツにhelloworldを表示)
@app.route("/encyclopedia", methods=["GET"])
def encyclopedia():
    places = read_places()
    return render_template("encyclopedia.html", places=places)

@app.route("/map", methods=["GET"])
def map():
    return render_template("map.html")


@app.route("/api/encyclopedia", methods=["POST"])
def create_encyclopedia():
    payload = request.get_json(silent=True) or {}
    place = (payload.get("place") or "").strip()

    # TODO: バリデーションのエラー処理を追加（空、長すぎる等）

    prompt = f"""
あなたは「写真で集める図鑑」アプリ用のコンテンツ編集者です。
入力された行先について、現地で“見つけやすく写真に撮りやすい”代表的な対象を厳選してください。
必ずウェブ検索で「その場所に実在し、一般客が観覧/撮影しやすい」ことを確認し、曖昧なら採用しないでください。

【選定ルール】
- encyclopediaは必ず8件（少なめ・厳選）
- 子どもが探しやすい/写真を撮りやすい対象を優先（目立つ・定番・通年・現地で案内されている）
- 期間限定・不確実・撮影禁止が多いものは避ける
- 動物園なら「人気で見つけやすい動物」を中心に。動物以外（有名スポット等）を混ぜてもよいが、写真に残しやすいものに限る
- nameは短く（固有名詞）
- textは子ども向けにやさしく、写真の手がかりになる“見た目の特徴”を入れる
- textは45字以内（日本語、句読点OK）

【出力形式】
以下のJSON形式「だけ」で返してください（前後の文章、説明、Markdownは禁止）。

{{
  "place": "{place}",
  "encyclopedia": [
    {{"name": "...", "text": "..."}},
    {{"name": "...", "text": "..."}},
    {{"name": "...", "text": "..."}},
    {{"name": "...", "text": "..."}},
    {{"name": "...", "text": "..."}}
  ]
}}
"""

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "low"},
        tools=[{"type": "web_search"}],
        input=prompt,
    )

    # TODO: ここにエラー処理を追加（JSONとしてパースできない等）
    data = json.loads(response.output_text)

    return jsonify(data)



if __name__ == "__main__":
    app.run(debug=True)
