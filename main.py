from flask import Flask, render_template, render_template_string, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os, json
import pandas as pd
from datetime import datetime
import csv

# TODO: promptを工夫する

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.2-2025-12-11"

# データベースのパス
DATABASE_PATH = "data/Encyclopedia.csv"

app = Flask(__name__)

# 場所の一覧取得
def read_places(path="data/Encyclopedia.csv"):
    df = pd.read_csv(path)
    return df["place"].unique().tolist()

def ensure_trailing_newline(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        if last != b"\n":  # 末尾が \r の場合でも \n を足せば \r\n になる
            f.write(b"\n")

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


@app.route("/api/save_encyclopedia", methods=["POST"])
def save_encyclopedia():
    payload = request.get_json(silent=True) or {}
    place = (payload.get("place") or "").strip()
    items = payload.get("encyclopedia") or []

    today = datetime.now().strftime("%Y-%m-%d")

    rows = 0
    
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    ensure_trailing_newline(DATABASE_PATH)  

    with open(DATABASE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in items:
            name = (item.get("name") or "").strip()
            description = (item.get("text") or "").strip()
            writer.writerow([name, "", description, place, today])
            rows += 1

    return jsonify({"success": True, "rows": rows})


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

    # response = client.responses.create(
    #     model=MODEL,
    #     reasoning={"effort": "low"},
    #     tools=[{"type": "web_search"}],
    #     input=prompt,
    # )

    response = client.responses.create(
        model="gpt-4.1-2025-04-14",
        tools=[{"type": "web_search"}],
        input=prompt,
    )

    # TODO: ここにエラー処理を追加（JSONとしてパースできない等）
    data = json.loads(response.output_text)

    return jsonify(data)



if __name__ == "__main__":
    app.run(debug=True)
