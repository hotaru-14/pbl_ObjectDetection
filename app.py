from flask import Flask, session

app = Flask(__name__)
app.secret_key = "your-secret-key"  # セッションを使用するために必要

# GETメソッド用の関数：現在のカウントとボタンを表示
@app.get("/")
def get_counter():
    # セッションにカウントがなければ0で初期化
    if "count" not in session:
        session["count"] = 0
    
    # GETリクエスト時は現在のカウントとボタンを表示
    return f"""
    <p>現在のカウント: {session['count']}</p>
    <form method="POST">
        <button type="submit">カウントアップ</button>
    </form>
    """

# POSTメソッド用の関数：カウントを増やす
@app.post("/")
def post_counter():
    # セッションにカウントがなければ0で初期化
    if "count" not in session:
        session["count"] = 0
    
    # POSTリクエスト（ボタンが押された時）の場合、カウントを増やす
    session["count"] += 1
    return f"<p>カウント: {session['count']}</p><a href='/'>戻る</a>"

# スクリプトが直接実行された場合にFlaskアプリケーションを起動
# これがないと、python3 app.pyを実行してもサーバーが起動しない
if __name__ == "__main__":
    app.run(debug=True)
