import os

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    api_base_url="https://www.googleapis.com/oauth2/v2/",
    client_kwargs={"scope": "openid email profile"},
)
model = joblib.load("student_score_model.pkl")
model_columns = joblib.load("model_columns.pkl")

NUM_COLS = ["age", "study_hours", "class_attendance", "sleep_hours"]
CAT_COLS = [
    "gender", "course", "internet_access", "sleep_quality",
    "study_method", "facility_rating", "exam_difficulty"
]

COURSE_MAP = {
    "bsc": "b.sc",
    "b.sc": "b.sc",
    "btech": "b.tech",
    "b.tech": "b.tech",
    "ba": "ba",
    "bba": "bba",
    "bca": "bca",
    "diploma": "diploma",
}

STUDY_METHOD_MAP = {
    "group-study": "group_study",
    "group study": "group_study",
    "online videos": "online_videos",
    "online-videos": "online_videos",
    "self-study": "self-study",
    "mixed": "mixed",
}


def _safe_next_url(next_url: str | None) -> str | None:
    if next_url and next_url.startswith("/"):
        return next_url
    return None


@app.context_processor
def inject_user():
    return {"current_user": session.get("user")}

@app.route("/")
def landing():
    return render_template("home.html")


@app.route("/login")
def login():
    next_url = request.args.get("next")
    return render_template("login.html", next_url=next_url)


@app.route("/auth/google")
def auth_google():
    next_url = _safe_next_url(request.args.get("next"))
    if next_url:
        session["next_url"] = next_url
    redirect_uri = url_for("auth_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_callback():
    token = google.authorize_access_token()
    user_info = google.get("userinfo").json()
    session["user"] = {
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
    }

    next_url = _safe_next_url(session.pop("next_url", None))
    return redirect(next_url or url_for("landing"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("user"):
        return redirect(url_for("login", next=request.path))

    prediction = None
    history = session.get("prediction_history", [])

    if request.method == "POST":
        data = pd.DataFrame([request.form])

        data[NUM_COLS] = data[NUM_COLS].apply(pd.to_numeric, errors="coerce")

        for col in CAT_COLS:
            data[col] = (
                data[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        data["course"] = data["course"].map(COURSE_MAP).fillna(data["course"])
        data["study_method"] = data["study_method"].map(STUDY_METHOD_MAP).fillna(data["study_method"])
        data["sleep_quality"] = data["sleep_quality"].replace({"average": "poor"})
        data["facility_rating"] = data["facility_rating"].replace({"high": "medium"})
        data["exam_difficulty"] = data["exam_difficulty"].replace({"easy": "moderate"})

        data = pd.get_dummies(data)
        data = data.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(data)[0]

        history.append({
            "score": float(prediction),
            "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })
        history = history[-10:]
        session["prediction_history"] = history

    return render_template("prediction.html", prediction=prediction, history=history)

if __name__ == "__main__":
    app.run(debug=True)