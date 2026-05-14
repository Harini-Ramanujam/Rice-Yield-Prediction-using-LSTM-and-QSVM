import os
import sqlite3
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "agri_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# DATABASE (BULLETPROOF)
# -------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

# -------------------------------------------------
# LOAD ML MODELS
# -------------------------------------------------
IMG_SIZE = 224

class_names = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot"
]

disease_model = load_model(
    os.path.join(BASE_DIR, "models", "best_rice_leaf_disease_mobilenetv2.h5")
)

qsvr = joblib.load(os.path.join(BASE_DIR, "models", "qsvr_model.joblib"))
quantum_kernel = joblib.load(os.path.join(BASE_DIR, "models", "quantum_kernel.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "models", "label_encoders.joblib"))
X_train = joblib.load(os.path.join(BASE_DIR, "models", "X_train_qsvr.joblib"))

# -------------------------------------------------
# AUTH ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            conn.close()
            return redirect("/login")
        except sqlite3.IntegrityError:
            return "❌ Username already exists"

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/dashboard")
        else:
            return "❌ Invalid username or password"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("dashboard.html")

# -------------------------------------------------
# DISEASE PREDICTION
# -------------------------------------------------
from werkzeug.utils import secure_filename
import uuid

@app.route("/disease", methods=["GET", "POST"])
def disease():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":

        # 1️⃣ CHECK IMAGE
        if "image" not in request.files:
            return "❌ No image uploaded"

        file = request.files["image"]
        if file.filename == "":
            return "❌ Please select an image file"

        # 2️⃣ SAFE UNIQUE FILENAME
        unique_name = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(path)

        # 3️⃣ PREPROCESS IMAGE
        img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4️⃣ MODEL PREDICTION
        pred = disease_model.predict(img_array)
        disease_name = class_names[np.argmax(pred)]
        confidence = round(float(np.max(pred)) * 100, 2)

        # 5️⃣ SEND ONLY REQUIRED DATA
        return render_template(
            "result.html",
            title="Rice Leaf Disease Detection",
            prediction=disease_name,
            confidence=confidence,
            image_file=unique_name
        )

    return render_template("disease.html")



# -------------------------------------------------
# YIELD PREDICTION
# -------------------------------------------------
@app.route("/yield", methods=["GET", "POST"])
def yield_prediction():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        f = request.form

        X_cat = [
            label_encoders["Crop_Type"].transform([f["crop"]])[0],
            label_encoders["Region"].transform([f["region"]])[0],
            label_encoders["State"].transform([f["state"]])[0]
        ]

        X_input = np.array([X_cat + [
            float(f["rainfall"]),
            float(f["temp"]),
            float(f["humidity"]),
            float(f["fertilizer"]),
            float(f["ph"]),
            float(f["moisture"]),
            float(f["sunlight"])
        ]])

        X_scaled = scaler.transform(X_input)
        X_q = X_scaled[:, :4]
        K_input = quantum_kernel.evaluate(X_q, X_train)

        predicted_yield = round(float(qsvr.predict(K_input)[0]), 2)

        return render_template(
            "result.html",
            mode="yield",   # 🔥 IMPORTANT
            title="Rice Yield Prediction",
            prediction=f"{predicted_yield} kg/ha"
        )

    return render_template("yield.html")

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
