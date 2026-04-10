from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import joblib
from pathlib import Path

# ==================================
# FLASK CONFIG (WAJIB UNTUK VERCEL)
# ==================================
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ==================================
# PATH MODEL
# ==================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.path.join(BASE_DIR, "model_ann.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model & Scaler Loaded")
except Exception as e:
    print("❌ ERROR LOAD MODEL:", e)

# ==================================
# HALAMAN UTAMA
# ==================================
@app.route("/")
def index():

    tahun = [2015,2016,2017,2018,2019,2020,2021,2022,2023]
    laki = [4500,4700,4800,5000,5200,5300,5400,5600,5800]
    perempuan = [4300,4400,4600,4700,4900,5000,5100,5200,5400]

    return render_template(
        "index.html",
        tahun=tahun,
        laki=laki,
        perempuan=perempuan,
        mae=40.61,
        mse=2787.69,
        r2=0.9772
    )

# ==================================
# API PREDICT (FIX VERCEL)
# ==================================
@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:
        return jsonify({"error": "Model tidak terbaca"}), 500

    try:
        # 🔥 VERCEL WAJIB JSON
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON kosong"}), 400

        tahun_val = int(data["tahun"])

        # estimasi laki-laki
        laki_estimasi = 4500 + (tahun_val - 2015) * 200

        input_data = np.array([[tahun_val, laki_estimasi]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        perempuan_pred = float(prediction[0])

        return jsonify({
            "status": "success",
            "tahun": tahun_val,
            "laki_laki": laki_estimasi,
            "perempuan": round(perempuan_pred,2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==================================
# EXPORT UNTUK VERCEL
# ==================================
app = app
