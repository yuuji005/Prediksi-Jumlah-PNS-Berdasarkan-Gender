from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import joblib
from pathlib import Path

# ==================================
# FLASK CONFIG (WAJIB VERCEL)
# ==================================
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ==================================
# LOAD MODEL
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
# DATA HISTORIS
# ==================================
tahun_hist = [2015,2016,2017,2018,2019,2020,2021,2022,2023]
laki_hist = [4500,4700,4800,5000,5200,5300,5400,5600,5800]
perempuan_hist = [4300,4400,4600,4700,4900,5000,5100,5200,5400]


# ==================================
# DASHBOARD
# ==================================
@app.route("/")
def index():

    return render_template(
        "index.html",
        tahun=tahun_hist,
        laki=laki_hist,
        perempuan=perempuan_hist,
        mae=40.61,
        mse=2787.69,
        r2=0.9772
    )


# ==================================
# PREDICT API (FINAL FIX)
# ==================================
@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:
        return jsonify({"error": "Model tidak ditemukan"}), 500

    try:
        # ✅ SUPPORT FORM + JSON
        tahun_input = request.form.get("tahun")

        if tahun_input is None:
            json_data = request.get_json(silent=True)
            if json_data:
                tahun_input = json_data.get("tahun")

        if not tahun_input:
            return jsonify({"error": "Input tahun kosong"}), 400

        tahun_val = int(tahun_input)

        # ==================================
        # MODEL INPUT (1 FEATURE)
        # ==================================
        input_data = np.array([[tahun_val]])
        input_scaled = scaler.transform(input_data)

        perempuan_pred = float(model.predict(input_scaled)[0])

        # ==================================
        # ESTIMASI LAKI-LAKI (SMOOTH)
        # ==================================
        growth = np.mean(np.diff(laki_hist))
        laki_estimasi = laki_hist[-1] + (
            tahun_val - tahun_hist[-1]
        ) * growth

        # ==================================
        # ANTI GRAFIK ANJLOK
        # ==================================
        perempuan_pred = max(perempuan_hist[-1], perempuan_pred)

        return jsonify({
            "status": "success",
            "tahun": tahun_val,
            "laki_laki": round(float(laki_estimasi),2),
            "perempuan": round(perempuan_pred,2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==================================
# EXPORT UNTUK VERCEL
# ==================================
app = app
