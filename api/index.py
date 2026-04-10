from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib  # Pastikan 'joblib' ada di requirements.txt
from pathlib import Path

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# ==================================
# KONFIGURASI PATH (PENTING UNTUK VERCEL)
# ==================================
# Mengambil path folder 'api' saat ini, lalu naik satu tingkat ke root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.path.join(BASE_DIR, 'model_ann.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# ==================================
# LOAD MODEL & SCALER
# ==================================
# Kita muat model sekali saja saat aplikasi dijalankan
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# ==================================
# ROUTE DASHBOARD
# ==================================
@app.route("/")
def index():
    # Data historis untuk ditampilkan di grafik Dashboard
    tahun = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    laki = [4500, 4700, 4800, 5000, 5200, 5300, 5400, 5600, 5800]
    perempuan = [4300, 4400, 4600, 4700, 4900, 5000, 5100, 5200, 5400]

    # Metrik Evaluasi (Bisa dihitung dinamis atau hardcoded dari hasil latihan)
    mae_val = 40.61
    mse_val = 2787.69
    r2_val = 0.9772

    return render_template(
        "index.html",
        tahun=tahun,
        laki=laki,
        perempuan=perempuan,
        mae=mae_val,
        mse=mse_val,
        r2=r2_val
    )

# ==================================
# ROUTE PREDIKSI
# ==================================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model belum terload di server"}), 500

    try:
        tahun_input = int(request.form["tahun"])
        
        # Logika Prediksi Menggunakan Model .pkl
        # Misal model mengharapkan input [[Tahun, Laki_Laki]]
        # Di sini kita asumsikan input dummy untuk Laki-Laki jika hanya Tahun yang diinput
        laki_estimasi = 4500 + (tahun_input - 2015) * 200 
        
        input_data = np.array([[tahun_input, laki_estimasi]])
        input_scaled = scaler.transform(input_data)
        
        prediction_scaled = model.predict(input_scaled)
        
        # Jika scaler kamu hanya untuk Y, gunakan inverse_transform pada hasil
        # Jika tidak, sesuaikan dengan logika scaler di train.py kamu
        perempuan_pred = float(prediction_scaled[0]) 

        return jsonify({
            "tahun": tahun_input,
            "laki_laki": float(laki_estimasi),
            "perempuan": round(perempuan_pred, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Agar objek 'app' bisa diakses oleh Vercel
app = app

if __name__ == "__main__":
    app.run(debug=True)
