from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# Inisialisasi Flask dengan path folder yang benar untuk Vercel
app = Flask(__name__, 
            template_folder='../templates', 
            static_folder='../static')

# ==================================
# KONFIGURASI PATH MODEL (FIX ERROR 500/400)
# ==================================
# Menentukan root directory (satu tingkat di atas folder api)
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.path.join(BASE_DIR, 'model_ann.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load Model & Scaler saat server mulai
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model & Scaler loaded successfully!")
except Exception as e:
    model = None
    scaler = None
    print(f"Error loading model/scaler: {e}")

# ==================================
# ROUTE DASHBOARD (HALAMAN UTAMA)
# ==================================
@app.route("/")
def index():
    # Data historis untuk grafik awal
    tahun = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    laki = [4500, 4700, 4800, 5000, 5200, 5300, 5400, 5600, 5800]
    perempuan = [4300, 4400, 4600, 4700, 4900, 5000, 5100, 5200, 5400]

    # Metrik Evaluasi Model
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
# ROUTE PREDIKSI (API UNTUK TOMBOL)
# ==================================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model file (.pkl) tidak ditemukan di server"}), 500

    try:
        # 1. Ambil data dari request (bisa dari Form atau JSON)
        tahun_input = request.form.get("tahun")
        
        if not tahun_input:
            return jsonify({"error": "Input tahun tidak ditemukan"}), 400
            
        tahun_val = int(tahun_input)

        # 2. Siapkan fitur input
        # Karena model ANN kamu dilatih dengan 2 kolom [Tahun, Laki-laki]
        # Kita buat estimasi pertumbuhan Laki-laki sederhana
        laki_estimasi = 4500 + (tahun_val - 2015) * 200 
        
        # 3. Bentuk array input dan lakukan Scaling
        input_data = np.array([[tahun_val, laki_estimasi]])
        input_scaled = scaler.transform(input_data)
        
        # 4. Lakukan Prediksi
        prediction_scaled = model.predict(input_scaled)
        
        # Pastikan hasil prediksi berupa angka tunggal
        if isinstance(prediction_scaled, np.ndarray):
            perempuan_pred = float(prediction_scaled[0])
        else:
            perempuan_pred = float(prediction_scaled)

        # 5. Kirim balik hasil ke Dashboard
        return jsonify({
            "tahun": tahun_val,
            "laki_laki": float(laki_estimasi),
            "perempuan": round(perempuan_pred, 2),
            "status": "success"
        })

    except Exception as e:
        # Mengirim pesan error spesifik agar muncul di Console browser
        return jsonify({"error": str(e)}), 400

# Ekspos objek app untuk Vercel
app = app

if __name__ == "__main__":
    app.run(debug=True)
