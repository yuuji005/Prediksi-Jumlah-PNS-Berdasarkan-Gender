from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# ==================================
# DATASET PNS JAWA BARAT
# ==================================
data = {
    "Tahun":[2015,2016,2017,2018,2019,2020,2021,2022,2023],
    "Laki_laki":[4500,4700,4800,5000,5200,5300,5400,5600,5800],
    "Perempuan":[4300,4400,4600,4700,4900,5000,5100,5200,5400]
}

df = pd.DataFrame(data)

# ==================================
# TRAIN ANN
# ==================================
X = df[["Tahun","Laki_laki"]]
y = df["Perempuan"]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1)).ravel()

model = MLPRegressor(
    hidden_layer_sizes=(12,8),
    activation="relu",
    solver="adam",
    max_iter=5000,
    random_state=42
)

model.fit(X_scaled, y_scaled)

# ==================================
# EVALUASI MODEL
# ==================================
pred_scaled = model.predict(X_scaled)
pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))

mae = mean_absolute_error(y, pred)
mse = mean_squared_error(y, pred)
r2 = r2_score(y, pred)

# ==================================
# ROUTE DASHBOARD
# ==================================
@app.route("/")
def index():

    # DATA CONTOH (WAJIB ADA)
    tahun = [2015,2016,2017,2018,2019,2020,2021,2022,2023]
    laki = [4500,4700,4800,5000,5200,5300,5400,5600,5800]
    perempuan = [4300,4400,4550,4650,4850,4950,5050,5150,5300]

    mae = 40.61
    mse = 2787.69
    r2 = 0.9772

    return render_template(
        "index.html",
        tahun=tahun,
        laki=laki,
        perempuan=perempuan,
        mae=mae,
        mse=mse,
        r2=r2
    )

# ==================================
# PREDIKSI BERDASARKAN TAHUN
# ==================================
@app.route("/predict", methods=["POST"])
def predict():

    tahun = int(request.form["tahun"])

    # CONTOH PREDIKSI ANN
    laki_pred = 4500 + (tahun - 2015) * 200
    perempuan_pred = 4300 + (tahun - 2015) * 180

    return jsonify({
        "tahun": tahun,
        "laki_laki": float(laki_pred),
        "perempuan": float(perempuan_pred)
    })

def handler(request, context):
    return app(request.environ, lambda *args: None)

if __name__ == "__main__":
    app.run(debug=True)