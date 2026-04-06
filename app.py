from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "dataset"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

model = None
df_global = None


#halaman utama
@app.route("/")
def index():
    return render_template("index.html")


#upload dataset
@app.route("/upload", methods=["POST"])
def upload():
    global model, df_global

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath, sep=";")
    df = df[["tahun", "persentase_penduduk_miskin"]]
    df["tahun_index"] = df["tahun"] - df["tahun"].min()

    X = df[["tahun_index"]]
    y = df["persentase_penduduk_miskin"]

    model = LinearRegression()
    model.fit(X, y)

    df_global = df

   #grafik awal
    _buat_grafik()

    return redirect(url_for("hasil"))


#input manual
@app.route("/manual", methods=["POST"])
def manual():
    global model, df_global

    tahun = int(request.form["tahun"])
    nilai = float(request.form["nilai"])

    data_baru = pd.DataFrame({
        "tahun": [tahun],
        "persentase_penduduk_miskin": [nilai]
    })

    if df_global is None:
        df_global = data_baru
    else:
        df_global = pd.concat([df_global, data_baru], ignore_index=True)

    df_global["tahun_index"] = df_global["tahun"] - df_global["tahun"].min()

    X = df_global[["tahun_index"]]
    y = df_global["persentase_penduduk_miskin"]

    model = LinearRegression()
    model.fit(X, y)

    _buat_grafik()

    return redirect(url_for("hasil"))


#grafik regresi linear
def _buat_grafik(tahun_prediksi=None, nilai_prediksi=None):
    """
    Membuat scatter plot data aktual + garis regresi.
    Jika tahun_prediksi dan nilai_prediksi diberikan,
    tambahkan titik prediksi.
    """
    if df_global is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

   #data aktual
    ax.scatter(
        df_global["tahun"],
        df_global["persentase_penduduk_miskin"],
        color="#27ae60",
        s=60,
        zorder=3,
        label="Data Aktual"
    )

    #garis regresi
    tahun_min = df_global["tahun"].min()
    tahun_max = df_global["tahun"].max()

    if tahun_prediksi and tahun_prediksi > tahun_max:
        x_line = np.linspace(tahun_min, tahun_prediksi, 200)
    else:
        x_line = np.linspace(tahun_min, tahun_max, 200)

    x_index = x_line - tahun_min
    y_line = model.predict(x_index.reshape(-1, 1))

    ax.plot(x_line, y_line, color="#2980b9", linewidth=2,
            zorder=2, label="Garis Regresi")

    #titik prediksi
    if tahun_prediksi is not None and nilai_prediksi is not None:
        ax.scatter(
            tahun_prediksi,
            nilai_prediksi,
            color="#e74c3c",
            s=150,
            zorder=4,
            label=f"Prediksi {tahun_prediksi}: {nilai_prediksi:.2f}%"
        )

    ax.set_title("Visualisasi Regresi Linear Prediksi Kemiskinan",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tahun", fontsize=12)
    ax.set_ylabel("Persentase Kemiskinan (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("static/grafik.png", dpi=120)
    plt.close(fig)

#Halaman Hasil
@app.route("/hasil", methods=["GET", "POST"])
def hasil():
    global df_global, model

    prediksi = None
    tahun_prediksi = None

    if df_global is None:
        return redirect(url_for("index"))

    tabel = df_global[["tahun", "persentase_penduduk_miskin"]].to_dict(orient="records")

    if request.method == "POST":
        tahun_prediksi = int(request.form["tahun"])
        tahun_index = np.array([[tahun_prediksi - df_global["tahun"].min()]])
        prediksi = round(model.predict(tahun_index)[0], 4)

       #titik prediksi
        _buat_grafik(tahun_prediksi=tahun_prediksi, nilai_prediksi=prediksi)

    #hasil grafik
    grafik = "grafik.png"

    return render_template(
        "hasil.html",
        prediksi=prediksi,
        tahun_prediksi=tahun_prediksi,
        grafik=grafik,
        tabel=tabel
    )


#run app
if __name__ == "__main__":
    app.run(debug=True)