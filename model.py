import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load dataset
df = pd.read_csv("dataset.csv", sep=";")

print("\nKolom Dataset:")
print(df.columns)

#proses data
df = df.dropna().reset_index(drop=True)

#ubah data 
df_nasional = (
    df.groupby("tahun")["persentase_penduduk_miskin"]
    .mean()
    .reset_index()
)

df_nasional["tahun_ke"] = (
    df_nasional["tahun"] - df_nasional["tahun"].min()
)

#smoothing data
df_nasional["smooth"] = (
    df_nasional["persentase_penduduk_miskin"]
    .rolling(window=3, center=True)
    .mean()
)

df_nasional = df_nasional.dropna().reset_index(drop=True)

#fitur dan target
X = df_nasional[["tahun_ke"]]
Y = df_nasional["smooth"]

#train model
model = LinearRegression()
model.fit(X, Y)

#evaluasi model
y_pred = model.predict(X)

mae = mean_absolute_error(Y, y_pred)
mse = mean_squared_error(Y, y_pred)
r2 = r2_score(Y, y_pred)

print("\n HASIL VALIDASI")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)

#simpan model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel berhasil dibuat!")

#prediksi
print("\n===== PREDIKSI MASA DEPAN =====")

tahun_masa_depan = list(range(2025, 2031))

tahun_ke_future = [
    t - df_nasional["tahun"].min()
    for t in tahun_masa_depan
]

data_future = pd.DataFrame({
    "tahun_ke": tahun_ke_future
})

prediksi_future = model.predict(data_future)

hasil_prediksi = pd.DataFrame({
    "Tahun": tahun_masa_depan,
    "Prediksi Kemiskinan (%)": prediksi_future
})

print(hasil_prediksi)

#visualisasi hasil
plt.figure(figsize=(10,6))

#data aktual
plt.scatter(
    df_nasional["tahun"],
    df_nasional["persentase_penduduk_miskin"],
    s=70,
    label="Data Aktual"
)

#model garis prediksi
plt.plot(
    df_nasional["tahun"],
    y_pred,
    color="red",
    linewidth=2,
    label="Data Prediksi"
)

plt.title("Hasil Prediksi Regresi Linear")
plt.xlabel("Tahun")
plt.ylabel("Persentase Penduduk Miskin (%)")

plt.legend()
plt.grid(True)

plt.savefig("static/grafik_prediksi.png")
plt.show()

#cek data
print("\nPreview Data Nasional:")
print(df_nasional.head())
