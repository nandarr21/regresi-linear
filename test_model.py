import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

tahun = np.array([[2027]])

prediksi = model.predict(tahun)

print("Prediksi Kemiskinan Tahun 2027:", prediksi[0])