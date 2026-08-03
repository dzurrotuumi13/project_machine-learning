# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load pipeline models
model_logreg = joblib.load("models/logreg_model.pkl")
model_rf = joblib.load("models/rf_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Ambil data dari form
            age = float(request.form["age"])
            sex = 1 if request.form["sex"] == "male" else 0
            cp = int(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = int(request.form["fbs"])
            restecg = int(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = int(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = int(request.form["slope"])
            model_choice = request.form["model"]

            # Susun input array
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                    restecg, thalach, exang, oldpeak, slope]])

            # Pilih model
            model = model_logreg if model_choice == "logreg" else model_rf

            # Prediksi
            pred = model.predict(input_data)[0]
            hasil = "Penderita Penyakit Jantung" if pred == 1 else "Tidak Terindikasi Penyakit Jantung"
            prediction = f"Hasil Prediksi ({'Logistic Regression' if model_choice == 'logreg' else 'Random Forest'}): {hasil}"

        except Exception as e:
            prediction = f"Terjadi kesalahan: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
