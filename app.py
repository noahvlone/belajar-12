from flask import Flask, render_template, request
import pickle
import pandas as pd
from supabase import create_client, Client
import uvicorn

app = Flask(__name__)

# ----------------------------
# Load Model
# ----------------------------
with open("best_diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Supabase Config
# ----------------------------
SUPABASE_URL = "https://gdujqesvbhxvllwmqeet.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdkdWpxZXN2Ymh4dmxsd21xZWV0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc0NzE3MDcsImV4cCI6MjA3MzA0NzcwN30.lFII_A6ohk5f5BhmGSzdmsA9iuQ-Ek0XgJ0e1DZVZrE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Kolom input
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        # Ambil dari form (semua lowercase)
        raw_data = {
            "Pregnancies": request.form["pregnancies"],
            "Glucose": request.form["glucose"],
            "BloodPressure": request.form["bloodpressure"],
            "SkinThickness": request.form["skinthickness"],
            "Insulin": request.form["insulin"],
            "BMI": request.form["bmi"],
            "DiabetesPedigreeFunction": request.form["diabetespedigreefunction"],
            "Age": request.form["age"]
        }

        # Konversi ke float
        values = {k: float(v) for k, v in raw_data.items()}

        # DataFrame sesuai fitur model
        data = pd.DataFrame([values])

        pred = model.predict(data)[0]
        result = "Diabetes" if pred == 1 else "Tidak Diabetes"

        return render_template("result.html", data=values, result=result)

    except Exception as e:
        return f"Error: {e}"


@app.route("/predict_supabase", methods=["POST"])
def predict_supabase():
    patient_id = int(request.form["patient_id"])
    response = supabase.table("patients").select("*").eq("id", patient_id).execute()

    if response.data:
        row = response.data[0]

        # Mapping dari supabase (lowercase) ke model (CamelCase)
        data_dict = {
            "Pregnancies": row["pregnancies"],
            "Glucose": row["glucose"],
            "BloodPressure": row["bloodpressure"],
            "SkinThickness": row["skinthickness"],
            "Insulin": row["insulin"],
            "BMI": row["bmi"],
            "DiabetesPedigreeFunction": row["diabetespedigreefunction"],
            "Age": row["age"]
        }

        data = pd.DataFrame([data_dict])
        pred = model.predict(data)[0]
        result = "Diabetes" if pred == 1 else "Tidak Diabetes"
        return render_template("result.html", data=data_dict, result=result)
    else:
        return "Data pasien tidak ditemukan."

if __name__ == "__main__":
     app.run(host="82.197.71.171", port=4100, debug=True)


