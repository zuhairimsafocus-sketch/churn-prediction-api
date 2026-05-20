from fastapi import FastAPI
import joblib
import pandas as pd

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "../models/churn_model_v2.pkl"
)

# =========================
# CREATE APP
# =========================

app = FastAPI()


@app.get("/")
def home():

    return {
        "message":"Churn API Running"
    }


@app.post("/predict")
def predict(data:dict):

    try:

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0][1]

        return {

    "prediction": (
        "Churn"
        if int(prediction)==1
        else "No Churn"
    ),

    "churn_probability":
    f"{probability:.2%}"

}

    except Exception as e:

        return {
            "error":str(e)
        }