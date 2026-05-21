from fastapi import FastAPI, HTTPException, header
import joblib
import pandas as pd
import time

# ============================
# LOAD MODEL
# ============================

model = joblib.load(
    "../models/churn_model_v2.pkl"
)

# ============================
# CREATE APP
# ============================

app = FastAPI()
API_KEY = "zul12345"

@app.get("/")
def home():

    return {
        "message":"Churn API Running"
    }

# ============================
# PREDICTION ENDPOINT
# ============================

@app.post("/predict")
def predict(data:dict, api_key: str = header(None)):

    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:

        start = time.time()

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0][1]

        latency = time.time() - start

        result = {
            "prediction": (
                "Churn"
                if int(prediction)==1
                else "No Churn"
            ),

            "churn_probability":
            f"{probability:.2%}",

            "latency":
            f"{latency:.4f} seconds",

            "model_version":
            "v2"
        }

        print(result)

        return result

    except Exception as e:

        return {
            "error":str(e)
        }