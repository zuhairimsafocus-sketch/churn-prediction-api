from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
import joblib
import pandas as pd
import time
import os
import logging
import json
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path


# ============================
# LOAD MODEL
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model_v2.pkl"

model = joblib.load(MODEL_PATH)

class CustomerInput(BaseModel):
    PaymentDelay:int
    SupportCalls:int
    Tenure:int
    Gender:str
    SubscriptionType:str
    ContractLength:str

# ============================
# CREATE APP
# ============================

app = FastAPI()

Instrumentator().instrument(app).expose(app)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

API_KEY = os.getenv("API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@app.get("/")
def home():

    return {
        "message":"Churn API Running"
    }


@app.get("/health")
def health():

    return {
        "status":"healthy",
        "model_version":"v2"
    }

# ============================
# PREDICTION ENDPOINT
# ============================

@app.post("/predict")
@limiter.limit("5/minute")
def predict(
    request: Request,
    data: CustomerInput,
    api_key: str = Header(None)
):

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    try:
        start = time.time()

        df = pd.DataFrame([data.dict()])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        latency = time.time() - start

        result = {
            "prediction":
                "Churn" if int(prediction)==1 else "No Churn",

            "churn_probability":
                f"{probability:.2%}",

            "latency":
                f"{latency:.4f} seconds",

            "model_version":
                "v2"
        }

        logging.info(json.dumps(result))

        return result

    except Exception as e:
        return {
            "error": str(e)
        }