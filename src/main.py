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
from prometheus_client import Histogram
import time
import shap

# ============================
# LOAD MODEL
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model_v2.pkl"

model = joblib.load(MODEL_PATH)

explainer = shap.Explainer(model)

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

instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

API_KEY = os.getenv("API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency"
)

@app.middleware("http")
async def measure_latency(request, call_next):

    start_time = time.time()

    response = await call_next(request)

    REQUEST_LATENCY.observe(
        time.time() - start_time
    )

    return response

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

        input_data = data.model_dump()

        input_data = {
            "Payment Delay": input_data["PaymentDelay"],
            "Support Calls": input_data["SupportCalls"],
            "Tenure": input_data["Tenure"],
            "Gender": input_data["Gender"],
            "Subscription Type": input_data["SubscriptionType"],
            "Contract Length": input_data["ContractLength"]
}

        df = pd.DataFrame([input_data])


        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        shap_values = explainer(df)
        
        importances = dict(
            zip(df.columns, 
                shap_values.values[0]
                )
            ),
        
        top_features = sorted(importances.items(),
                              key=lambda x: abs(x[1]),
                              reverse=True
                              )[:3]

        latency = time.time() - start
        
        risk = "Low"
        
        if probability > 0.7:
            risk = "High 🔴"
            
        elif probability >0.3:
            risk = "Medium 🟡"
            
        else:
            risk = "Low 🟢"
            
        result = {
            "prediction":
                "Churn" if int(prediction)==1 else "No Churn",

            "churn_probability":
                f"{probability:.2%}",
                
            "risk_level":
                risk,
                
            "top_factors":
                [f"{k}: {round(v,3)}"
                 for k,v in top_features],

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