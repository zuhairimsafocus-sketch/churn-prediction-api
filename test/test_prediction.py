from fastapi.testclient import TestClient
from src.main import app
import os

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():

    sample = {
        "PaymentDelay":25,
        "SupportCalls":8,
        "Tenure":50,
        "Gender":"Female",
        "SubscriptionType":"Basic",
        "ContractLength":"Monthly"
    }

    response = client.post(
        "/predict",
       headers={
    "api-key": os.getenv("API_KEY", "zul12345")
        },
        
        json=sample
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result