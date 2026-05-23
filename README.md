# Customer Churn Prediction API

Machine Learning API for predicting customer churn using Logistic Regression and FastAPI.

## Tech Stack

- Python
- Scikit-learn
- FastAPI
- MLflow
- Docker
- GitHub

## Project Structure

```
customer-churn-api/
│
├── data/
├── models/
├── notebooks/
├── src/
│ ├── train.py
│ ├── prediction.py
│ └── main.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

### Machine Learning
- Customer churn prediction model
- Probability prediction
- Risk level indicator:
    - Low Risk 🟢
    - Medium Risk 🟡
    - High Risk 🔴

### API Features
- FastAPI REST API
- API key authentication
- Request rate limiting
- Model versioning
- Health check endpoint
- Swagger API documentation

### Monitoring
- Prometheus metrics collection
- Grafana monitoring dashboard
- API latency tracking
- Request rate monitoring
- Error rate monitoring
- CPU monitoring
- RAM monitoring
- Prediction request tracking

### Deployment
- Docker containerization
- Model versioning

## API Endpoint

POST:

```

/predict

```

Example:

```json
{
"Payment Delay":25,
"Support Calls":8,
"Tenure":50,
"Gender":"Female",
"Subscription Type":"Basic",
"Contract Length":"Monthly"
}
```

Response:

```json
{
"prediction":1,
"churn_probability":0.95
}
```
