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

- Customer churn prediction
- REST API deployment using FastAPI
- Experiment tracking using MLflow
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
