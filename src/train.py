import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
CONFUSION_PATH = BASE_DIR / "confusion_matrix.png"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# MLFLOW
# =========================

mlflow.set_experiment(
    "Customer Churn Prediction"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

x = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# PREPROCESSING
# =========================

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# =========================
# MODEL PIPELINE
# =========================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(max_iter=1000)
    )
])

# =========================
# TRAIN
# =========================

with mlflow.start_run():

    pipeline.fit(
        x_train,
        y_train
    )

    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    # Metrics
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # Parameters
    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred
    )

    plt.savefig(
        CONFUSION_PATH
    )

    mlflow.log_artifact(
        str(CONFUSION_PATH)
    )

    # Log model
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="churn_model"
    )

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(
    "Model saved successfully"
)

print(
    f"Accuracy: {accuracy}"
)