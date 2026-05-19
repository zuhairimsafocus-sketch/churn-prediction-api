import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri("../mlruns")
mlflow.set_experiment(
    "Customer Churn Prediction"
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# =========================
# LOAD DATA
# =========================

df = pd.read_csv("../data/customer_churn.csv")

# Features
x = df.drop("Churn", axis=1)

# Target
y = df["Churn"]

# =========================
# TRAIN TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# CATEGORICAL COLUMNS
# =========================

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

# =========================
# PREPROCESSING
# =========================

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
# PIPELINE
# =========================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        LogisticRegression()
    )
])

# =========================
# MLFLOW TRACKING
# =========================

with mlflow.start_run():

    # Train
    pipeline.fit(
        x_train,
        y_train
    )

    # Prediction
    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

    plt.savefig("../confusion_matrix.png")

    mlflow.log_artifact("../confusion_matrix.png")

    print(
        "Accuracy:",
        accuracy
    )

    # Log parameters
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

    # Log metrics
    mlflow.log_metric(
        "accuracy",
        accuracy
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
    "../models/churn_model.pkl"
)

print(
    "Model saved successfully"
)