import pandas as pd
import mlflow


# Load model dari MLflow registry
model = mlflow.sklearn.load_model(
    "models:/churn_model@staging"
)


def predict(data):

    sample_customer = pd.DataFrame([data])

    prediction = model.predict(
        sample_customer
    )

    return int(
        prediction[0]
    )


if __name__ == "__main__":

    sample = {

        "Payment Delay":25,
        "Support Calls":8,
        "Tenure":50,

        "Gender":"Female",
        "Subscription Type":"Basic",
        "Contract Length":"Monthly"

    }

    result = predict(
        sample
    )

    print(
        "Prediction:",
        result
    )