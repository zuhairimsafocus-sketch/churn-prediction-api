import joblib
import pandas as pd

import mlflow

model = mlflow.sklearn.load_model(
    "models:/churn_model@staging"
)

# =====================================
# SAMPLE INPUT
# =====================================

sample_customer = pd.DataFrame({

    'Payment Delay': [25],
    'Support Calls': [8],
    'Tenure': [50],

    'Gender': ['Female'],
    'Subscription Type': ['Basic'],
    'Contract Length': ['Monthly']
})

# =====================================
# PREDICT
# =====================================

prediction = model.predict(
    sample_customer
)

probability = model.predict_proba(
    sample_customer
)

# =====================================
# OUTPUT
# =====================================

print('Prediction:', prediction[0])

print(
    'Churn Probability:',
    probability[0][1]
)