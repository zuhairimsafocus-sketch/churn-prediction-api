import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder, OrdinalEncoder
)

from sklearn.linear_model import LogisticRegression

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv('../data/customer_churn.csv')

# =====================================
# FEATURES & TARGET
# =====================================

x = df.drop('Churn', axis=1)

y = df['Churn']

# =====================================
# SPLIT
# =====================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# PREPROCESSOR
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Payment Delay', 'Support Calls', 'Tenure']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender']),
        ('cat2', OrdinalEncoder(), ['Subscription Type', 'Contract Length'])
    ]
)

# =====================================
# PIPELINE
# =====================================

pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', LogisticRegression())
])

# =====================================
# TRAIN
# =====================================

pipeline.fit(x_train, y_train)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    pipeline,
    '../models/churn_model_v2.pkl'
)

print('Model saved successfully.')