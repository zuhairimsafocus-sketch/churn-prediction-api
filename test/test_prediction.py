from src.prediction import prediction 


def test_prediction():

    sample = {
        "Age": 35,
        "Gender": "Male",
        "Tenure": 12,
        "Usage Frequency": 5,
        "Support Calls": 2,
        "Payment Delay": 0,
        "Subscription Type": "Standard",
        "Contract Length": "Annual",
        "Total Spend": 500
    }

    result = predict(sample)

    assert result in [0,1]
