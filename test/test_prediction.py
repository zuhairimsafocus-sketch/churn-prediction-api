from src.prediction import predict

def test_prediction():

    sample = {

        "Payment Delay":25,
        "Support Calls":8,
        "Tenure":50,
        "Gender":"Female",
        "Subscription Type":"Basic",
        "Contract Length":"Monthly"

    }

    result = predict(sample)

    assert result in [0,1]