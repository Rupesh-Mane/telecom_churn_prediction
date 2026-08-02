import pickle

from src.config.config import MODEL_PATH


def predict(data):
    """
    Predict customer churn using the saved model.
    """

    # Load Saved Model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Predict
    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":

    print("=" * 50)
    print("PREDICTION MODEL READY")
    print("=" * 50)















"""import pickle

from src.config.config import MODEL_PATH


# Load model only once
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def predict(data):
    
    #Predict Churn
    

    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":

    print("=" * 50)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 50)"""