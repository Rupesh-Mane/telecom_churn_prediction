import pickle

from src.config.config import MODEL_PATH


# Load model only once
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def predict(data):
    """
    Predict Churn
    """

    prediction = model.predict(data)

    return prediction


if __name__ == "__main__":

    print("=" * 50)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 50)