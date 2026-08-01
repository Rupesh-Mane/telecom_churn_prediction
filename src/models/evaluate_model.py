import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from src.config.config import MODEL_PATH


def evaluate_model():

    # Load Test Data
    x_test = pd.read_csv("data/processed/x_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # Convert y_test into Series
    y_test = y_test.squeeze()

    # Load Trained Model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Prediction
    y_pred = model.predict(x_test)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()