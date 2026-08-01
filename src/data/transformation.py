import pandas as pd

from src.config.config import PROCESSED_DATA_PATH


def transform_data():

    # Load Processed Dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separate Features and Target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print("=" * 50)
    print("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(f"Features Shape : {X.shape}")
    print(f"Target Shape   : {y.shape}")

    return X, y


if __name__ == "__main__":

    X, y = transform_data()

    print("\nFirst 5 Rows")
    print(X.head())

    print("\nTarget")
    print(y.head())