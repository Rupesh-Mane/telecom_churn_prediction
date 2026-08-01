import pandas as pd

from src.config.config import PROCESSED_DATA_PATH


def transform_data():

    # Load Processed Dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separate Features and Target
    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    print("=" * 50)
    print("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(f"Features Shape : {x.shape}")
    print(f"Target Shape   : {y.shape}")

    return x, y


if __name__ == "__main__":

    x, y = transform_data()

    print("\nFirst 5 Rows")
    print(x.head())

    print("\nTarget")
    print(y.head())