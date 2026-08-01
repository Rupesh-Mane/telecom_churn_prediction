import pandas as pd

from src.data.ingestion import load_data


def validate_data():

    # Load dataset
    df = load_data()

    print("=" * 40)
    print("DATA VALIDATION")
    print("=" * 40)

    # Shape
    print("Shape :", df.shape)

    # Check if dataset is empty
    if df.empty:
        print("Dataset is Empty")
    else:
        print("Dataset Loaded Successfully")

    # Check missing values
    print("\nMissing Values")
    print(df.isnull().sum())

    # Check duplicate rows
    print("\nDuplicate Rows :", df.duplicated().sum())

    # Display column names
    print("\nColumns")
    print(df.columns.tolist())


if __name__ == "__main__":
    validate_data()