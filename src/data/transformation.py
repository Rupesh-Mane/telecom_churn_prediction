import pickle
import pandas as pd

from src.config.config import (
    CAT_IMPUTER_PATH,
    NUM_IMPUTER_PATH,
    ENCODER_PATH,
    SCALER_PATH,
)

# Numerical Columns
num_data = [
    "SeniorCitizen",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "MonthlyDataUsageGB",
    "NumberOfComplaints",
    "SatisfactionScore",
    "AvgCallDuration"
]

# Categorical Columns
cat_data = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def transform_data(df):

    # Load Artifacts
    with open(CAT_IMPUTER_PATH, "rb") as file:
        cat_imputer = pickle.load(file)

    with open(NUM_IMPUTER_PATH, "rb") as file:
        num_imputer = pickle.load(file)

    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    # Separate Numerical and Categorical Data
    num_df = df[num_data]
    cat_df = df[cat_data]

    # Handle Missing Values
    num_df = pd.DataFrame(
        num_imputer.transform(num_df),
        columns=num_data
    )

    cat_df = pd.DataFrame(
        cat_imputer.transform(cat_df),
        columns=cat_data
    )

    # One-Hot Encode
    cat_encoded = pd.DataFrame(
        encoder.transform(cat_df),
        columns=encoder.get_feature_names_out(cat_data)
    )

    # Scale Numerical Data
    num_scaled = pd.DataFrame(
        scaler.transform(num_df),
        columns=num_data
    )

    # Combine Numerical + Encoded Categorical
    final_data = pd.concat([num_scaled, cat_encoded], axis=1)

    return final_data


if __name__ == "__main__":

    print("Data Transformation Module Ready")







"""
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
    
"""