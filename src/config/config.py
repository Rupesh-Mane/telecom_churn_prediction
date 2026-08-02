# ==========================================
# DATASET PATHS
# ==========================================

RAW_DATA_PATH = "data/raw/telecom_customer_churn.csv"
PROCESSED_DATA_PATH = "data/processed/telecom_customer_processed.csv"


# ==========================================
# ARTIFACT PATHS
# ==========================================

MODEL_PATH = "artifacts/final_model.pkl"

ENCODER_PATH = "artifacts/encoder.pkl"

SCALER_PATH = "artifacts/scaler.pkl"

CAT_IMPUTER_PATH = "artifacts/cat_imputer.pkl"

NUM_IMPUTER_PATH = "artifacts/num_imputer.pkl"


# ==========================================
# MODEL SETTINGS
# ==========================================

TARGET_COLUMN = "Churn"

TEST_SIZE = 0.20

RANDOM_STATE = 42


if __name__ == "__main__":

    print("=" * 50)
    print("CONFIGURATION FILE")
    print("=" * 50)

    print(f"Raw Dataset      : {RAW_DATA_PATH}")
    print(f"Processed Dataset: {PROCESSED_DATA_PATH}")

    print(f"Model            : {MODEL_PATH}")
    print(f"Encoder          : {ENCODER_PATH}")
    print(f"Scaler           : {SCALER_PATH}")
    print(f"Cat Imputer      : {CAT_IMPUTER_PATH}")
    print(f"Num Imputer      : {NUM_IMPUTER_PATH}")

    print(f"Target Column    : {TARGET_COLUMN}")
    print(f"Test Size        : {TEST_SIZE}")
    print(f"Random State     : {RANDOM_STATE}")
    
#"Run this part only when I directly run this Python file."
# If another file imports config.py, the code inside that block will not run.
"""
For example:

from src.config.config import MODEL_PATH

This imports MODEL_PATH, but it won't automatically print everything inside the if __name__ == "__main__": block."""

#used the if __name__ == "__main__": block only when you want a small test.