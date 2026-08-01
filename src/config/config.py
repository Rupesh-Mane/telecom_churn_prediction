# Dataset Path

RAW_DATA_PATH = "data/raw/telecom_customer_churn.csv"
PROCESSED_DATA_PATH = "data/processed/telecom_customer_processed.csv"

# Saved ML artifacts
MODEL_PATH = "artifacts/final_model.pkl"
ENCODER_PATH = "artifacts/encoder.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

""" 
RAW_DATA_PATH
      ↓
Where is my original dataset?

MODEL_PATH
      ↓
Where is my final Random Forest model?

ENCODER_PATH
      ↓
Where is my saved encoder?

SCALER_PATH
      ↓
Where is my saved scaler?
"""


# ======================
# Model Settings
# ======================


TARGET_COLUMN = "Churn"

TEST_SIZE = 0.2

RANDOM_STATE = 42

if __name__ == "__main__":
    print(RAW_DATA_PATH)
    print(MODEL_PATH)
    print(ENCODER_PATH)
    print(SCALER_PATH)
    
#"Run this part only when I directly run this Python file."
# If another file imports config.py, the code inside that block will not run.
"""
For example:

from src.config.config import MODEL_PATH

This imports MODEL_PATH, but it won't automatically print everything inside the if __name__ == "__main__": block."""

#used the if __name__ == "__main__": block only when you want a small test.