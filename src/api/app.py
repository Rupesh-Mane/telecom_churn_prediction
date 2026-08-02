from fastapi import FastAPI
import pandas as pd

from src.api.schema import CustomerData
from src.pipeline.prediction_pipeline import PredictionPipeline

# Create FastAPI App
app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    version="1.0"
)

# Create Prediction Pipeline
pipeline = PredictionPipeline()


# Home Route
@app.get("/")
def home():

    return {
        "message": "Welcome to Telecom Customer Churn Prediction API"
    }


# Health Check Route
@app.get("/health")
def health():

    return {
        "status": "API is Running Successfully"
    }


# Prediction Route
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert Pydantic Model to Dictionary
    data = customer.model_dump()

    # Convert Dictionary to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = pipeline.predict(df)

    # Return Prediction
    return {
        "prediction": int(prediction[0])
    }


if __name__ == "__main__":

    print("FastAPI Application Ready") 