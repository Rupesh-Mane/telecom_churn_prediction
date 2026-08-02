from pydantic import BaseModel


class CustomerData(BaseModel):

    # Numerical Features
    
    SeniorCitizen: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    MonthlyDataUsageGB: float
    NumberOfComplaints: int
    SatisfactionScore: int
    AvgCallDuration: float

    # Categorical Features
    
    Gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str 
    
"""Schema means before sending data to the model,we first check whether the user entered the correct information."""