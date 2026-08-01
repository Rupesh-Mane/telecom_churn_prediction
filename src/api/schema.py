from pydantic import BaseModel


class CustomerData(BaseModel):

    Gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Tenure: int
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
    MonthlyCharges: float
    TotalCharges: float
    MonthlyDataUsageGB: float
    NumberOfComplaints: int
    SatisfactionScore: int
    AvgCallDuration: float