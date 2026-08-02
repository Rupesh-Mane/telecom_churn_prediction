import streamlit as st
import requests

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱"
)

st.title("📱 Telecom Customer Churn Prediction")

st.write("Enter Customer Details")
API_URL = "http://127.0.0.1:8000/predict"   # Local->This works only on your local machine, but not inside Docker.When using Docker Compose, change it to:
# API_URL = "http://fastapi:8000/predict"   # Docker


# ==========================
# Numerical Features
# ==========================

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1],
    help="0 = No, 1 = Yes"
)

Tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    help="Example: 24 months"
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    help="Example: 75.50"
)

TotalCharges = st.number_input(
    "Total Charges",
    help="Example: 1800.50"
)

MonthlyDataUsageGB = st.number_input(
    "Monthly Data Usage (GB)",
    help="Example: 120 GB"
)

NumberOfComplaints = st.number_input(
    "Number Of Complaints",
    min_value=0,
    help="Example: 2 complaints"
)

SatisfactionScore = st.number_input(
    "Satisfaction Score",
    min_value=1,
    max_value=10,
    help="Rate from 1 (Lowest) to 10 (Highest)"
)

AvgCallDuration = st.number_input(
    "Average Call Duration (Minutes)",
    help="Example: 15.5 minutes"
)


Gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    help="Select the customer's gender."
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"],
    help="Does the customer have a partner?"
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"],
    help="Does the customer have dependents?"
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"],
    help="Does the customer have a phone service?"
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"],
    help="Select whether the customer has multiple phone lines."
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
    help="Select the customer's internet service type."
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"],
    help="Does the customer subscribe to online security?"
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"],
    help="Does the customer subscribe to online backup?"
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"],
    help="Does the customer have device protection?"
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"],
    help="Does the customer have technical support?"
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"],
    help="Does the customer use streaming TV?"
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"],
    help="Does the customer use streaming movies?"
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"],
    help="Select the customer's contract type."
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
    help="Does the customer use paperless billing?"
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ],
    help="Select the customer's payment method."
)

# ==========================
# Predict Button
# ==========================

if st.button("Predict"):

    data = {
        "SeniorCitizen": SeniorCitizen,
        "Tenure": Tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "MonthlyDataUsageGB": MonthlyDataUsageGB,
        "NumberOfComplaints": NumberOfComplaints,
        "SatisfactionScore": SatisfactionScore,
        "AvgCallDuration": AvgCallDuration,
        "Gender": Gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod
    }

    response = requests.post(
        API_URL,
        json=data
    )

    
    result = response.json()

    if result["prediction"] == 1:
        st.write("Customer will Churn")
    else:
        st.write("Customer will NOT Churn")