# 📞 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple)
![Git](https://img.shields.io/badge/Git-Version%20Control-black?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end Machine Learning project for predicting telecom customer churn using customer demographic, service, usage, billing, and satisfaction information.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Project Objective](#-project-objective)
- [Dataset](#-dataset)
- [Dataset Features](#-dataset-features)
- [Data Quality Issues](#️-data-quality-issues)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Project Architecture](#️-project-architecture)
- [Project Structure](#-project-structure)
- [Notebook Workflow](#-notebook-workflow)
- [Data Preprocessing](#3️⃣-03-data-preprocessing)
- [Class Imbalance](#️-class-imbalance)
- [Model Training](#4️⃣-04-model-training)
- [Model Evaluation](#5️⃣-05-model-evaluation)
- [Model Performance](#-model-performance-summary)
- [Hyperparameter Tuning](#6️⃣-06-hyperparameter-tuning)
- [Final Model](#-final-model)
- [Model Serialization](#-model-serialization)
- [Installation](#-installation)
- [Usage](#️-usage)
- [Git and GitHub](#-git-and-github)
- [Large Model Files](#-large-model-files)
- [Planned Production Architecture](#-planned-production-architecture)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📊 Project Overview

Customer churn is a major business problem for telecom companies.

When customers leave a telecom company, the business loses recurring revenue and may need to spend additional resources to acquire new customers.

The objective of this project is to develop a Machine Learning classification model that predicts whether a telecom customer is likely to churn.

The model learns patterns from historical customer information such as:

- Customer demographics
- Customer tenure
- Phone services
- Internet services
- Online services
- Contract information
- Billing information
- Payment methods
- Monthly charges
- Total charges
- Monthly data usage
- Number of complaints
- Customer satisfaction
- Average call duration

The target variable is:

```text
Churn
```

Where:

```text
0 → Customer did not churn
1 → Customer churned
```

The project follows an end-to-end Machine Learning workflow from raw data analysis to model development and final model serialization.

---

## 💼 Business Problem

A telecom company wants to identify customers who are likely to leave its services.

The main business question is:

> Can we predict which customers are likely to churn before they leave the company?

A Machine Learning model can identify customers with a higher probability of churn.

The company can then take preventive actions such as:

- Offering personalized discounts
- Providing better customer support
- Offering improved service plans
- Contacting high-risk customers
- Improving customer satisfaction
- Providing customer retention offers

The overall business workflow is:

```text
Customer Data
      ↓
Machine Learning Model
      ↓
Predict Churn Risk
      ↓
Identify High-Risk Customers
      ↓
Customer Retention Strategy
      ↓
Reduce Customer Churn
```

---

## 🎯 Project Objective

The main objective of this project is to build a binary classification Machine Learning model that predicts customer churn.

The project includes:

- Understanding the business problem
- Understanding the raw dataset
- Performing data quality analysis
- Identifying missing values
- Checking duplicate records
- Performing Exploratory Data Analysis
- Performing correlation analysis
- Separating numerical and categorical features
- Handling missing values
- Encoding categorical variables
- Scaling numerical variables
- Splitting data into training and testing datasets
- Identifying class imbalance
- Handling class imbalance using Random Undersampling
- Training multiple Machine Learning models
- Evaluating model performance
- Comparing multiple models
- Selecting the best-performing model
- Performing hyperparameter tuning
  - Using GridSearchCV
  - Using Optuna
- Selecting the final model
- Saving the trained model using Pickle
- Preparing the project for future production deployment

---

## 📦 Dataset

The project uses a realistic telecom customer churn dataset containing approximately:

**60,000 customer records**

The dataset contains customer-level information related to:

- Customer demographics
- Telecom services
- Internet services
- Customer contracts
- Billing
- Payment methods
- Customer usage
- Customer complaints
- Customer satisfaction
- Customer churn status

The dataset was designed to represent realistic business data and includes data quality issues such as missing values and duplicate records.

---

## 📋 Dataset Features

The dataset contains the following types of information:

- CustomerID
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- MonthlyDataUsageGB
- NumberOfComplaints
- SatisfactionScore
- AvgCallDuration
- Churn

### 📚 Feature Description

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| SeniorCitizen | Indicates whether the customer is a senior citizen |
| Partner | Indicates whether the customer has a partner |
| Dependents | Indicates whether the customer has dependents |
| Tenure | Number of months the customer has stayed with the company |
| PhoneService | Indicates whether the customer has phone service |
| MultipleLines | Indicates whether the customer has multiple phone lines |
| InternetService | Type of internet service |
| OnlineSecurity | Indicates whether online security is subscribed |
| OnlineBackup | Indicates whether online backup is subscribed |
| DeviceProtection | Indicates whether device protection is subscribed |
| TechSupport | Indicates whether technical support is subscribed |
| StreamingTV | Indicates whether streaming TV is subscribed |
| StreamingMovies | Indicates whether streaming movies are subscribed |
| Contract | Customer contract type |
| PaperlessBilling | Indicates whether paperless billing is enabled |
| PaymentMethod | Customer payment method |
| MonthlyCharges | Customer's monthly charges |
| TotalCharges | Total charges paid by the customer |
| MonthlyDataUsageGB | Customer's monthly data usage |
| NumberOfComplaints | Number of customer complaints |
| SatisfactionScore | Customer satisfaction score |
| AvgCallDuration | Average customer call duration |
| Churn | Target variable indicating whether the customer churned |

---

## ⚠️ Data Quality Issues

The raw dataset was analyzed for common real-world data quality problems.

### Missing Values

The dataset contains missing values in different columns.

Missing values were identified during the Data Understanding stage and handled during Data Preprocessing.

Numerical and categorical features were treated separately during preprocessing.

### Duplicate Records

The dataset was checked for duplicate records.

Duplicate records can negatively affect model training because repeated observations may introduce bias.

Therefore, duplicate records were analyzed during the Data Understanding stage.

### Numerical Features

The numerical features include:

- SeniorCitizen
- Tenure
- MonthlyCharges
- TotalCharges
- MonthlyDataUsageGB
- NumberOfComplaints
- SatisfactionScore
- AvgCallDuration

These features were processed and scaled during preprocessing.

### Categorical Features

The categorical features include:

- Gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

These features were converted into numerical representations using One-Hot Encoding.

---

## 🔄 Machine Learning Workflow

The complete Machine Learning workflow followed in this project is:

```text
Raw Dataset
      ↓
Data Understanding
      ↓
Data Quality Analysis
      ↓
Exploratory Data Analysis
      ↓
Correlation Analysis
      ↓
Train-Test Split
      ↓
Missing Value Handling
      ↓
Categorical Encoding
      ↓
Numerical Feature Scaling
      ↓
Processed Dataset
      ↓
Class Imbalance Analysis
      ↓
Random Undersampling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Hyperparameter Tuning
      ↓
GridSearchCV
      ↓
Optuna Optimization
      ↓
Final Model Selection
      ↓
Model Serialization
      ↓
Future Production Deployment
```

---

## 🏗️ Project Architecture

The overall Machine Learning development workflow is:

```text
                    RAW DATASET
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Understanding  │
              │       & EDA         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Correlation         │
              │ Analysis            │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Preprocessing  │
              │                     │
              │ Missing Values      │
              │ Encoding            │
              │ Scaling             │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Train-Test Split    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Class Imbalance     │
              │ Handling            │
              │                     │
              │ Undersampling       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Training      │
              │                     │
              │ Logistic Regression │
              │ Decision Tree       │
              │ Random Forest       │
              │ Gradient Boosting   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Evaluation    │
              │                     │
              │ Accuracy            │
              │ Precision           │
              │ Recall              │
              │ F1-Score            │
              │ ROC-AUC             │
              │ Confusion Matrix    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Best Baseline Model │
              │ Random Forest       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Hyperparameter      │
              │ Tuning              │
              │                     │
              │ GridSearchCV        │
              │ Optuna              │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Final Model         │
              │ Random Forest       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Serialization │
              │ final_model.pkl     │
              └─────────────────────┘
```

---

## 📁 Project Structure

```text
telecom_churn_prediction/
│
├── data/
│   │
│   ├── raw/
│   │   └── customer_customer_churn.csv
│   │
│   ├── processed/
│   │   ├── telecom_customer_processed.csv
│   │   ├── x_train.csv
│   │   ├── y_train.csv
│   │   ├── x_test.csv
│   │   └── y_test.csv
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_correlation_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_final_model.ipynb
│
├── models/
│   ├── best_rf_model.pkl
│   └── final_model.pkl
│
├── reports/
│   └── plots/
│       ├── churn_distribution.png
│       ├── correlation_matrix.png
│       └── other_visualizations.png
│
├── src/
│   │
│   ├── config/
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── transformation.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── utils/
│       └── helpers.py
│
├── deployment/
│   │
│   ├── fastapi/
│   │   └── main.py
│   │
│   └── streamlit/
│       └── app.py
│
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

> **Note:** The `src/` and `deployment/` directories represent the planned production-oriented architecture. The current project development has been primarily performed through the numbered notebooks.

### 📂 Folder Explanation

**`data/`**
Contains datasets used throughout the Machine Learning project.

**`data/raw/`**
Contains the original raw dataset: `data/raw/customer_churn_telecom.csv`
The raw dataset should remain unchanged to maintain reproducibility.

**`data/processed/`**
Contains cleaned and transformed datasets used for Machine Learning, e.g. `telecom_customer_processed.csv`, `x_train.csv`, `y_train.csv`, `x_test.csv`, `y_test.csv`.

**`data/external/`**
Reserved for external datasets that may be used in future development.

**`notebooks/`**
Contains the sequential notebooks used for experimentation, analysis, preprocessing, model development, evaluation, and tuning.

**`models/`**
Contains serialized Machine Learning model artifacts, e.g. `best_rf_model.pkl`, `final_model.pkl`.

**`reports/plots/`**
Contains visualizations generated during data analysis, e.g. `churn_distribution.png`, `correlation_matrix.png`.

**`src/`**
Planned location for reusable, production-ready Python code:
- Data ingestion
- Data validation
- Data transformation
- Feature engineering
- Model training
- Model evaluation
- Prediction
- Utility functions

**`deployment/`**
Planned location for deployment applications:
- FastAPI backend
- Streamlit frontend

---

## 📓 Notebook Workflow

The notebooks are executed in the following sequence:

```text
01_data_understanding.ipynb
        ↓
02_correlation_analysis.ipynb
        ↓
03_data_preprocessing.ipynb
        ↓
04_model_training.ipynb
        ↓
05_model_evaluation.ipynb
        ↓
06_hyperparameter_tuning.ipynb
        ↓
07_final_model.ipynb
```

### 1️⃣ 01 Data Understanding

The first notebook focuses on understanding the raw dataset.

Activities include:

- Loading the raw dataset
- Checking dataset shape
- Displaying first and last rows
- Checking column names
- Checking data types
- Checking missing values
- Checking duplicate records
- Identifying numerical columns
- Identifying categorical columns
- Understanding the target variable

The purpose is to understand the dataset before applying Machine Learning techniques.

### 2️⃣ 02 Correlation Analysis

The second notebook focuses on Exploratory Data Analysis and correlation analysis.

Activities include:

- Analyzing numerical features
- Analyzing categorical features
- Generating correlation matrix
- Visualizing feature relationships
- Analyzing relationships between numerical features and churn
- Saving plots to the `reports/plots/` directory

Numerical variables analyzed include: SeniorCitizen, Tenure, MonthlyCharges, TotalCharges, MonthlyDataUsageGB, NumberOfComplaints, SatisfactionScore, AvgCallDuration.

Correlation analysis was used as an exploratory technique to understand relationships between variables. Correlation alone was not treated as the final measure of feature importance.

### 3️⃣ 03 Data Preprocessing

The third notebook prepares the dataset for Machine Learning.

The preprocessing workflow is:

```text
Raw Data
    ↓
Missing Value Handling
    ↓
Separate Numerical and Categorical Features
    ↓
Categorical Encoding
    ↓
Numerical Scaling
    ↓
Combine Processed Features
    ↓
Prepare Final Dataset
```

#### 🧹 Missing Value Handling

**Numerical Features**

Missing numerical values were handled using imputation.

```python
from sklearn.impute import SimpleImputer

imp_num_data = SimpleImputer(
    strategy="mean"
)
```

**Categorical Features**

Missing categorical values were handled using the most frequent category.

```python
imp_cat_data = SimpleImputer(
    strategy="most_frequent"
)
```

#### 🔠 Categorical Encoding

Categorical variables were converted into numerical features using One-Hot Encoding.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    drop="first",
    sparse_output=False
)
```

This converts categorical values into numerical columns that Machine Learning algorithms can process.

#### 📏 Feature Scaling

Numerical features were scaled using standardization.

Scaling helps bring numerical features to a comparable scale. For example, `Tenure`, `MonthlyCharges`, and `TotalCharges` may have very different numerical ranges. Scaling transforms these features into a standardized range suitable for Machine Learning algorithms.

#### 🎯 Target Variable

The target variable is `Churn`, representing:

```text
0 → No Churn
1 → Churn
```

The target variable was separated from the input features before model training.

### 4️⃣ 04 Model Training

This notebook focuses on training multiple Machine Learning models.

The models trained were:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The training workflow is:

```text
Processed Dataset
       ↓
Separate X and y
       ↓
Train-Test Split
       ↓
Handle Class Imbalance
       ↓
Train Multiple Models
       ↓
Save Trained Models
```

#### ✂️ Train-Test Split

The dataset was divided into:

**Training Data** — used for model training, class imbalance handling, and hyperparameter optimization.

**Testing Data** — used for final evaluation and measuring performance on unseen data.

The test dataset was kept separate from model training.

#### ⚖️ Class Imbalance

The target distribution was:

```text
Churn = 0 → 51,548
Churn = 1 →  7,552
```

This indicates that the target variable is imbalanced. The majority class is `0` (No Churn) and the minority class is `1` (Churn).

If class imbalance is ignored, a model may become biased toward the majority class.

#### 🔽 Random Undersampling

Random Undersampling was used to address class imbalance in the training data.

```text
Majority Class
      ↓
Randomly Remove Some Majority Samples
      ↓
Balance Training Data
      ↓
Train Machine Learning Model
```

The important principle is: **Random Undersampling should be applied only to the training data.** The original test data should remain untouched so that final evaluation represents the real-world class distribution.

### 5️⃣ 05 Model Evaluation

This notebook evaluates the trained Machine Learning models.

The evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

#### 📊 Classification Report

The classification report provides Precision, Recall, F1-Score, Support, and Accuracy.

For a churn problem, Recall can be especially important because missing a customer who is actually going to churn can be costly.

#### 🧮 Confusion Matrix

```text
                    Predicted
                 No Churn   Churn

Actual No Churn      TN       FP

Actual Churn         FN       TP
```

Where: TP → True Positive, TN → True Negative, FP → False Positive, FN → False Negative.

#### 📈 ROC-AUC

ROC-AUC was used to compare the ability of different models to distinguish between churn and non-churn customers.

The baseline model comparison was:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.618890 |
| Decision Tree | 0.641162 |
| Random Forest | 0.819863 |
| Gradient Boosting | 0.655263 |

The best baseline model was **Random Forest**, with a ROC-AUC score of **0.819863**.

Therefore, Random Forest was selected for further hyperparameter tuning.

### 6️⃣ 06 Hyperparameter Tuning

This notebook focuses on improving the performance of the Random Forest model.

Two hyperparameter optimization approaches were explored:

- GridSearchCV
- Optuna

#### 🔍 GridSearchCV

GridSearchCV searches through predefined combinations of hyperparameters.

Random Forest hyperparameters can include: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.

The goal is to identify the hyperparameter combination that produces the best model performance.

#### 🧠 Optuna

Optuna was used for automated hyperparameter optimization.

```text
Optuna
   ↓
Create Trial
   ↓
Select Hyperparameters
   ↓
Train Random Forest
   ↓
Evaluate Model
   ↓
Calculate ROC-AUC
   ↓
Return Score
   ↓
Run Next Trial
   ↓
Find Best Hyperparameters
```

The objective was to maximize ROC-AUC.

The best Random Forest model discovered during optimization was saved as `models/best_rf_model.pkl`.

### 7️⃣ 07 Final Model

This notebook focuses on final model selection and model serialization.

```text
Load Test Data
      ↓
Load Tuned Random Forest
      ↓
Generate Predictions
      ↓
Generate Classification Report
      ↓
Generate Confusion Matrix
      ↓
Calculate ROC-AUC
      ↓
Compare Original vs Tuned Model
      ↓
Select Final Model
      ↓
Save Final Model
```

The final model is saved as `models/final_model.pkl`.

---

## 🏆 Final Model

The best baseline model identified during model evaluation was **Random Forest Classifier**, with a baseline ROC-AUC of **0.819863**.

Random Forest was then selected for hyperparameter tuning using GridSearchCV and Optuna.

The final tuned model was saved as `models/final_model.pkl`.

> The final tuned ROC-AUC should be updated in this README after completing the final evaluation of the tuned model on the untouched test dataset.

---

## 💾 Model Serialization

The final trained model was saved using Pickle.

```python
import pickle

with open("../models/final_model.pkl", "wb") as file:
    pickle.dump(best_rf_model, file)
```

The model can later be loaded using:

```python
import pickle

with open("../models/final_model.pkl", "rb") as file:
    model = pickle.load(file)
```

This allows the trained model to be reused without retraining it every time.

### 🔐 Data Leakage Prevention

Data leakage prevention is an important part of the project.

```text
Original Dataset
       ↓
Train-Test Split
       ↓
Training Data
       ↓
Preprocessing / Sampling
       ↓
Model Training

Test Data
       ↓
Kept Separate
       ↓
Final Evaluation
```

The test dataset is not used for training the final model. This ensures that final evaluation represents performance on unseen data.

---

## 📊 Model Performance Summary

The initial baseline model performance was:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.618890 |
| Decision Tree | 0.641162 |
| Random Forest | 0.819863 |
| Gradient Boosting | 0.655263 |

**Best Baseline Model:** Random Forest
**Baseline ROC-AUC:** 0.819863

**Next Step:** The Random Forest model was selected for GridSearchCV + Optuna hyperparameter optimization → Final Model.

---

## 🛠️ Technologies Used

**Programming**
- Python 3.11

**Data Processing**
- Pandas
- NumPy

**Data Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn

**Imbalanced Learning**
- imbalanced-learn

**Hyperparameter Optimization**
- GridSearchCV
- Optuna

**Model Serialization**
- Pickle

**Development**
- Jupyter Notebook
- VS Code

**Version Control**
- Git
- GitHub

**Planned Deployment**
- FastAPI
- Streamlit
- Docker
- Docker Compose

---

## 📦 Installation

**1. Clone the Repository**

```bash
git clone https://github.com/Rupesh-Mane/telecom_churn_prediction.git
```

**2. Navigate to the Project**

```bash
cd telecom_churn_prediction
```

**3. Create Virtual Environment**

```bash
python -m venv telecomvenv
```

**4. Activate Virtual Environment**

On Windows:

```bash
telecomvenv\Scripts\activate
```

**5. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the notebooks in the following order:

```text
01_data_understanding.ipynb
        ↓
02_correlation_analysis.ipynb
        ↓
03_data_preprocessing.ipynb
        ↓
04_model_training.ipynb
        ↓
05_model_evaluation.ipynb
        ↓
06_hyperparameter_tuning.ipynb
        ↓
07_final_model.ipynb
```

To load the final trained model:

```python
import pickle

with open("models/final_model.pkl", "rb") as file:
    model = pickle.load(file)
```

The input data must be processed using the same preprocessing steps used during model training before generating predictions.

---

## 📌 Git and GitHub

Git was used for version control throughout the project.

The project tracks:

- Data analysis notebooks
- Preprocessing notebooks
- Model training
- Model evaluation
- Hyperparameter tuning
- Optuna optimization
- Final model development
- Project documentation

The standard Git workflow is:

```text
Working Directory
       ↓
git add
       ↓
git commit
       ↓
git push
       ↓
GitHub Repository
```

Example:

```bash
git add .
git commit -m "Add model evaluation"
git push
```

---

## 📦 Large Model Files

The trained Random Forest model files are large Pickle artifacts.

GitHub has a maximum file size limit of 100 MB for individual files. Therefore, large model files should not normally be committed directly to a standard GitHub repository.

For production projects, model artifacts can be managed using:

- Git LFS
- Cloud Object Storage
- MLflow
- MLflow Model Registry

A `.gitignore` entry can be used to prevent Pickle models from being tracked:

```text
models/*.pkl
```

The recommended architecture is:

```text
GitHub
   │
   ├── Source Code
   ├── Notebooks
   ├── Configuration
   └── Documentation

Model Storage
   │
   ├── final_model.pkl
   └── Model Versions
```

This keeps the Git repository lightweight while allowing large model artifacts to be managed separately.

---

## 🚀 Planned Production Architecture

The next stage of this project is to convert the Machine Learning model into a production-style application.

```text
                    USER
                      │
                      ▼
             ┌─────────────────┐
             │    Streamlit    │
             │    Frontend     │
             └────────┬────────┘
                      │
                      │ HTTP Request
                      ▼
             ┌─────────────────┐
             │     FastAPI     │
             │     Backend     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Preprocessing  │
             │    Pipeline     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  final_model    │
             │ Random Forest   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Prediction   │
             │ Churn / No Churn│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    Streamlit    │
             │ Display Result  │
             └─────────────────┘
```

### ⚡ FastAPI

FastAPI will be used as the backend Machine Learning prediction API.

The API will:

- Receive customer information
- Validate input data
- Apply required preprocessing
- Load the trained Machine Learning model
- Generate a prediction
- Return the prediction as JSON

Planned endpoint: `POST /predict`

```text
Customer Data
     ↓
FastAPI
     ↓
Input Validation
     ↓
Preprocessing
     ↓
Model Prediction
     ↓
JSON Response
```

Example response:

```json
{
    "prediction": 1,
    "result": "Customer is likely to churn"
}
```

### 🖥️ Streamlit

Streamlit will be used as the frontend user interface.

The user will enter customer information such as: Gender, Senior Citizen, Tenure, Internet Service, Contract, Monthly Charges, Total Charges, Satisfaction Score, Number of Complaints.

Streamlit will send the data to the FastAPI backend.

```python
import requests

response = requests.post(
    "http://fastapi:8000/predict",
    json=data
)
```

FastAPI will process the request and return the prediction, and Streamlit will display the result — e.g. *"Customer is likely to churn"* or *"Customer is unlikely to churn"*.

### 🐳 Docker

Docker will be used to containerize the application.

```text
Docker Compose
│
├── FastAPI Container
│      │
│      └── Prediction API
│
└── Streamlit Container
       │
       └── User Interface
```

FastAPI and Streamlit will run as separate containers, communicating through a Docker network. Streamlit will communicate with FastAPI using the Docker service name: `http://fastapi:8000/predict`, where `fastapi` is the service name defined in `docker-compose.yml`.

### 🔄 Final Production Workflow

```text
Customer
   ↓
Streamlit UI
   ↓
Enter Customer Information
   ↓
HTTP POST Request
   ↓
FastAPI API
   ↓
Validate Input
   ↓
Preprocess Input
   ↓
Load final_model.pkl
   ↓
Generate Prediction
   ↓
Return JSON Response
   ↓
Streamlit Displays Result
```

---

## 🔮 Future Improvements

**Machine Learning Improvements**
- Build a reusable preprocessing Pipeline
- Save preprocessing objects
- Add probability-based churn prediction
- Perform prediction threshold tuning
- Optimize Recall for churn customers
- Perform advanced feature engineering
- Perform feature selection
- Test additional Machine Learning algorithms
- Perform cross-validation
- Improve model interpretability

**Software Engineering Improvements**
- Move notebook code into the `src/` directory
- Create reusable Python modules
- Add unit tests
- Add integration tests
- Add logging
- Add exception handling
- Add configuration management
- Improve project documentation

**Deployment Improvements**
- Complete FastAPI backend
- Complete Streamlit frontend
- Dockerize FastAPI
- Dockerize Streamlit
- Create Docker Compose configuration
- Deploy the application to the cloud

**MLOps Improvements**
- MLflow experiment tracking
- Model versioning
- Model registry
- Data drift detection
- Model monitoring
- Automated model retraining
- Experiment tracking
- Model performance monitoring

**DevOps Improvements**
- GitHub Actions
- CI/CD pipeline
- Automated testing
- Automated Docker image builds
- Cloud deployment

---

## 🎓 Project Learning Outcomes

This project demonstrates practical knowledge of:

Python for Machine Learning, Pandas, NumPy, Data Cleaning, Missing Value Handling, Exploratory Data Analysis, Correlation Analysis, Categorical Feature Encoding, Numerical Feature Scaling, Train-Test Splitting, Class Imbalance, Random Undersampling, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Classification Metrics, Confusion Matrix, ROC-AUC, Model Comparison, GridSearchCV, Optuna, Hyperparameter Optimization, Model Serialization, Git, GitHub, FastAPI, Streamlit, Docker, Docker Compose, and Production Machine Learning Architecture.

---

## 🏢 Real-World ML Engineering Perspective

This project demonstrates how a Machine Learning project can evolve from experimentation into a production-ready system.

The complete Machine Learning lifecycle can be represented as:

```text
Business Problem
       ↓
Data Collection
       ↓
Data Understanding
       ↓
Data Validation
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Class Imbalance Handling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
Final Model Selection
       ↓
Model Serialization
       ↓
API Development
       ↓
Frontend Development
       ↓
Docker Containerization
       ↓
Cloud Deployment
       ↓
Model Monitoring
       ↓
Model Retraining
```

The current Machine Learning development workflow has been completed through: Data Understanding, Exploratory Data Analysis, Correlation Analysis, Data Preprocessing, Class Imbalance Handling, Model Training, Model Evaluation, Model Comparison, Hyperparameter Tuning, Optuna Optimization, Final Model Selection, and Model Serialization.

The next stage is production deployment using FastAPI + Streamlit + Docker + Docker Compose.

---

## ⭐ Project Highlights

- End-to-end Telecom Customer Churn Prediction
- Approximately 60,000 customer records
- Realistic telecom customer data
- Data quality analysis
- Missing value handling
- Duplicate record analysis
- Exploratory Data Analysis
- Correlation Analysis
- One-Hot Encoding
- Numerical Feature Scaling
- Train-Test Split
- Class Imbalance Analysis
- Random Undersampling
- Multiple Machine Learning Algorithms
- Model Comparison
- ROC-AUC Evaluation
- Confusion Matrix
- Classification Report
- GridSearchCV
- Optuna Hyperparameter Optimization
- Random Forest Model Selection
- Final Model Serialization
- Git Version Control
- GitHub Repository
- Production-Oriented Project Structure
- Planned FastAPI Backend
- Planned Streamlit Frontend
- Planned Docker Deployment
- Planned MLOps Improvements

---

## 🤝 Contributing

Contributions are welcome.

To contribute:

**1. Fork the Repository**

Create your own fork of the project.

**2. Create a New Branch**

```bash
git checkout -b feature/your-feature
```

**3. Make Your Changes**

Implement your changes and test them.

**4. Commit Your Changes**

```bash
git add .
git commit -m "Add your feature"
```

**5. Push Your Branch**

```bash
git push origin feature/your-feature
```

**6. Open a Pull Request**

Create a Pull Request and describe the changes you made.

---

## 📜 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project according to the terms of the MIT License.

---

## 📬 Contact

**Author:** Rupesh Mane

**GitHub:** [https://github.com/Rupesh-Mane](https://github.com/Rupesh-Mane)

**Project Repository:** [https://github.com/Rupesh-Mane/telecom_churn_prediction](https://github.com/Rupesh-Mane/telecom_churn_prediction)

---

## ⭐ Acknowledgement

This project was developed as an end-to-end Machine Learning project to demonstrate practical skills in Data Analysis, Data Preprocessing, Machine Learning, Class Imbalance Handling, Model Evaluation, Hyperparameter Optimization, Model Serialization, Git and GitHub, and Production-Oriented ML Architecture.

If you find this project useful, consider giving the repository a ⭐ on GitHub.