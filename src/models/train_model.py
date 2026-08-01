"""   
=============================
train_model.py
=============================

Purpose:
--------
This file is responsible for training the final Machine Learning model.

Why do we need it?
------------------
During experimentation, model training is done inside Jupyter notebooks.

In a production project, we should not train the model manually using notebooks every time.

Instead, train_model.py automates the training process.

Typical Workflow:
-----------------
1. Load processed training data.
2. Train the selected Machine Learning model.
3. Save the trained model as final_model.pkl.
4. The saved model is later used by FastAPI and Streamlit for prediction.

Current Project:
----------------
In this project, the model has already been trained inside the notebooks.

The following files are already available:

- artifacts/final_model.pkl
- artifacts/encoder.pkl
- artifacts/scaler.pkl

Therefore, train_model.py is not required for prediction.

Future Use:
-----------
If a new telecom dataset is received in the future, train_model.py can be executed to retrain the model automatically without opening the notebooks.

Project Flow:
-------------
Raw Dataset
      ↓
Data Preprocessing
      ↓
Model Training (train_model.py)
      ↓
Save final_model.pkl
      ↓
Prediction (predict_model.py)
      ↓
FastAPI
      ↓
Streamlit
      ↓
Docker


"""