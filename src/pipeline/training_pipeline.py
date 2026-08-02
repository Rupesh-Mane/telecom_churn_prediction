"""
training_pipeline.py

Purpose:
--------
This module is responsible for running the complete model training pipeline.

Typical workflow:
1. Load Data
2. Validate Data
3. Transform Data
4. Train Model
5. Evaluate Model
6. Save Final Model

In this project, the complete training pipeline has already been executed in the following Jupyter notebooks:

01_data_understanding.ipynb
02_eda.ipynb
03_data_preprocessing.ipynb
04_model_training.ipynb
05_model_evaluation.ipynb
06_hyperparameter_tuning.ipynb
07_final_model.ipynb

The final trained model is saved as:

artifacts/final_model.pkl

Therefore, this file is not used during FastAPI prediction.
"""


def training_pipeline():
    print("Model training pipeline has already been completed in Jupyter Notebook.")


if __name__ == "__main__":
    training_pipeline()















"""
Training Pipeline

This file controls the complete training workflow.

Current Project:
The model has already been trained in Jupyter notebooks.

Future Purpose:
If a new dataset is available, this pipeline will automate the
entire training process.


from src.data.ingestion import load_data
from src.data.validation import validate_data
from src.data.transformation import transform_data


class TrainingPipeline:

    def run_pipeline(self):

        print("Loading Data...")
        load_data()

        print("Validating Data...")
        validate_data()

        print("Transforming Data...")
        transform_data()

        print("Training Pipeline Completed")


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run_pipeline()
   """ 