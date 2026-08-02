from src.data.transformation import transform_data
from src.models.predict_model import predict


class PredictionPipeline:

    def __init__(self):
        pass

    def predict(self, data):

        # Transform Raw Customer Data
        transformed_data = transform_data(data)

        # Predict
        prediction = predict(transformed_data)

        return prediction


if __name__ == "__main__":

    print("=" * 50)
    print("PREDICTION PIPELINE READY")
    print("=" * 50)