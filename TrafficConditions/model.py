import json
import numpy as np
import joblib
import tensorflow as tf

class TrafficPredictionModel:
    def __init__(self, model_path=None, scaler_path=None, traffic_data_path=None):
        self.model = None
        self.scaler = None
        self.traffic_data = None

        # Load model, scaler, and traffic data if paths are provided
        if model_path:
            self.set_model(model_path)
        if scaler_path:
            self.set_scaler(scaler_path)
        if traffic_data_path:
            self.set_traffic_data(traffic_data_path)

    # Setter for the model
    def set_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    # Setter for the scaler
    def set_scaler(self, scaler_path):
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")

    # Setter for traffic data
    def set_traffic_data(self, traffic_data_path):
        with open(traffic_data_path, "r") as f:
            traffic_data = json.load(f)
        self.traffic_data = np.array(list(traffic_data.values())).reshape(-1, 1)
        print(f"Traffic data loaded from {traffic_data_path}")

    # Method to predict traffic conditions
    def predict_traffic(self):
        if self.model is None or self.scaler is None or self.traffic_data is None:
            raise ValueError("Model, scaler, or traffic data not set. Use the appropriate setters first.")

        # Scale the traffic data
        traffic_data_scaled = self.scaler.transform(self.traffic_data)

        # Reshape the traffic data for prediction
        traffic_data_np = traffic_data_scaled.reshape(-1, 24, 1)

        # Predict traffic conditions
        traffic_predictions = self.model.predict(traffic_data_np)

        # Rescale predictions back to the original scale
        traffic_predictions_rescaled = self.scaler.inverse_transform(traffic_predictions.reshape(-1, 24))

        # Flatten the predictions to get a continuous time series
        flattened_predictions = traffic_predictions_rescaled.flatten()

        return flattened_predictions
    
    def map_traffic(self, traffic_time):
        mapping = {'Bad': 350.0, 'High': 325.0, 'Moderate': 300.0, 'Low': 275.0, 'Good': 250.0}

        if traffic_time > mapping["Bad"]:
            return 1.5
        elif traffic_time > mapping["High"]:
            return 1.25
        elif traffic_time > mapping["Moderate"]:
            return 1.1
        elif traffic_time > mapping["Low"]:
            return 1.05
        else:
            return 1.0
        
    