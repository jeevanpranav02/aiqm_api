import os
import traceback

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"  # Suppress TensorFlow warnings and info messages

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)


class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.sc = MinMaxScaler(feature_range=(0, 1))
        df = pd.read_csv("data/AirQDataset.csv")
        # print(df.head())
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date

        df_date = pd.DataFrame(df.groupby("Date")["PM2.5"].mean())
        df_date

        dataset = df_date.values
        self.sc.fit(dataset)

    def parse_input(self, input_data: str):
        input_data_str = str(input_data)
        # Parsing input
        string_data_list = input_data_str[1 : len(input_data_str) - 1].split(",")
        print("input_data in list : ", string_data_list)
        input_data = [[float(i.strip())] for i in string_data_list]
        print("input_data as float list : ", input_data)

        # Preparing model for prediction with a sample input
        input_sequence = np.array(
            [
                [0.1490341],
                [0.11468296],
                [0.14883455],
                [0.16959085],
                [0.15208376],
                [0.13489773],
                [0.13458441],
                [0.08793108],
                [0.1269797],
                [0.14903182],
                [0.15870598],
                [0.15524401],
                [0.1159401],
                [0.10717495],
                [0.12105375],
                [0.08491009],
                [0.0568121],
                [0.04234922],
                [0.03699188],
                [0.0461129],
                [0.06506279],
                [0.07323999],
                [0.04883218],
                [0.04588855],
                [0.04567193],
                [0.05728015],
                [0.06375924],
                [0.07257467],
                [0.07021512],
                [0.06278834],
                [0.06367414],
                [0.0574194],
                [0.03340227],
                [0.0330348],
                [0.03442732],
                [0.04200883],
                [0.04303001],
                [0.05825491],
                [0.06386754],
                [0.07139595],
                [0.0878924],
                [0.09915249],
                [0.09971336],
                [0.09254575],
                [0.10436284],
                [0.1286662],
                [0.12440354],
                [0.12198597],
                [0.12498375],
                [0.11762273],
                [0.12216003],
                [0.10557271],
                [0.11847217],
                [0.10853829],
                [0.11139894],
                [0.0621385],
                [0.05134646],
                [0.07303884],
                [0.0685557],
                [0.08393588],
            ]
        )
        input_sequence = input_sequence.reshape(
            1, input_sequence.shape[0], input_sequence.shape[1]
        )
        prediction = self.model.predict(input_sequence)
        prediction = self.sc.inverse_transform(prediction)
        # Parsing the string to float
        input_sequence = np.array(input_data)
        print("input_sequence (before reshaping): ", input_sequence)

        # Check for NaN values in input_sequence
        if np.isnan(input_sequence).any():
            print("Input sequence contains NaN values!")
            return None

        for i in range(len(input_sequence)):
            input_sequence[i] = input_sequence[i] / 263
        input_sequence = input_sequence.reshape(
            1, input_sequence.shape[0], input_sequence.shape[1]
        )
        print("input_sequence (after reshaping): ", input_sequence)
        return input_sequence

    def predict(self, input_data):
        try:
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return None


model_path = "models/model69.h5"

model_predictor = ModelPredictor(model_path)


@app.route("/")
def index():
    return "Welcome to the API!"


@app.route("/predict", methods=["POST"])
def predict_method():
    try:
        # Get input data from request JSON
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract input sequence from JSON data
        input_sequence = request_data.get("input_sequence")
        if not input_sequence:
            return jsonify({"error": "Input sequence not provided"}), 400

        # Parse input data
        data = model_predictor.parse_input(input_sequence)
        print("parsed data inside api : ", data)

        # Make prediction
        prediction = model_predictor.predict(data)
        prediction = model_predictor.sc.inverse_transform(prediction)
        if prediction is not None:
            print("Prediction:", prediction[0][0])

        return jsonify(
            {
                "prediction": str(prediction[0][0]),
            }
        )
    except Exception as e:
        print(str(e))
        return (
            jsonify(
                {
                    "error": str(e),
                    "input_sequence": input_sequence,
                    "input_sequence_type": str(type(input_sequence)),
                }
            ),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
