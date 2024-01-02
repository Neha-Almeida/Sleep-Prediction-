from flask import Flask, request, jsonify
import numpy as np
import pickle
import joblib

# Load model
rfc_model = joblib.load("Models/RFC_MODEL.pkl")
scaler_model = joblib.load("Models/SCALER_MODEL.pkl")


def retrun_prediction(model, scaler, sample_json):
    a = []

    for i in list(sample_json.values()):
        a.append(int(i))

    data = [a]

    scale_data = scaler.transform(data)

    idx = model.predict(scale_data)

    labels = ["None", "Insomaina", "Apnea"]

    return labels[idx[0]]


app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


@app.route('/prediction', methods=['POST'])
def predict_flower():
    # RECIEVE THE REQUEST
    content = request.json

    # PRINT THE DATA PRESENT IN THE REQUEST
    print("[INFO] Request: ", content)

    # PREDICT THE CLASS USING HELPER FUNCTION
    results = retrun_prediction(model=rfc_model,
                                scaler=scaler_model,
                                sample_json=content)

    # PRINT THE RESULT
    print("[INFO] Responce: ", results)

    # SEND THE RESULT AS JSON OBJECT
    return jsonify(results)


if __name__ == '__main__':
    app.run("0.0.0.0")
