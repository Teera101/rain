import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model = joblib.load("best_model.pkl")
model = final_model["model"]
label_encoder = final_model["label_encoder"]  

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask Weather Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")
        required_fields = ["precipitation", "temp_max", "temp_min", "wind"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = [
            float(data["precipitation"]),
            float(data["temp_max"]),
            float(data["temp_min"]),
            float(data["wind"])
        ]
        print(f"Features: {features}")

        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)

        weather_predicted = label_encoder.inverse_transform([prediction[0]])[0]
        print(f"Prediction: {weather_predicted}")

        return jsonify({"prediction": weather_predicted})

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
