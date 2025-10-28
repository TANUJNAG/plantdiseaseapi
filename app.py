from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
import os

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("model/FINAL_95.84acc_model.keras")

# Load class labels
with open("model/class_labels.json") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Load treatment recommendations
with open("model/treatment_recommendations.json") as f:
    treatment_recommendations = json.load(f)

# Image preprocessing
def preprocess_image(file, target_size=(224, 224)):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = preprocess_image(request.files["image"])
        pred = model.predict(img)
        predicted_class = int(np.argmax(pred[0]))
        predicted_label = class_labels[predicted_class]
        confidence = float(pred[0][predicted_class])
        treatment = treatment_recommendations.get(predicted_label, {
            "pesticide": "Not available",
            "fertilizer": "Not available"
        })

        return jsonify({
            "crop_type": predicted_label.split("___")[0].split("_")[0],
            "disease_detected": predicted_label,
            "confidence": round(confidence, 3),
            "treatment": f"Pesticide: {treatment['pesticide']}, Fertilizer: {treatment['fertilizer']}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run locally or on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)