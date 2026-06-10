from flask import Flask, request, jsonify
import numpy as np
from preprocess import preprocess_image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins for production

model = None
model_type = None

# Attempt to load TensorFlow/Keras model first
try:
    from tensorflow.keras.models import load_model
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
        model_type = "keras"
        print("Loaded Keras CNN model from model.h5")
except ImportError:
    pass

# Fallback to scikit-learn joblib model
if model is None:
    try:
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        print(f"Looking for model at: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        print(f"Files in dir: {os.listdir(os.path.dirname(__file__))}")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_type = "sklearn"
            print("Loaded scikit-learn MLP model from model.joblib")
        else:
            print("Warning: No model found!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route("/")
def index():
    return jsonify({"status": "Digit Recognizer API is running!", "model": model_type})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "model_type": model_type})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "No model loaded."}), 500

    data = request.json["image"]
    img = preprocess_image(data)

    if model_type == "keras":
        preds = model.predict(img)[0]
    else:
        flat_img = img.reshape(1, 28 * 28)
        preds = model.predict_proba(flat_img)[0]

    probabilities = [float(p) for p in preds]
    top = preds.argsort()[-2:][::-1]

    return jsonify({
        "first_guess": int(top[0]),
        "second_guess": int(top[1]),
        "probabilities": probabilities
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
