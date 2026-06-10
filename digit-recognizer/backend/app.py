from flask import Flask, request, jsonify
import numpy as np
from preprocess import preprocess_image
from flask_cors import CORS
import os

# Serve static files from the frontend folder
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

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
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_type = "sklearn"
            print("Loaded scikit-learn MLP model from model.joblib")
        else:
            print("Warning: No model found! Please run training first.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Serve index.html at root
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "No model loaded. Please run training first."}), 500

    data = request.json["image"]
    img = preprocess_image(data)  # returns shape (1, 28, 28, 1)

    if model_type == "keras":
        preds = model.predict(img)[0]
    else:
        # sklearn MLP expects flattened input (1, 784)
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
    app.run(debug=True)
