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
    if os.path.exists("backend/model.h5"):
        model = load_model("backend/model.h5")
        model_type = "keras"
        print("Loaded Keras CNN model from model.h5")
except ImportError:
    pass

# Fallback to scikit-learn joblib model
if model is None:
    try:
        import joblib
        if os.path.exists("backend/model.joblib"):
            model = joblib.load("backend/model.joblib")
            model_type = "sklearn"
            print("Loaded scikit-learn MLP model from model.joblib (fallback)")
        else:
            print("Warning: No model found! Please train a model first.")
    except Exception as e:
        print(f"Error loading fallback model: {e}")

# Serve index.html at root url
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "No model loaded. Please run training first."}), 500
        
    data = request.json["image"]
    img = preprocess_image(data) # returns shape (1, 28, 28, 1)
    
    if model_type == "keras":
        preds = model.predict(img)[0]
    else:
        # sklearn MLP model expects flattened input (1, 784)
        flat_img = img.reshape(1, 28 * 28)
        # predict_proba returns class probabilities for each digit
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
