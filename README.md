# 🧠 AI Digit Recognizer

> A real-time handwritten digit recognition web app powered by a deep learning CNN model trained on the MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN%20Model-orange?style=for-the-badge&logo=tensorflow)
![HTML5](https://img.shields.io/badge/HTML5-Canvas-red?style=for-the-badge&logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-Glassmorphism-blue?style=for-the-badge&logo=css3)

---

## ✨ Features

- 🎨 **Draw on Canvas** — Freehand digit drawing with adjustable brush size
- ⚡ **Real-time Prediction** — Auto-predicts as you draw using a trained CNN
- 📊 **Probability Distribution** — Live bar chart showing confidence for all 10 digits (0–9)
- 🔄 **Undo / Redo** — Full undo/redo stack with `Ctrl+Z` / `Ctrl+Y`
- 🖼️ **Model Input Preview** — View the 28×28 downsampled image sent to the model
- 🌐 **Modern UI** — Glassmorphism + Skeuomorphism design with smooth animations
- 📱 **Touch Support** — Works on tablets and mobile devices

---

## 🖥️ Demo

| Draw a Digit | Get Instant Prediction |
|:---:|:---:|
| Draw any digit (0–9) on the canvas | The CNN model predicts in real time |

---

## 🏗️ Project Structure

```
digit-recognizer/
├── frontend/
│   ├── index.html       # Main UI
│   ├── style.css        # Glassmorphism + Skeuomorphism styles
│   └── script.js        # Canvas drawing & API integration
├── backend/
│   ├── app.py           # Flask REST API server
│   ├── preprocess.py    # Image preprocessing pipeline
│   ├── model.h5         # Trained Keras CNN model
│   └── model.joblib     # Trained Scikit-learn model
└── training/
    ├── train_model.py           # CNN training script (Keras)
    └── train_model_sklearn.py   # Sklearn training script
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/sainagb47/digit-recognizer.git
cd digit-recognizer
```

### 2. Install Dependencies

```bash
pip install flask flask-cors tensorflow scikit-learn numpy pillow
```

### 3. Start the Backend Server

```bash
cd digit-recognizer/backend
python app.py
```

> The Flask API will run at `http://127.0.0.1:5000`

### 4. Open the Frontend

```bash
cd digit-recognizer/frontend
python -m http.server 8000
```

Then open your browser and go to: **http://localhost:8000**

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Dataset | MNIST (70,000 handwritten digits) |
| Architecture | Convolutional Neural Network (CNN) |
| Input Size | 28 × 28 grayscale image |
| Output | 10 classes (digits 0–9) |
| Framework | TensorFlow / Keras |

---

## 🎨 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5 Canvas, Vanilla CSS (Glassmorphism), JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML Model | TensorFlow/Keras CNN |
| Preprocessing | NumPy, Pillow |

---

## 📡 API Reference

### `POST /predict`

Accepts a base64-encoded PNG image and returns digit predictions.

**Request Body:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "first_guess": 7,
  "second_guess": 1,
  "probabilities": [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91, 0.01, 0.01]
}
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


