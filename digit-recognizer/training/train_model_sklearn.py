import urllib.request
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import joblib

def main():
    print("Checking for MNIST dataset...")
    data_path = "training/mnist.npz"
    if not os.path.exists(data_path):
        os.makedirs("training", exist_ok=True)
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST from {url}...")
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")
        
    print("Loading data...")
    with np.load(data_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        
    # Flatten and normalize the images
    x_train_flat = x_train.reshape(-1, 28 * 28) / 255.0
    x_test_flat = x_test.reshape(-1, 28 * 28) / 255.0
    
    # Train on 50,000 samples for better generalization
    subset_size = 50000
    print(f"Training high-performance MLP Classifier on {subset_size} samples...")
    
    # More complex architecture: 3 hidden layers (150, 100, 50)
    # Enable early stopping to prevent overfitting and speed up training
    mlp = MLPClassifier(
        hidden_layer_sizes=(150, 100, 50),
        max_iter=100,
        alpha=1e-4,
        solver='adam',
        verbose=True,
        random_state=42,
        learning_rate_init=.001,
        batch_size=256,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(x_train_flat[:subset_size], y_train[:subset_size])
    
    print("Evaluating model...")
    train_acc = mlp.score(x_train_flat[:subset_size], y_train[:subset_size])
    test_acc = mlp.score(x_test_flat, y_test)
    print(f"Training accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    
    print("Saving model to backend/model.joblib...")
    os.makedirs("backend", exist_ok=True)
    joblib.dump(mlp, "backend/model.joblib")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
