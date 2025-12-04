import numpy as np
from tensorflow.keras.models import load_model
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "autoencoder_model.keras")  

# Load the model
model = load_model(MODEL_PATH)

def predict_anomaly(image, threshold=0.01):
    """Run reconstruction and compute anomaly flag."""
    image_batch = np.expand_dims(image, axis=0)
    
    reconstructions = model.predict(image_batch)
    mse = np.mean((image - reconstructions[0]) ** 2)
    
    return {
        "reconstruction_mse": float(mse),
        "is_anomaly": bool(mse > threshold)
    }


