import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "wine_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_data(features: list[list[float]]) -> np.ndarray:
    """
    Load the trained model and predict for given features.
    Args:
        features (list of list of float]): 2D array-like input for prediction.
    Returns:
        numpy.ndarray: Predicted class labels.
    """
    return model.predict(features)
