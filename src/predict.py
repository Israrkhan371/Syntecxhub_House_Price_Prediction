"""Prediction utilities for the saved model."""

import joblib
import pandas as pd


MODEL_PATH = "models/house_price_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """Load the trained model from disk."""
    return joblib.load(model_path)


def predict_prices(model, features: pd.DataFrame) -> list:
    """Return predictions for provided feature rows."""
    return model.predict(features)
