"""Model training and persistence utilities."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data_preprocessing import select_features


TARGET_COLUMN = "MedHouseVal"


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train a Linear Regression model using the standard train/test split."""
    features, target = select_features(df, target_column=TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nTraining completed using train_test_split(test_size=0.2, random_state=42)")
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    return model, X_train, X_test, y_train, y_test


def save_model(model: LinearRegression, output_path: str = "models/house_price_model.pkl") -> None:
    """Save the trained model using joblib."""
    model_path = Path(output_path)
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
