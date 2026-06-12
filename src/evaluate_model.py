"""Evaluation and interpretation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
    """Compute core regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
    }


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a user-friendly format."""
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"R²   : {metrics['R2 Score']:.4f}")


def show_coefficient_table(model, feature_names) -> pd.DataFrame:
    """Create a sorted coefficient table for interpretation."""
    coefficients = pd.Series(model.coef_, index=feature_names)
    coefficient_table = pd.DataFrame(
        {
            "Feature Name": coefficients.index,
            "Coefficient Value": coefficients.values,
            "Absolute Impact": coefficients.abs().values,
        }
    ).sort_values("Absolute Impact", ascending=False)

    return coefficient_table[["Feature Name", "Coefficient Value"]].reset_index(drop=True)


def show_sample_predictions(X_test, y_test, model, n: int = 5) -> pd.DataFrame:
    """Generate a small sample of actual vs predicted prices."""
    predictions = model.predict(X_test)
    sample = pd.DataFrame(
        {
            "Actual Price": y_test.iloc[:n].values,
            "Predicted Price": predictions[:n],
        }
    )
    return sample
