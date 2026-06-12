"""Data loading, cleaning, and exploration utilities."""

import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


TARGET_COLUMN = "MedHouseVal"


def ensure_dataset(output_path: str = "data/housing.csv") -> pd.DataFrame:
    """Load the dataset from disk or download it from sklearn if missing."""
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)

    if output_file.exists():
        print(f"Loading existing dataset from {output_file}")
        return pd.read_csv(output_file)

    print("Dataset not found. Downloading the California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print key dataset exploration summaries."""
    print("\nDataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe().round(4))
    print("\nCorrelation matrix:")
    print(df.corr(numeric_only=True).round(4))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and fill missing numeric values."""
    cleaned_df = df.copy()

    # Handle missing values by filling numeric columns with median
    numeric_columns = cleaned_df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if cleaned_df[column].isnull().any():
            median_value = cleaned_df[column].median()
            cleaned_df[column] = cleaned_df[column].fillna(median_value)

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    return cleaned_df


def select_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """Select all relevant numerical features and the target column."""
    numeric_df = df.select_dtypes(include="number")
    features = numeric_df.drop(columns=[target_column], errors="ignore")
    target = numeric_df[target_column]
    return features, target
