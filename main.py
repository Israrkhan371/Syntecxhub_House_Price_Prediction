"""Main entry point for the House Price Prediction project."""

from src.data_preprocessing import clean_dataset, ensure_dataset, explore_dataset, select_features
from src.evaluate_model import calculate_metrics, print_metrics, show_coefficient_table, show_sample_predictions
from src.train_model import save_model, train_and_evaluate


def main() -> None:
    """Run the complete machine learning pipeline."""
    print("=" * 70)
    print("House Price Prediction Project")
    print("=" * 70)

    # 1) Load or download the dataset
    data_path = "data/housing.csv"
    df = ensure_dataset(data_path)

    # 2) Explore the dataset
    print("\n[1] Dataset Preview")
    print(df.head())
    explore_dataset(df)

    # 3) Prepare and clean features
    cleaned_df = clean_dataset(df)
    features_df, target_series = select_features(cleaned_df)

    print("\n[2] Cleaned data shape:", cleaned_df.shape)
    print("Selected features:", list(features_df.columns))

    # 4) Train/test split and model training
    model, X_train, X_test, y_train, y_test = train_and_evaluate(cleaned_df)

    # 5) Evaluate the model
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    print("\n[3] Model Evaluation")
    print_metrics(metrics)

    # 6) Coefficient interpretation
    print("\n[4] Feature Coefficients (sorted by absolute impact)")
    coefficient_table = show_coefficient_table(model, features_df.columns)
    print(coefficient_table.to_string(index=False))

    # 7) Save the trained model
    save_model(model, "models/house_price_model.pkl")
    print("\n[5] Model saved to models/house_price_model.pkl")

    # 8) Example predictions
    print("\n[6] Example Predictions")
    sample_predictions = show_sample_predictions(X_test, y_test, model, n=5)
    print(sample_predictions.to_string(index=False))

    print("\nPipeline completed successfully.")
    print("Run 'python main.py' again to retrain the model.")


if __name__ == "__main__":
    main()
