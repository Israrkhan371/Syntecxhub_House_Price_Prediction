# House Price Prediction

## Project Overview
This project builds a simple machine learning pipeline to predict California housing prices using a Linear Regression model. It includes data loading, cleaning, exploration, model training, evaluation, coefficient interpretation, model saving, and example predictions.

## Dataset Information
- Source: California Housing dataset from scikit-learn
- File: data/housing.csv
- Target column: MedHouseVal

## Technologies Used
- Python
- pandas
- scikit-learn
- joblib
- NumPy

## Installation Steps
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
From the project root:
```bash
python main.py
```

## Sample Output
The script prints:
- first 5 rows of the dataset
- shape, dtypes, missing values, summary statistics, and correlation matrix
- RMSE, MAE, and R²
- coefficient table sorted by impact
- example actual vs predicted house prices

## Evaluation Metrics
The trained Linear Regression model reports:
- RMSE
- MAE
- R² Score

## Project Structure
```text
Syntecxhub_House_Price_Prediction/
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── main.py
```
