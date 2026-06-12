# 🏠 House Price Prediction

A Machine Learning project that predicts house prices using a **Linear Regression** model trained on the California Housing Dataset. This project demonstrates the complete Machine Learning workflow, including data preprocessing, model training, evaluation, interpretation, and model persistence.

---

## 📌 Project Objective

The objective of this project is to build a predictive model that estimates housing prices based on demographic and geographic features. The project provides hands-on experience with data analysis, feature engineering, regression modeling, and performance evaluation.

---

## 🚀 Project Overview

This project covers the complete Machine Learning pipeline:

* Data collection and loading
* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Feature selection
* Train-test splitting
* Linear Regression model training
* Model evaluation
* Feature coefficient interpretation
* Model saving and reuse
* Example predictions

---

## 📊 Dataset Information

**Dataset:** California Housing Dataset

**Source:** Scikit-learn

**Target Variable:** `MedHouseVal` (Median House Value)

### Dataset Statistics

| Property | Value       |
| -------- | ----------- |
| Rows     | 20,640      |
| Features | 8           |
| Target   | MedHouseVal |

### Input Features

| Feature    | Description                    |
| ---------- | ------------------------------ |
| MedInc     | Median income in block group   |
| HouseAge   | Median house age               |
| AveRooms   | Average rooms per household    |
| AveBedrms  | Average bedrooms per household |
| Population | Population of block group      |
| AveOccup   | Average household occupancy    |
| Latitude   | Latitude coordinate            |
| Longitude  | Longitude coordinate           |

---

## 🛠 Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## 📂 Project Structure

```text
Syntecxhub_House_Price_Prediction/
│
├── data/
│   └── housing.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔄 Machine Learning Workflow

### 1. Data Loading

* Load housing dataset
* Display dataset information
* Explore features and target variable

### 2. Data Cleaning

* Handle missing values
* Remove duplicate records
* Prepare clean dataset for training

### 3. Feature Selection

* Select relevant numerical features
* Separate features and target variable

### 4. Train-Test Split

* 80% Training Data
* 20% Testing Data
* Random State = 42

### 5. Model Training

* Linear Regression Algorithm
* Supervised Learning Approach

### 6. Model Evaluation

The model is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

### 7. Model Persistence

* Save trained model using Joblib
* Reuse model without retraining

### 8. Prediction

* Generate predictions on unseen data
* Compare actual vs predicted values

---

## 📈 Model Performance

The model reports the following evaluation metrics:

| Metric   | Description                  |
| -------- | ---------------------------- |
| RMSE     | Root Mean Squared Error      |
| MAE      | Mean Absolute Error          |
| R² Score | Coefficient of Determination |

### Sample Results

```text
RMSE : 0.7456
MAE  : 0.5332
R²   : 0.5758
```

---

## 📊 Outputs Generated

The project displays:

* First 5 rows of dataset
* Dataset shape
* Data types
* Missing values summary
* Statistical summary
* Correlation matrix
* Feature coefficient table
* Model evaluation metrics
* Actual vs Predicted house prices

---

## 💾 Saved Model

The trained model is automatically saved to:

```text
models/house_price_model.pkl
```

This allows future predictions without retraining the model.

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone <repository-url>
cd Syntecxhub_House_Price_Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the project using:

```bash
python main.py
```

The application will:

1. Load or download the dataset
2. Explore and preprocess the data
3. Train the Linear Regression model
4. Evaluate performance
5. Save the trained model
6. Display sample predictions

---

## 🔮 Future Improvements

Possible enhancements include:

* Random Forest Regression
* XGBoost Regression
* Hyperparameter Tuning
* Feature Engineering
* Cross-Validation
* Streamlit Web Application
* Model Deployment to Cloud
* Interactive Dashboard

---

## 👨‍💻 Author

**Muhammad Israr Khan**

Machine Learning Intern @Syntecxhub

BS Computer Science – Bahria University Islamabad

LinkedIn:
https://www.linkedin.com/in/muhammad-israr-khan-06a06628b/

---

## 📜 License

This project was developed for educational and internship purposes as part of the Machine Learning Internship Program at Syntecxhub.
