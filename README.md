# 🏎️ HyperDrive ML: Used Car Price Prediction

## Table of Contents

* [Project Overview](#-project-overview)
* [Project Structure](#-project-structure)
* [Dataset](#-dataset)
* [Workflow](#-workflow)
* [Technologies and Libraries](#-technologies-and-libraries)
* [Key Results](#-key-results)
* [Usage Example](#-usage-example)
* [Contributions](#-contribution)
* [Acknowledgements](#-acknowledgements)

## 📋 Project Overview

A comprehensive machine learning project for predicting used car prices using various regression models with hyperparameter tuning. This project implements a complete machine learning pipeline for predicting used car prices, including data preprocessing, exploratory data analysis, and hyperparameter optimization using Optuna.

## 📂 Project Structure

```
project_folder/
├── used_cars.csv                          # Raw dataset from Kaggle
├── used_cars_preprocessed.csv             # Cleaned and processed data
├── data_preprocessing.ipynb                # Data cleaning and feature engineering
├── data_analysis.ipynb                     # Exploratory data analysis (EDA)
├── model_hyperparameter_tuning.ipynb      # Model training and optimization (Google Colab)
├── prediction_new_data.ipynb              # Predictions on new data
├── custom_transformer.py                  # Custom Scikit-Learn transformer class
├── final_pipeline.joblib                  # Serialized trained model pipeline
├── best_params.json                        # Optimal hyperparameters from Optuna
└── README.md
```

## 📊 Dataset

**Raw dataset**

**Source:** [Used Cars Dataset - Kaggle](https://www.kaggle.com/datasets/leiwangetc666/used-cars)

**Size:** 4009 rows 12 features

**Features:**
Brand, Model, Model Year, Mileage, Fuel Type, Engine, Transmission, Exterior Color, Interior Color, Accident, Clean Title, Price.

**Preprocessed dataset**

**Size:** 3998 rows 11 features

**Features:**
Brand, Model, Model Year, Engine Volume, Engine Type, Transmission, Fuel Type, Exterior Color, Mileage(Miles), Damage, Price(USD)

## 🔄 Workflow

### 1. Data Preprocessing ([`data_preprocessing.ipynb`](data_preprocessing.ipynb))
- Handles data cleaning
- Feature Engineering new columns
- Outlier detection and treatment
- Output: [`used_cars_preprocessed.csv`](used_cars_preprocessed.csv)

### 2. Exploratory Data Analysis ([`data_analysis.ipynb`](data_analysis.ipynb))
- Univariate and multivariate analysis
- Distribution analysis of numerical data
- Correlation analysis with target variable
- Statistical validation, including Variance Inflation Factor (VIF) analysis to detect and mitigate multicollinearity.

### 3. Custom Scikit-Learn Transformer ([`custom_transformer.py`](custom_transformer.py))
- A dedicated module featuring the OrderedTargetEncoder. This custom transformer manages high-cardinality categorical features with their smoothed target means, and then maps these smoothed means to an ordered integer sequence.

### 4. Model Development & Hyperparameter Tuning ([`model_hyperparameter_tuning.ipynb`](model_hyperparameter_tuning.ipynb))
This notebook performs extensive hyperparameter optimization using Optuna.
**⚠️ Note:** This notebook was executed in **Google Colab** due to intensive computational requirements for hyperparameter optimization.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IVID0iNZe3tGoVCtIyQKlGQlA03FAGTF?usp=sharing)

**Models Implemented:**
- Linear and Regualarized Linear Models
- Ridge & Lasso Regression
- K-Nearest Neighbors (KNN), Support Vector Regression
- Regression Trees (Decision Tree, Random Forest)
- Boosting Regressor
- XGBoost
- Ensemble Methods

**Optimization Framework:** Optuna with Bayesian optimization (TPE Sampler) with implementation of pruning

### 5. Production Model and Hyperparameter Results

- [`best_param.json`](best_param.json): Stores the best hyperparameter settings found for each model, ensuring reproducibility and easy reference for tuning 
- [`final_pipeline.joblib`](final_pipeline.joblib): The serialized, complete machine learning pipeline featuring all transformations and the final, best-performing SVR model, optimized for deployment.

### 6. Prediction on New Data ([`prediction_new_data.ipynb`](prediction_new_data.ipynb))
Make predictions on new vehicle data using the trained and optimized pipeline stored in joblib file.

## 📦 Technologies and Libraries

| Category | Tools |
| :--- | :--- |
| **Data Manipulation** | ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-eb8a2f?style=for-the-badge&logo=xgboost&logoColor=white) |
| **Statistical Analysis** | ![Statsmodels](https://img.shields.io/badge/statsmodels-%232d5e80.svg?style=for-the-badge&logo=statsmodels&logoColor=white) |
| **Optimization & Serialization** | ![Optuna](https://img.shields.io/badge/Optuna-blue?style=for-the-badge) ![Joblib](https://img.shields.io/badge/Joblib-orange?style=for-the-badge) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Seaborn](https://img.shields.io/badge/Seaborn-blue?style=for-the-badge) |

### Key Library Functions:
- **Pandas & NumPy**: Essential for data cleaning, transformation, and handling numerical arrays.
- **Scikit-learn**: Used for building the model pipeline, cross-validation, and implementing diverse algorithms.
- **Statsmodels**: Specifically utilized for advanced statistical testing, including **Variance Inflation Factor (VIF)** analysis to ensure feature independence.
- **Optuna**: Powers the automated hyperparameter tuning process for maximum model and time efficiency.
- **XGBoost**: High-performance gradient boosting with integration of GPU.
- **Matplotlib & Seaborn**: Used to generate exploratory data analysis (EDA) plots and feature correlation heatmaps.
- **Joblib**: Used for saving and loading the finalized `final_pipeline.joblib` for production-ready deployment.

### Running in Google Colab

Since the original notebook uses Google Drive, follow these steps to use `model_hyperparameter_tuning.ipynb` in Google Colab:

1. **Upload files to Google Drive:**
   ```
   /MyDrive/folder_path/
   ├── model_hyperparameter_tuning.ipynb
   └── used_cars_preprocessed.csv
   ```

2. **In the notebook, mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Set working directory:**
   ```python
   import os
   os.chdir('/content/drive/MyDrive/folder_path')
   ```

**Performance Metrics used:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## 📈 Key Results

### Best Hyperparameters (SVR Model)
```json
{
  "smoothing": 1.9872974219495898,
  "kernel": "linear",
  "degree": 1,
  "gamma": 0.00274552559906995,
  "tol": 0.0003713971940428087,
  "C": 6.553865827463084,
  "epsilon": 0.05900718360099283
}
```

### Model Performance
- **Test R² Score:** 0.876
- **Mean Absolute Error:** 0.202* 
- **Mean Squared Error:** 0.081*
*Calculated on log-transformed scale

## 💻 Usage Example

```python
# Import necessary libraries
import joblib
import numpy as np, pandas as pd
from custom_transformer import OrderedTargetEncoder

# Load trained model
pipeline = joblib.load('final_pipeline.joblib')

# Create a new DataFrame for prediction
new_car = pd.DataFrame({
    'brand': ['BMW'],
    'model': ['M3'],
    'model_year': [2023],
    'engine_type': ['I6'],
    'fuel_type': ['Gasoline'],
    'ext_col': ['black'],
    'mileage in miles': [5000],
    'damage': ['None reported']
})

# Make predictions
predicted_price = np.exp(pipeline.predict(new_car))
print(f"Predicted Price: ${predicted_price[0].astype(int)}")
```

## 🤝 Contribution

Contributions are welcome to enhance model accuracy or add new features.

- Fork the Project.
- Create your Feature Branch (`git checkout -b feature-name`).
- Commit your Changes (`git commit -m 'Add new feature'`).
- Push to the Branch (`git push origin feature-name`).
- Open a Pull Request.

## 🙏 Acknowledgements

This project was developed using various open source libraries and frameworks: Jupyter Notebook, Python, VS Code, Google Colab to provide a cloud-based, interactive computing environment for seamless code execution and GPU support, leveraging NumPy for high-performance numerical operations and Pandas for efficient data manipulation and cleaning. Data visualization was performed using Matplotlib and Seaborn to derive statistical insights, while Scikit-Learn provided the essential pipeline architecture for preprocessing and model evaluation. The predictive power of the project relies on XGBoost for gradient boosting, with Optuna utilized for automated hyperparameter tuning through its Bayesian optimization framework. Additionally, joblib was used for model serialization and pipelining efficiency, and dataset provided by Kaggle.
