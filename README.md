# CODSOFT-ML
# Customer Churn Prediction Model

This repository contains a Python machine learning script designed to predict customer churn using a Gradient Boosting Classifier. The model processes customer demographic and behavioral attributes to classify whether a customer is likely to exit the service (`Exited`).

## ⚠️ Important Dataset Information

**The dataset is NOT included in this repository.**
Because machine learning datasets are best kept out of version control to maintain a lightweight repository, you must supply the dataset locally.

**To run this project locally:**
1. Obtain or download the `churn_modelling.csv` dataset.
2. Place the `churn_modelling.csv` file directly into the exact same folder as the Python script[cite: 7].
3. Ensure you have a `.gitignore` file in your repository that includes `churn_modelling.csv` so Git does not attempt to upload it.

## Key Features
* **Automated Data Loading & Cleaning:** Dynamically loads `churn_modelling.csv`, removes missing values, and drops non-predictive identifier columns like `RowNumber`, `CustomerId`, `Surname`, and `Unnamed: 0`[cite: 7].
* **Categorical Encoding:** Converts categorical variables (such as `Geography` and `Gender`) into numeric dummy variables using one-hot encoding (`pd.get_dummies`)[cite: 7].
* **Feature Scaling:** Uses `StandardScaler` to normalize numeric features so larger numerical values (such as salary) do not overpower smaller scales[cite: 7].
* **Stratified Data Splitting:** Divides data into an 80% training and 20% testing split while preserving the target distribution using stratification (`stratify=y`)[cite: 7].
* **Gradient Boosting Model:** Trains a robust `GradientBoostingClassifier` using 100 estimators for high classification accuracy[cite: 7].
* **Comprehensive Evaluation:** Prints the overall model accuracy percentage alongside a complete Classification Report (Precision, Recall, F1-Score)[cite: 7].

## Prerequisites & Installation

This script requires Python and the following libraries to execute successfully:
* `pandas`
* `scikit-learn`

You can install the required dependencies using your terminal or command prompt:
```bash
pip install pandas scikit-learn

