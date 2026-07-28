# CODSOFT-ML
# Credit Card Fraud Detection Model

This repository contains a Python machine learning script designed to detect fraudulent credit card transactions. It uses a Random Forest Classifier to analyze numerical transaction data and identify potential fraud.

## Features
* **Automated Data Loading:** Dynamically loads the `fraudTrain.csv` dataset from the script's directory.
* **Data Preprocessing:** Automatically handles missing values and isolates numeric features, ensuring the model only trains on compatible data types. It also removes unnecessary index columns like `Unnamed: 0`[cite: 6].
* **Stratified Splitting:** Divides the dataset into 70% training and 30% testing sets, preserving the ratio of fraudulent to legitimate transactions using stratification (`stratify=y`)[cite: 6].
* **Random Forest Model:** Trains a robust `RandomForestClassifier` utilizing 100 estimators and multi-core processing (`n_jobs=-1`) for efficiency[cite: 6].
* **Comprehensive Evaluation:** Outputs a Confusion Matrix and a Classification Report to accurately assess model performance[cite: 6].

## Prerequisites & Installation

This script requires Python and the following libraries:
* pandas
* scikit-learn

You can install the required dependencies using your terminal:
```bash
pip install pandas scikit-learn
