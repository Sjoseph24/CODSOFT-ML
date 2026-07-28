# CODSOFT-ML
# Credit Card Fraud Detection Model

This repository contains a Python machine learning script designed to detect fraudulent credit card transactions. It uses a Random Forest Classifier to analyze numerical transaction data and identify potential fraud.

## ⚠️ Important Dataset Information

**The dataset is NOT included in this repository.** 
Because machine learning datasets are typically very large, it is standard practice to exclude them from version control to keep the repository lightweight and fast. 

**To run this project locally:**
1. Download the `fraudTrain.csv` dataset from its original source (e.g., [Kaggle](https://www.kaggle.com/)).
2. Place the downloaded `fraudTrain.csv` file directly into the exact same folder as the Python script.
3. Ensure you have a `.gitignore` file in your repository that includes `fraudTrain.csv` so Git does not attempt to upload it.

## Key Features
* **Automated Data Loading:** Dynamically loads the `fraudTrain.csv` dataset from the script's directory.
* **Data Preprocessing:** Automatically handles missing values and isolates numeric features, ensuring the model only trains on compatible data types[cite: 6]. It also removes unnecessary index columns like `Unnamed: 0`[cite: 6].
* **Stratified Splitting:** Divides the dataset into 70% training and 30% testing sets, preserving the ratio of fraudulent to legitimate transactions using stratification (`stratify=y`)[cite: 6].
* **Random Forest Model:** Trains a robust `RandomForestClassifier` utilizing 100 estimators and multi-core processing (`n_jobs=-1`) for efficiency[cite: 6].
* **Comprehensive Evaluation:** Outputs a Confusion Matrix and a Classification Report to accurately assess model performance[cite: 6].

## Prerequisites & Installation

This script requires Python and the following libraries to execute successfully:
* pandas
* scikit-learn

You can install the required dependencies using your terminal or command prompt:
```bash
pip install pandas scikit-learn
