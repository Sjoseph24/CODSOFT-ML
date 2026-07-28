# CODSOFT-ML
# SMS Spam Detection Model

This repository contains a Python machine learning script designed to classify SMS messages as either spam or legitimate (ham). The model utilizes a Support Vector Machine (SVM) classifier alongside TF-IDF vectorization to accurately analyze and categorize text data.

## ⚠️ Important Dataset Information

**The dataset is NOT included in this repository.**
Text-based machine learning datasets are best kept out of version control to maintain a lightweight repository. You must supply the dataset locally to run the code.

**To run this project locally:**
1. Obtain or download the standard SMS spam collection dataset (often found on Kaggle) and save it as `spam.csv`.
2. Place the `spam.csv` file directly into the exact same folder as the Python script[cite: 8].
3. Ensure you have a `.gitignore` file in your repository that includes `spam.csv` so Git does not attempt to upload it.

## Key Features
* **Automated Data Loading & Cleaning:** Dynamically loads the `spam.csv` file using `latin-1` encoding to prevent standard read errors[cite: 8]. It automatically renames default columns (`v1` to `label` and `v2` to `message`) and drops unnecessary empty columns and missing values[cite: 8].
* **Text Feature Extraction:** Converts the raw text messages into numerical features using `TfidfVectorizer` while filtering out standard English stop words to improve model focus[cite: 8].
* **Data Splitting:** Divides the dataset into an 80% training and 20% testing split with a fixed random state (`random_state=42`) for reproducible results[cite: 8].
* **SVM Classification:** Trains a Support Vector Machine (`SVC`) model using a linear kernel, which is highly effective and optimized for text classification tasks.
* **Comprehensive Evaluation:** Prints the overall model accuracy percentage alongside a complete Classification Report to evaluate precision, recall, and f1-scores.

## Prerequisites & Installation

This script requires Python and the following libraries to execute successfully:
* `pandas`
* `scikit-learn`

You can install the required dependencies using your terminal or command prompt:
```bash
pip install pandas scikit-learn
