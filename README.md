# CODSOFT-ML
# Movie Genre Classification Model

This repository contains a Python-based machine learning pipeline designed to predict movie genres based on their plot descriptions. Using movie database information, the script processes training data and applies natural language processing techniques to categorize films automatically.

## Key Features
* **Data Processing:** The script loads movie data formatted with ID, Title, Genre, and Description fields. It automatically cleans the dataset by dropping missing or blank rows that could cause the code to crash.
* **Text Feature Extraction:** It utilizes `TfidfVectorizer` to convert text-based movie descriptions into numerical features. It filters out standard English stop words to improve the text extraction process[cite: 2].
* **Model Training:** The project implements a `MultinomialNB` (Naive Bayes) classifier from the `scikit-learn` library[cite: 2]. This model is trained to learn the relationship between the extracted TF-IDF features and the target movie genres[cite: 2].
* **Evaluation & Metrics:** The script automatically splits the data into training and testing sets using an 80/20 split[cite: 2]. It predicts the test set outcomes and outputs the model's overall accuracy score alongside a comprehensive classification report[cite: 2].

## Prerequisites & Installation
This project requires Python and the following libraries to execute successfully:
* `pandas`[cite: 2]
* `scikit-learn`[cite: 2]

You can install the necessary dependencies using your terminal or command prompt:
```bash
pip install pandas scikit-learn
