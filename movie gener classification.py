import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the data
# This automatically gets the folder where your python script is saved
current_directory = os.path.dirname(os.path.abspath(__file__))
# This joins that folder path with your file name
file_path = os.path.join(current_directory, 'train_data.txt')

try:
    df = pd.read_csv(file_path, sep=' ::: ', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
    
    # FIX: Drop any empty rows or blank lines that cause the code to crash
    df.dropna(subset=['DESCRIPTION', 'GENRE'], inplace=True)

    print("--- Data Preview ---")
    print(df.head(), "\n")

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(df['DESCRIPTION'], df['GENRE'], test_size=0.2, random_state=42)

    # 3. Text Feature Extraction using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4. Train Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # 5. Predict and Evaluate
    predictions = model.predict(X_test_tfidf)

    print("--- Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at {file_path}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")