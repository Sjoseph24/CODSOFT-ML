import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the data dynamically
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'spam.csv') # Ensure your downloaded file is named spam.csv

try:
    print("⏳ Loading SMS dataset...")
    # Spam datasets often use latin-1 encoding instead of utf-8
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Standardize column names if it's the standard Kaggle dataset (v1=label, v2=text)
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        # Drop extra empty columns that usually come with this dataset
        df = df[['label', 'message']]

    print("--- Data Preview ---")
    print(df.head(), "\n")

    # Drop any missing values
    df.dropna(inplace=True)

    # 2. Split Data
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # 3. Text Feature Extraction using TF-IDF
    print("⏳ Converting text into numerical features...")
    tfidf = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4. Train Support Vector Machine (SVM) Classifier
    print("🧠 Training the SVM model (this is great for text!)...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    # 5. Predict and Evaluate
    print("📊 Evaluating the model...")
    predictions = svm_model.predict(X_test_tfidf)

    print("\n🎉 --- Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, predictions))

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at {file_path}. Make sure the name matches perfectly!")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")