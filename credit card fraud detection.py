import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the data dynamically from the script's folder
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'fraudTrain.csv')

try:
    print("⏳ Loading dataset (this might take a moment depending on the file size)...")
    df = pd.read_csv(file_path)
    
    print("--- Data Preview ---")
    print(df.head(), "\n")

    # Drop any rows with missing values to prevent errors
    df.dropna(inplace=True) 
    
    # 2. Separate Features and Target
    # Your dataset uses 'is_fraud' as the target column
    if 'is_fraud' not in df.columns:
        print("❌ ERROR: 'is_fraud' column not found.")
    else:
        # Machine learning models need numbers, not text or dates. 
        # We will grab only the numeric columns for our features.
        numeric_df = df.select_dtypes(include=['number'])
        
        # X will be our features (everything EXCEPT is_fraud and the index column)
        X = numeric_df.drop('is_fraud', axis=1)
        
        if 'Unnamed: 0' in X.columns:
            X = X.drop('Unnamed: 0', axis=1)
            
        # y will be our target
        y = numeric_df['is_fraud']

        # 3. Split Data
        print("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # 4. Train Random Forest Model
        print("🧠 Training the Random Forest model (this may take a few minutes)...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        # 5. Predict and Evaluate
        print("📊 Evaluating the model...")
        y_pred = rf_model.predict(X_test)
        
        print("\n🎉 --- Model Results ---")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at {file_path}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")