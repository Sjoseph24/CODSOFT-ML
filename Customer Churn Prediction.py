import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the data dynamically
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'churn_modelling.csv') # Ensure this matches your file name

try:
    print("⏳ Loading dataset...")
    df = pd.read_csv(file_path)
    
    print("--- Data Preview ---")
    print(df.head(), "\n")

    # 2. Preprocess the Data
    df.dropna(inplace=True) 
    
    # FIX: Updated to match your dataset's column name
    target_col = 'Exited' 
    
    if target_col not in df.columns:
        print(f"❌ ERROR: '{target_col}' column not found. Please check your CSV.")
    else:
        # Separate Features and Target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Drop useless ID columns that don't predict churn
        cols_to_drop = [col for col in ['RowNumber', 'CustomerId', 'Surname', 'Unnamed: 0'] if col in X.columns]
        X = X.drop(columns=cols_to_drop)

        # Convert text columns (like Geography, Gender) into numbers automatically
        X = pd.get_dummies(X, drop_first=True)

        # 3. Split Data
        print("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale the numerical features so large numbers (like Salary) don't overpower small numbers (like Age)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Train Gradient Boosting Model
        print("🧠 Training the Gradient Boosting model...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)

        # 5. Predict and Evaluate
        print("📊 Evaluating the model...")
        y_pred = gb_model.predict(X_test_scaled)
        
        print("\n🎉 --- Model Results ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
        print("Classification Report:\n", classification_report(y_test, y_pred))

except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at {file_path}. Make sure the name matches perfectly!")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")