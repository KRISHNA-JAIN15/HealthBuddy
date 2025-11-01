import os
import sys
import time
import numpy as np
import pandas as pd
import pickle  # Import pickle for saving the model
import warnings
from collections import Counter

# --- 1. Standard Library Imports (for Preprocessing) ---
print("Loading standard utility libraries (sklearn, pandas, numpy)...")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 2. Configuration (NEEDS YOUR INPUT) ---

# ❗ **ACTION REQUIRED**: Make sure this path points to your Maternal Health CSV file.
DATA_PATH = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\MentalHealth\Maternal Health Risk Data Set.csv'

# ❗ **ACTION REQUIRED**: Make sure this path points to your 'Classifier_codes' folder.
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

# ❗ **ACTION REQUIRED**: Define where to save the final model.
MODEL_SAVE_PATH = r'maternal_health_randomforest.pkl'


# --- 3. Import Your Custom Random Forest Classifier ---
print(f"Importing your custom classifier from '{CLASSIFIER_DIR}'...")
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # Import the classes needed for RandomForest.
    # (Your RandomForest.py file already contains DecisionTree and Node)
    from RandomForest import RandomForestClassifier, DecisionTreeClassifier, Node
    print("✅ Successfully imported RandomForestClassifier and dependencies.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---")
    print(f"Error importing 'RandomForestClassifier' from 'RandomForest.py': {e}")
    sys.exit(1)


# --- 4. Main Execution ---
if __name__ == "__main__":
    
    # --- Check Paths ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    print(f"\n{'='*25} Starting Training Pipeline {'='*25}")

    # --- 1. Load Data ---
    print(f"Loading data from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded. Shape: {df.shape}")

    # --- 2. Preprocessing (on FULL dataset) ---
    print("Starting preprocessing...")
    
    df_processed = df.copy()

    # Define X (features) and y (target)
    X = df_processed.drop('RiskLevel', axis=1)
    y = df_processed['RiskLevel']

    # Define Feature Types
    ALL_FEATURES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
    
    print(f"Total features defined: {len(ALL_FEATURES)}")

    # Create Preprocessing Pipeline
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Encode the target variable
    print("Encoding target variable (y)...")
    le = LabelEncoder()
    y_processed = le.fit_transform(y)
    
    print("\nPreprocessing setup complete.")
    
    # --- 3. Train Random Forest Model ---
    print("\n--- Training Random Forest Model ---")
    
    # --- THIS IS THE FIX ---
    # Initialize your from-scratch RandomForest model
    # using only the arguments it accepts in __init__
    model_rf = RandomForestClassifier(
        n_trees=100,         # Using 100 trees for good performance
        max_depth=10, 
        min_samples_split=5
    )
    
    # --- 4. Create and Save the *Full Pipeline* ---
    # This pipeline bundles the preprocessor and the model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_rf) # Use the model instance
    ])
    
    # Fit the full pipeline on the *raw* X and encoded y
    print("Fitting full pipeline (preprocessor + model) for saving...")
            
    start_time = time.time()
    # Pass the raw feature DataFrame (X) and the encoded target (y_processed)
    full_pipeline.fit(X, y_processed)
    duration = time.time() - start_time
    print(f"✅ Full pipeline training complete. (Time: {duration:.2f}s)")
    
    # Save the bundled pipeline
    print(f"\n--- Saving Model ---")
    try:
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(full_pipeline, f)
            
        print(f"✅ Full pipeline (preprocessor + model) successfully saved to: {MODEL_SAVE_PATH}")
        print("   You can now load this .pkl file to make predictions on new, raw data.")

    except Exception as e:
        import traceback
        print(f"❌ Error saving model: {e}")
        traceback.print_exc()

    print(f"\n{'='*25} Pipeline Finished {'='*25}")