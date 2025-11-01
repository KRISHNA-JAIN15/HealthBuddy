import os
import sys
import time
import numpy as np
import pandas as pd
import pickle  # Import pickle for saving the model
import warnings
from collections import Counter # Needed by KNN

# --- 1. Standard Library Imports (for Preprocessing) ---
print("Loading standard utility libraries (sklearn, pandas, numpy)...")
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 2. Configuration (NEEDS YOUR INPUT) ---

# ❗ **ACTION REQUIRED**: Make sure this path points to your CKD CSV file.
DATA_PATH = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\ChronicKidneyDisease\chronic_kidney_disease_full.csv'

# ❗ **ACTION REQUIRED**: Make sure this path points to your 'Classifier_codes' folder.
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

# ❗ **ACTION REQUIRED**: Define where to save the final model.
MODEL_SAVE_PATH = r'knn_ckd_model.pkl'


# --- 3. Import Your Custom KNN Classifier ---
print(f"Importing your custom classifier from '{CLASSIFIER_DIR}'...")
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # Import only the KNN class
    from KNN import KNN
    print("✅ Successfully imported KNN classifier.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---")
    print(f"Error importing 'KNN' from 'KNN.py': {e}")
    sys.exit(1)


# --- 4. Main Execution ---
if __name__ == "__main__":
    
    # --- Check Paths ---
    if DATA_PATH == r'path/to/your/chronic_kidney_disease_full.csv':
        print("="*60)
        print("🛑 ERROR: Please update the 'DATA_PATH' variable (line 30)")
        print("       to point to your chronic_kidney_disease_full.csv file.")
        print("="*60)
        sys.exit(1)
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    print(f"\n{'='*25} Starting Training Pipeline {'='*25}")

    # --- 1. Load Data ---
    # Define Column Names (from metadata)
    column_names = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
        'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
        'cad', 'appet', 'pe', 'ane', 'class'
    ]
    
    print(f"Loading data from '{DATA_PATH}'...")
    df = pd.read_csv(
        DATA_PATH, 
        header=None, 
        names=column_names, 
        na_values='?'
    )
    print(f"Dataset loaded successfully. Shape: {df.shape}")


    # --- 2. Preprocessing (on FULL dataset) ---
    print("Starting preprocessing...")

    # Define feature types based on your metadata
    NUMERIC_FEATURES = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    NOMINAL_FEATURES = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    TARGET = 'class'

    # --- Clean Data ---
    # Convert numeric columns to numeric type
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- THIS IS THE FIX ---
    # Define *only* the columns that are ACTUALLY strings
    NOMINAL_FEATURES_STRING = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    string_cols = NOMINAL_FEATURES_STRING + [TARGET]
    # --- END OF FIX ---
    
    # Clean whitespace from nominal (string) columns
    for col in string_cols:
        # Check if column exists and is object type before stripping
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('ckd\t', 'ckd') # Fix known issue in target

    # Define X (features) and y (target)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # CRITICAL: Drop rows where the target 'class' is missing
    y_known_idx = y.dropna().index
    X = X.loc[y_known_idx]
    y = y.loc[y_known_idx]

    # --- Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- Create the Full ColumnTransformer ---
    # This setup is correct. 'sg', 'al', 'su' are in NOMINAL_FEATURES
    # and will be correctly treated as categories by the OneHotEncoder.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('nom', nominal_transformer, NOMINAL_FEATURES)
        ],
        remainder='drop'
    )

    # Encode the target variable (e.g., 'ckd' -> 0)
    print("Encoding target variable (y)...")
    le = LabelEncoder()
    y_processed = le.fit_transform(y)
    
    print("\nPreprocessing setup complete.")
    
    # --- 3. Train KNN Model ---
    print("\n--- Training KNN Model ---")
    
    # Initialize your from-scratch KNN model
    model_knn = KNN(
        k=5 # A good default for k
    )
    
    # --- 4. Create and Save the *Full Pipeline* ---
    # This pipeline bundles the preprocessor and the model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_knn) # Use the model instance
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