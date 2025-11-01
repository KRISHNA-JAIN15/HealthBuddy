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
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 2. Configuration (NEEDS YOUR INPUT) ---

# ❗ **ACTION REQUIRED**: Make sure this path points to your Obesity CSV file.
DATA_PATH = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\ObesityLevel\ObesityDataSet_raw_and_data_sinthetic.csv'

# ❗ **ACTION REQUIRED**: Make sure this path points to your 'Classifier_codes' folder.
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

# ❗ **ACTION REQUIRED**: Define where to save the final model.
MODEL_SAVE_PATH = r'decision_tree_model.pkl'


# --- 3. Import Your Custom Decision Tree Classifier ---
print(f"Importing your custom classifier from '{CLASSIFIER_DIR}'...")
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # Import only the DecisionTree class
    from DecisionTree import DecisionTreeClassifier # Assumes Node is in this file
    print("✅ Successfully imported DecisionTreeClassifier.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---")
    print(f"Error importing 'DecisionTreeClassifier' from 'DecisionTree.py': {e}")
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
    
    # Set df_processed (no leaky feature to drop this time)
    df_processed = df.copy()

    # --- !!! THIS IS THE FIX !!! ---
    # Define X (features) and y (target) using the correct column name
    X = df_processed.drop('NObeyesdad', axis=1)
    y = df_processed['NObeyesdad']
    # --- !!! END OF FIX !!! ---

    # --- Define Feature Types for Obesity Dataset ---
    CONTINUOUS_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # All other non-binary text columns
    CATEGORICAL_FEATURES = ['Gender', 'CAEC', 'CALC', 'MTRANS']
    
    # Binary 'yes'/'no' columns
    BINARY_FEATURES = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    
    print(f"Total features defined: {len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES) + len(BINARY_FEATURES)}")

    # --- Create Preprocessing Pipelines ---
    
    # Pipeline for continuous data: Impute (mean) -> Scale (StandardScaler)
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical data: Impute (most frequent) -> One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Pipeline for binary 'yes'/'no' data: Impute (most frequent) -> Convert to 1/0
    # We must do this mapping *before* the ColumnTransformer
    print("Converting binary 'yes'/'no' columns to 1/0...")
    for col in BINARY_FEATURES:
        if X[col].dtype == 'object':
            X[col] = X[col].map({'yes': 1, 'no': 0})
            
    # Now we can just treat them as "binary_numeric"
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # --- Create the Full ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, CONTINUOUS_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bin', binary_transformer, BINARY_FEATURES)
        ],
        remainder='passthrough' # Will pass through any columns not listed
    )

    # Apply the pipeline
    print("Fitting and transforming all features (X)...")
    X_processed = preprocessor.fit_transform(X)
    
    # Encode the target variable (e.g., 'Normal_Weight' -> 0)
    print("Encoding target variable (y)...")
    le = LabelEncoder()
    y_processed = le.fit_transform(y)
    
    print("\nPreprocessing complete.")
    print(f"Final processed data shape: {X_processed.shape}")
    print(f"Final target shape: {y_processed.shape}")

    # --- 3. Train Decision Tree Model ---
    print("\n--- Training Decision Tree Model ---")
    model = DecisionTreeClassifier(
        max_depth=10, 
        min_samples_split=5
    )
    
    start_time = time.time()
    model.fit(X_processed, y_processed)
    duration = time.time() - start_time
    
    print(f"✅ Training complete. (Time: {duration:.2f}s)")

    # --- 4. Save the Model ---
    print(f"\n--- Saving Model ---")
    try:
        # Save the *entire pipeline* (preprocessor + model) for easier use later
        # We create a final pipeline to bundle them
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit the full pipeline one last time on the *raw* X and *encoded* y
        # Note: We must re-fit the preprocessor on the raw data
        print("Re-fitting full pipeline (preprocessor + model) for saving...")
        
        # We need to re-map the binary columns for the *raw* X
        X_for_pipeline = df.drop('NObeyesdad', axis=1)
        for col in BINARY_FEATURES:
            if X_for_pipeline[col].dtype == 'object':
                X_for_pipeline[col] = X_for_pipeline[col].map({'yes': 1, 'no': 0})
                
        full_pipeline.fit(X_for_pipeline, y_processed)
        
        # Save the bundled pipeline
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(full_pipeline, f)
            
        print(f"✅ Full pipeline (preprocessor + model) successfully saved to: {MODEL_SAVE_PATH}")
        print("   You can now load this .pkl file to make predictions on new, raw data.")

    except Exception as e:
        import traceback
        print(f"❌ Error saving model: {e}")
        traceback.print_exc()

    print(f"\n{'='*25} Pipeline Finished {'='*25}")