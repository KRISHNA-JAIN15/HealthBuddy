import pickle
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter # This is needed for your KNN class to load

# --- 1. Configuration ---

# ❗ **ACTION REQUIRED**: Update this to the FULL, ABSOLUTE path to your code folder.
# pickle needs this to find your custom class definitions (e.g., KNN)
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

MODEL_FILENAME = 'knn_ckd_model.pkl'

# --- 2. Add Classifier Path and Import Definitions ---
# This is the crucial fix for the 'ModuleNotFoundError'.
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    print("Please update the path in this script (line 16).")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # We must import the class definitions so pickle can use them
    from KNN import KNN
except ImportError as e:
    print(f"Could not import 'KNN' class from {CLASSIFIER_DIR}")
    print(f"Error: {e}")
    sys.exit(1)


# --- 3. Define the Class Names (Target Labels) ---
# This must match the order from the LabelEncoder (alphabetical)
CLASS_NAMES = [
    'ckd', 
    'notckd'
]

# --- 4. Define Mock Data Samples for Prediction ---
# These samples must be RAW (not preprocessed).
# The saved pipeline will handle all scaling and one-hot encoding.

# Mock Sample 1: Based on typical 'ckd' (high risk) profile
# High 'sc', low 'hemo', 'al' > 0, 'sg' low, 'htn' = yes
sample_1_ckd = {
    'age': 62.0,
    'bp': 80.0,
    'sg': '1.010', # Nominal
    'al': '3.0',   # Nominal
    'su': '0.0',   # Nominal
    'rbc': 'abnormal', # Nominal
    'pc': 'normal',    # Nominal
    'pcc': 'present',  # Nominal
    'ba': 'notpresent', # Nominal
    'bgr': 423.0,
    'bu': 53.0,
    'sc': 1.8,
    'sod': 135.0, # Using a typical value, as it was missing
    'pot': 4.0,   # Using a typical value, as it was missing
    'hemo': 9.6,
    'pcv': 31.0,
    'wc': 7500.0,
    'rc': 3.8,    # Using a typical value, as it was missing
    'htn': 'yes', # Nominal
    'dm': 'yes',  # Nominal
    'cad': 'no',  # Nominal
    'appet': 'poor', # Nominal
    'pe': 'no',   # Nominal
    'ane': 'yes'  # Nominal
}

# Mock Sample 2: Based on typical 'notckd' (low risk) profile
# Low 'sc', high 'hemo', 'al' = 0, 'sg' high, 'htn' = no
sample_2_notckd = {
    'age': 40.0,
    'bp': 80.0,
    'sg': '1.025', # Nominal
    'al': '0.0',   # Nominal
    'su': '0.0',   # Nominal
    'rbc': 'normal',   # Nominal
    'pc': 'normal',    # Nominal
    'pcc': 'notpresent', # Nominal
    'ba': 'notpresent', # Nominal
    'bgr': 120.0,
    'bu': 40.0,
    'sc': 1.0,
    'sod': 142.0,
    'pot': 4.5,
    'hemo': 15.0,
    'pcv': 48.0,
    'wc': 8000.0,
    'rc': 5.2,
    'htn': 'no',  # Nominal
    'dm': 'no',   # Nominal
    'cad': 'no',  # Nominal
    'appet': 'good', # Nominal
    'pe': 'no',   # Nominal
    'ane': 'no'   # Nominal
}


# Define the original column order (must match the order used during training)
COLUMN_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
    'cad', 'appet', 'pe', 'ane'
]

# --- 5. Load Model and Predict ---
try:
    print(f"Loading model from '{MODEL_FILENAME}'...")
    with open(MODEL_FILENAME, 'rb') as f:
        # This line will now work because Python knows what 'KNN' is
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Convert the list of sample dictionaries into a DataFrame
    new_samples_df = pd.DataFrame([sample_1_ckd, sample_2_notckd], columns=COLUMN_ORDER)

    # Make predictions
    # The 'model' is the full_pipeline, so it will:
    # 1. Take the raw DataFrame
    # 2. Run the internal preprocessor (impute, scale, one-hot encode)
    # 3. Run the internal KNN's predict method
    # 4. Output the encoded labels (e.g., [0, 1])
    predictions_encoded = model.predict(new_samples_df)
    
    # Map the encoded labels back to the string names
    predictions_labels = [CLASS_NAMES[p] for p in predictions_encoded]

    print("\n--- Prediction Complete ---")
    
    print(f"\nSample 1 (Expected 'ckd'):")
    print(f"  Predicted Encoded Label: {predictions_encoded[0]}")
    print(f"  Predicted Class Name: {predictions_labels[0]}")
    
    print(f"\nSample 2 (Expected 'notckd'):")
    print(f"  Predicted Encoded Label: {predictions_encoded[1]}")
    print(f"  Predicted Class Name: {predictions_labels[1]}")

except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()