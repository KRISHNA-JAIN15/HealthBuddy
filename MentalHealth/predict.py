import pickle
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter # Needed by the imported custom classes

# --- 1. Configuration ---

# ❗ **ACTION REQUIRED**: Update this to the FULL, ABSOLUTE path to your code folder.
# pickle needs this to understand the custom class definitions
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

MODEL_FILENAME = 'maternal_health_randomforest.pkl'

# --- 2. Add Classifier Path and Import Definitions ---
# This is the crucial fix for the 'ModuleNotFoundError'.
# Python needs to know where to find your custom .py files.
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    print("Please update the path in this script (line 16).")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # We must import all custom classes that were part of the saved model
    # so pickle can rebuild the objects in memory.
    from DecisionTree import DecisionTreeClassifier, Node
    from RandomForest import RandomForestClassifier
except ImportError as e:
    print(f"Could not import custom classes from {CLASSIFIER_DIR}")
    print(f"Error: {e}")
    sys.exit(1)


# --- 3. Define the Class Names (Target Labels) ---
# This must match the order from the LabelEncoder (alphabetical)
CLASS_NAMES = [
    'high risk', 
    'low risk', 
    'mid risk'
]

# --- 4. Define a New Sample for Prediction ---
# This data must be RAW (not preprocessed)
# Using the 'high risk' example you provided
new_sample = {
    'Age': 25,
    'SystolicBP': 130,
    'DiastolicBP': 80,
    'BS': 15.0,
    'BodyTemp': 98.0,
    'HeartRate': 86
}

# Define the original column order (must match the order used during training)
COLUMN_ORDER = [
    'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'
]

# --- 5. Load Model and Predict ---
try:
    print(f"Loading model from '{MODEL_FILENAME}'...")
    with open(MODEL_FILENAME, 'rb') as f:
        # This line will now work because Python knows what
        # RandomForestClassifier, DecisionTreeClassifier, and Node are.
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Convert the single sample dictionary into a DataFrame
    # It *must* have the same columns in the same order as the training data
    new_sample_df = pd.DataFrame([new_sample], columns=COLUMN_ORDER)

    # Make prediction
    # The 'model' is the full_pipeline, so it will:
    # 1. Take the raw DataFrame
    # 2. Run the internal preprocessor (impute, scale)
    # 3. Run the internal RandomForest's predict method
    # 4. Output the encoded label (e.g., 0, 1, 2...)
    prediction_encoded = model.predict(new_sample_df)
    
    # Get the first (and only) prediction
    encoded_label = prediction_encoded[0]
    
    # Map the encoded label back to the string name
    prediction_label = CLASS_NAMES[encoded_label]

    print("\n--- Prediction Complete ---")
    print(f"Raw Input Data: {new_sample}")
    print(f"Predicted Encoded Label: {encoded_label}")
    print(f"Predicted Class Name: {prediction_label}")

except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()