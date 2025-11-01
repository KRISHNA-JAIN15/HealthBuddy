import pickle
import pandas as pd
import numpy as np
import sys
import os

# --- 1. Configuration ---

# ❗ **ACTION REQUIRED**: Update this to the FULL, ABSOLUTE path to your code folder.
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

MODEL_FILENAME = 'decision_tree_model.pkl'

# --- 2. Add Classifier Path and Import Definitions ---
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    print("Please update the path in this script (line 16).")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # We must import the class definitions so pickle can use them
    from DecisionTree import DecisionTreeClassifier, Node
except ImportError as e:
    print(f"Could not import DecisionTreeClassifier from {CLASSIFIER_DIR}")
    print(f"Error: {e}")
    sys.exit(1)


# --- 3. Define the Class Names (Target Labels) ---
CLASS_NAMES = [
    'Insufficient_Weight', 
    'Normal_Weight', 
    'Obesity_Type_I', 
    'Obesity_Type_II', 
    'Obesity_Type_III', 
    'Overweight_Level_I', 
    'Overweight_Level_II'
]

# --- 4. Define a New Sample for Prediction ---
new_sample = {
    'Gender': 'Male',
    'Age': 25.0,
    'Height': 1.80,
    'Weight': 95.0,
    'family_history_with_overweight': 'yes',
    'FAVC': 'yes',
    'FCVC': 1.0,  # Eats vegetables 1=Never
    'NCP': 3.0,   # 3 main meals
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'CH2O': 1.0,  # 1 liter of water
    'SCC': 'no',
    'FAF': 0.0,   # No physical activity
    'TUE': 2.0,   # High tech use
    'CALC': 'Sometimes',
    'MTRANS': 'Automobile'
}

# Define the original column order (must match the order used during training)
COLUMN_ORDER = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]

# --- 5. Load Model and Predict ---
try:
    print(f"Loading model from '{MODEL_FILENAME}'...")
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Convert the single sample dictionary into a DataFrame
    new_sample_df = pd.DataFrame([new_sample], columns=COLUMN_ORDER)

    # --- !!! THIS IS THE FIX !!! ---
    # Apply the same manual 'yes'/'no' mapping as in the training script
    # The preprocessor saved inside the model *expects* 1s and 0s.
    BINARY_FEATURES = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    print("Converting binary 'yes'/'no' columns to 1/0 for prediction...")
    for col in BINARY_FEATURES:
        if new_sample_df[col].dtype == 'object':
            new_sample_df[col] = new_sample_df[col].map({'yes': 1, 'no': 0})
    # --- !!! END OF FIX !!! ---

    # Make prediction
    # The model.predict() will now receive 1s and 0s for binary features,
    # which is what it was trained on.
    prediction_encoded = model.predict(new_sample_df)
    
    encoded_label = prediction_encoded[0]
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