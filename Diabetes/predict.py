import pickle
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter # Needed by KNN and Decision Trees

# --- 1. Configuration ---

# ❗ **ACTION REQUIRED**: Update this to the FULL, ABSOLUTE path to your code folder.
# This is NECESSARY for pickle to find the definitions for LogisticRegression, OvRWrapper, etc.
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

MODEL_FILENAME = 'logreg_ckd_model.pkl'

# --- 2. Add Classifier Path and Import Dependencies ---
# Python must find these classes *before* loading the model file.
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    print("Please update the path in this script (line 16).")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # Import all custom classes that were part of the saved pipeline
    from LogisticRegression import LogisticRegression
    from LinearDiscriminantAnalysis import LDA
    from KNN import KNN
    from AdaBoost import AdaBoost, DecisionStump, StumpNode
    from MultiLayerPerceptron import SimpleMLP
    from SVM import LinearSVM
    
    # We redefine the OvRWrapper here so it's guaranteed to be available
    # for pickle to find (as it's part of the saved pipeline structure).
    class OvRWrapper:
        def __init__(self, binary_classifier_class, **kwargs):
            self.binary_classifier_class = binary_classifier_class
            self.kwargs = kwargs
            self.models = {}
            self.classes_ = None
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.binary_le = LabelEncoder()
            y_binary = self.binary_le.fit_transform(y)
            model = self.binary_classifier_class(**self.kwargs)
            model.fit(X, y_binary)
            self.models[self.binary_le.classes_[1]] = model 
        def predict(self, X):
            model_key = list(self.models.keys())[0]
            binary_preds = self.models[model_key].predict(X)
            return self.binary_le.inverse_transform(binary_preds)
    
    print("✅ Successfully imported custom class definitions.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---\nError importing a class: {e}")
    sys.exit(1)


# --- 3. Define the Class Names (Target Labels) and Column Order ---

# This must match the order from the training LabelEncoder (alphabetical)
CLASS_NAMES = [
    'ckd',      # Encoded as 0
    'notckd'    # Encoded as 1
]

# Full list of original feature names
COLUMN_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
    'cad', 'appet', 'pe', 'ane'
]

# --- 4. Define Mock Data Samples (RAW Format) ---
# The pipeline handles '?' (missing), scaling, and OHE.

# Mock Sample 1: HIGH RISK (Expected 'ckd' - high waste, low blood cell count)
sample_1_ckd = {
    'age': 65.0,
    'bp': 130.0,
    'sg': 1.010,
    'al': 4.0,   # High Albumin
    'su': 3.0,
    'rbc': 'abnormal',
    'pc': 'abnormal',
    'pcc': 'present',
    'ba': 'yes',
    'bgr': 250.0,
    'bu': 120.0,
    'sc': 5.5,   # High Creatinine
    'sod': 130.0,
    'pot': 6.0,
    'hemo': 9.0,   # Low Hemoglobin
    'pcv': 28.0, # Low PCV
    'wc': 12000.0,
    'rc': 3.0,
    'htn': 'yes',
    'dm': 'yes',
    'cad': 'yes',
    'appet': 'poor',
    'pe': 'yes',
    'ane': 'yes'
}

# Mock Sample 2: LOW RISK (Expected 'notckd' - healthy profile)
sample_2_notckd = {
    'age': 35.0,
    'bp': 80.0,
    'sg': 1.025,
    'al': 0.0,   # Zero Albumin
    'su': 0.0,
    'rbc': 'normal',
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': 90.0,
    'bu': 20.0,
    'sc': 0.8,
    'sod': 145.0,
    'pot': 4.0,
    'hemo': 15.0, # High Hemoglobin
    'pcv': 45.0,
    'wc': 6500.0,
    'rc': 5.5,
    'htn': 'no',
    'dm': 'no',
    'cad': 'no',
    'appet': 'good',
    'pe': 'no',
    'ane': 'no'
}

# --- 5. Load Model and Predict ---
try:
    print(f"Loading model from '{MODEL_FILENAME}'...")
    with open(MODEL_FILENAME, 'rb') as f:
        # Load the full pipeline object
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Combine mock samples into a single DataFrame
    new_samples_df = pd.DataFrame([sample_1_ckd, sample_2_notckd], columns=COLUMN_ORDER)
    print("\n--- Making Predictions ---")

    # The pipeline handles all preprocessing, just pass the raw data
    predictions_encoded = model.predict(new_samples_df)
    
    # Map the encoded labels back to the string names
    predictions_labels = [CLASS_NAMES[p] for p in predictions_encoded]

    print(f"Sample 1 (High Risk Profile):")
    print(f"  Predicted Encoded Label: {predictions_encoded[0]}")
    print(f"  Predicted Class Name: {predictions_labels[0]}")
    
    print(f"Sample 2 (Low Risk Profile):")
    print(f"  Predicted Encoded Label: {predictions_encoded[1]}")
    print(f"  Predicted Class Name: {predictions_labels[1]}")

except FileNotFoundError:
    print(f"\nError: Model file not found at '{MODEL_FILENAME}'.")
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")
    import traceback
    traceback.print_exc()