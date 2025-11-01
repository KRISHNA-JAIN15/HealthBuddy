import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import warnings
from collections import Counter

# --- Sklearn Imports (for Preprocessing) ---
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
MODEL_SAVE_PATH = r'logreg_ckd_model.pkl' # <-- New Model Name

# --- 3. Import Your Custom Classifiers ---
print(f"Importing your custom classifiers from '{CLASSIFIER_DIR}'...")
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # Import the classes needed for this pipeline
    from LogisticRegression import LogisticRegression
    from LinearDiscriminantAnalysis import LDA # Used as dependency for OvR import fix
    from KNN import KNN # Used as dependency for OvR import fix
    # We must import all dependencies used in the OvRWrapper
    from AdaBoost import AdaBoost, DecisionStump, StumpNode
    from MultiLayerPerceptron import SimpleMLP
    from SVM import LinearSVM
    
    print("✅ Successfully imported LogisticRegression and dependencies.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---\nError importing a class: {e}")
    sys.exit(1)


# --- 4. OvRWrapper (Required for Binary Models) ---
# Paste the necessary wrapper here to be self-contained
class OvRWrapper:
    """Wraps a binary classifier to support multi-class OvR classification."""
    def __init__(self, binary_classifier_class, **kwargs):
        self.binary_classifier_class = binary_classifier_class
        self.kwargs = kwargs
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.binary_le = LabelEncoder() # Use 0/1
        y_binary = self.binary_le.fit_transform(y)
        model = self.binary_classifier_class(**self.kwargs)
        model.fit(X, y_binary)
        self.models[self.binary_le.classes_[1]] = model 
    
    def predict(self, X):
        model_key = list(self.models.keys())[0]
        binary_preds = self.models[model_key].predict(X)
        return self.binary_le.inverse_transform(binary_preds)


# --- 5. Main Execution ---
if __name__ == "__main__":
    
    # --- Check Paths ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    print(f"\n{'='*25} Starting Training Pipeline {'='*25}")

    # --- 1. Load Data ---
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

    # --- 2. Data Cleaning & Type Conversion (FIXED) ---
    print("Starting data cleaning and type conversion...")
    
    NUMERIC_FEATURES = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    NOMINAL_FEATURES = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    TARGET = 'class'

    # Convert known numeric columns (including numeric nominals)
    for col in NUMERIC_FEATURES + ['sg', 'al', 'su']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean whitespace from *only* the true string/object columns
    # We must be careful not to include the numeric columns we just converted
    nominal_string_cols = [col for col in NOMINAL_FEATURES if col not in ['sg', 'al', 'su']]
    
    for col in nominal_string_cols + [TARGET]:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('ckd\t', 'ckd') # Fix known issue in target

    # Define X and y (features and target)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # CRITICAL: Drop rows where the target 'class' is missing
    y_known_idx = y.dropna().index
    X = X.loc[y_known_idx]
    y = y.loc[y_known_idx]
    
    print(f"Dataset cleaned. Final shape: {X.shape}")


    # --- 3. Create Preprocessing Pipeline (Impute, Scale, OHE) ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('nom', nominal_transformer, NOMINAL_FEATURES)
        ],
        remainder='drop'
    )

    # Encode the target variable (e.g., 'ckd' -> 0)
    le = LabelEncoder()
    y_processed = le.fit_transform(y)
    
    print("\nPreprocessing pipeline defined.")
    
    
    # --- 4. Train Logistic Regression Model (Wrapped in OvR) ---
    print("\n--- Training Logistic Regression Model ---")
    
    # Wrap the LogisticRegression model for compatibility and multi-class support (even though it's binary here)
    model_logreg = OvRWrapper(
        LogisticRegression,
        learning_rate=0.01,
        n_iters=1000
    )
    
    # --- 5. Create and Save the *Full Pipeline* ---
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_logreg) # Use the OvR-wrapped model
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

    except Exception as e:
        import traceback
        print(f"❌ Error saving model: {e}")
        traceback.print_exc()

    print(f"\n{'='*25} Pipeline Finished {'='*25}")