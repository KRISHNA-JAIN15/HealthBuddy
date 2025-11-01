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

# ❗ **ACTION REQUIRED**: Make sure this path points to your GallStone CSV file.
DATA_PATH = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\GallStone\gallstone.csv'

# ❗ **ACTION REQUIRED**: Make sure this path points to your 'Classifier_codes' folder.
CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'

# ❗ **ACTION REQUIRED**: Define where to save the final model.
MODEL_SAVE_PATH = r'gallstone_adaboost_model.pkl'


# --- 3. Import Your Custom AdaBoost Classifier ---
print(f"Importing your custom classifier from '{CLASSIFIER_DIR}'...")
if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    sys.exit(1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.append(os.path.abspath(CLASSIFIER_DIR))

try:
    # Import the classes needed for AdaBoost.
    # (AdaBoost.py needs DecisionStump and StumpNode to run)
    from AdaBoost import AdaBoost, DecisionStump, StumpNode
    print("✅ Successfully imported AdaBoost and dependencies.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---")
    print(f"Error importing 'AdaBoost' from 'AdaBoost.py': {e}")
    sys.exit(1)


# --- 4. Helper Class for Binary Classification ---
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
        if hasattr(self, 'binary_le') and self.binary_le:
             # Binary case
             model_key = list(self.models.keys())[0]
             binary_preds = self.models[model_key].predict(X)
             return self.binary_le.inverse_transform(binary_preds)
        else:
            # Multi-class OvR case
            all_scores = []
            for cls in self.classes_:
                model = self.models[cls]
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    all_scores.append(probas[:, 1] if probas.ndim == 2 and probas.shape[1] == 2 else probas)
                elif hasattr(model, 'decision_function'):
                    all_scores.append(model.decision_function(X))
                else:
                    all_scores.append(model.predict(X))
            
            scores_matrix = np.stack(all_scores, axis=1)
            best_class_indices = np.argmax(scores_matrix, axis=1)
            return self.classes_[best_class_indices]


# --- 5. Main Execution ---
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

    # --- 2. Preprocessing (matching your notebook preprocessing) ---
    print("Starting preprocessing...")
    
    # Define target variable
    TARGET = 'Gallstone Status'
    
    # Define X (features) and y (target)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Available columns: {list(X.columns)}")

    # --- Define Feature Lists (matching your notebook) ---
    # All features that are continuous, integer, or binary that need scaling
    NUMERIC_FEATURES = [
        'Age', 'Height', 'Weight', 'Body_Mass_Index_BMI', 'Total_Body_Water_TBW', 
        'Extracellular_Water_ECW', 'Intracellular_Water_ICW', 'Extracellular_Fluid/Total_Body_Water_ECF/TBW', 
        'Total_Body_Fat_Ratio_TBFR', 'Lean_Mass_LM', 'Body_Protein_Content_Protein', 
        'Visceral_Fat_Rating_VFR', 'Bone_Mass_BM', 'Muscle_Mass_MM', 'Obesity', 
        'Total_Fat_Content_TFC', 'Visceral_Fat_Area_VFA', 'Visceral_Muscle_Area_VMA_Kg', 
        'Glucose', 'Total_Cholesterol_TC', 'Low_Density_Lipoprotein_LDL', 
        'High_Density_Lipoprotein_HDL', 'Triglyceride', 'Aspartat_Aminotransferaz_AST', 
        'Alanin_Aminotransferaz_ALT', 'Alkaline_Phosphatase_ALP', 'Creatinine', 
        'Glomerular_Filtration_Rate_GFR', 'C-Reactive_Protein_CRP', 'Hemoglobin_HGB', 
        'Vitamin_D', 'Coronary_Artery_Disease_CAD', 'Hypothyroidism', 'Hyperlipidemia', 'Diabetes_Mellitus_DM'
    ]

    # True Categorical/Ordinal features that need OHE
    CATEGORICAL_FEATURES = ['Gender', 'Comorbidity', 'Hepatic_Fat_Accumulation_HFA']

    # Filter features to only include those that actually exist in the dataset
    existing_numeric_features = [f for f in NUMERIC_FEATURES if f in X.columns]
    existing_categorical_features = [f for f in CATEGORICAL_FEATURES if f in X.columns]

    print(f"Filtered NUMERIC_FEATURES ({len(existing_numeric_features)} features)")
    print(f"Filtered CATEGORICAL_FEATURES ({len(existing_categorical_features)} features)")

    # Update the feature lists to use only existing features
    NUMERIC_FEATURES = existing_numeric_features
    CATEGORICAL_FEATURES = existing_categorical_features

    # --- Create Preprocessing Pipelines ---
    # Pipeline for NUMERIC data: Impute (mean) -> Scale (StandardScaler)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for CATEGORICAL data: Impute (most frequent) -> One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- Create the Full ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    print("Preprocessing setup complete.")
    
    # --- 3. Train AdaBoost Model ---
    print("\n--- Training AdaBoost Model ---")
    
    # Initialize your from-scratch AdaBoost model wrapped in OvR
    model_adaboost = OvRWrapper(AdaBoost, n_estimators=50)
    
    # --- 4. Create and Save the *Full Pipeline* ---
    # This pipeline bundles the preprocessor and the model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_adaboost)
    ])
    
    # Fit the full pipeline on the raw X and y
    print("Fitting full pipeline (preprocessor + model) for saving...")
            
    start_time = time.time()
    # Pass the raw feature DataFrame (X) and the target (y)
    full_pipeline.fit(X, y)
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

    