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

MODEL_FILENAME = 'gallstone_adaboost_model.pkl'

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
    from AdaBoost import AdaBoost, DecisionStump, StumpNode
    from sklearn.preprocessing import LabelEncoder
    print("✅ Successfully imported AdaBoost and dependencies.")
except ImportError as e:
    print(f"Could not import custom classes from {CLASSIFIER_DIR}")
    print(f"Error: {e}")
    sys.exit(1)


# --- 3. Define the OvRWrapper Class (needed for model loading) ---
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


# --- 4. Define the Class Names (Target Labels) ---
# This must match the order from the original dataset
CLASS_NAMES = [
    'No', 
    'Yes'
]

# --- 5. Define Mock Samples for Prediction ---
# This data must be RAW (not preprocessed)
# Mock Sample 1: Patient likely to have gallstones (older, overweight, female)
mock_sample_1 = {
    'Age': 58,
    'Gender': 'Female',
    'Height': 160.5,
    'Weight': 78.2,
    'Body_Mass_Index_BMI': 30.4,
    'Total_Body_Water_TBW': 28.5,
    'Extracellular_Water_ECW': 12.8,
    'Intracellular_Water_ICW': 15.7,
    'Extracellular_Fluid/Total_Body_Water_ECF/TBW': 0.45,
    'Total_Body_Fat_Ratio_TBFR': 38.2,
    'Lean_Mass_LM': 48.3,
    'Body_Protein_Content_Protein': 9.2,
    'Visceral_Fat_Rating_VFR': 12,
    'Bone_Mass_BM': 2.1,
    'Muscle_Mass_MM': 46.2,
    'Obesity': 1,
    'Total_Fat_Content_TFC': 29.9,
    'Visceral_Fat_Area_VFA': 95.6,
    'Visceral_Muscle_Area_VMA_Kg': 42.1,
    'Glucose': 125.0,
    'Total_Cholesterol_TC': 245.0,
    'Low_Density_Lipoprotein_LDL': 145.0,
    'High_Density_Lipoprotein_HDL': 35.0,
    'Triglyceride': 180.0,
    'Aspartat_Aminotransferaz_AST': 45.0,
    'Alanin_Aminotransferaz_ALT': 52.0,
    'Alkaline_Phosphatase_ALP': 125.0,
    'Creatinine': 0.9,
    'Glomerular_Filtration_Rate_GFR': 75.0,
    'C-Reactive_Protein_CRP': 3.2,
    'Hemoglobin_HGB': 12.5,
    'Vitamin_D': 18.0,
    'Coronary_Artery_Disease_CAD': 0,
    'Hypothyroidism': 1,
    'Hyperlipidemia': 1,
    'Diabetes_Mellitus_DM': 1,
    'Comorbidity': 'Yes',
    'Hepatic_Fat_Accumulation_HFA': 'Moderate'
}

# Mock Sample 2: Patient unlikely to have gallstones (younger, healthy weight, male)
mock_sample_2 = {
    'Age': 28,
    'Gender': 'Male',
    'Height': 175.0,
    'Weight': 72.0,
    'Body_Mass_Index_BMI': 23.5,
    'Total_Body_Water_TBW': 42.1,
    'Extracellular_Water_ECW': 16.8,
    'Intracellular_Water_ICW': 25.3,
    'Extracellular_Fluid/Total_Body_Water_ECF/TBW': 0.40,
    'Total_Body_Fat_Ratio_TBFR': 15.2,
    'Lean_Mass_LM': 61.0,
    'Body_Protein_Content_Protein': 11.8,
    'Visceral_Fat_Rating_VFR': 4,
    'Bone_Mass_BM': 3.1,
    'Muscle_Mass_MM': 57.9,
    'Obesity': 0,
    'Total_Fat_Content_TFC': 10.9,
    'Visceral_Fat_Area_VFA': 35.2,
    'Visceral_Muscle_Area_VMA_Kg': 55.7,
    'Glucose': 88.0,
    'Total_Cholesterol_TC': 165.0,
    'Low_Density_Lipoprotein_LDL': 95.0,
    'High_Density_Lipoprotein_HDL': 55.0,
    'Triglyceride': 75.0,
    'Aspartat_Aminotransferaz_AST': 22.0,
    'Alanin_Aminotransferaz_ALT': 18.0,
    'Alkaline_Phosphatase_ALP': 68.0,
    'Creatinine': 0.8,
    'Glomerular_Filtration_Rate_GFR': 110.0,
    'C-Reactive_Protein_CRP': 0.5,
    'Hemoglobin_HGB': 15.2,
    'Vitamin_D': 32.0,
    'Coronary_Artery_Disease_CAD': 0,
    'Hypothyroidism': 0,
    'Hyperlipidemia': 0,
    'Diabetes_Mellitus_DM': 0,
    'Comorbidity': 'No',
    'Hepatic_Fat_Accumulation_HFA': 'None'
}

# Define the original column order (must match the order used during training)
COLUMN_ORDER = [
    'Age', 'Gender', 'Height', 'Weight', 'Body_Mass_Index_BMI', 'Total_Body_Water_TBW', 
    'Extracellular_Water_ECW', 'Intracellular_Water_ICW', 'Extracellular_Fluid/Total_Body_Water_ECF/TBW', 
    'Total_Body_Fat_Ratio_TBFR', 'Lean_Mass_LM', 'Body_Protein_Content_Protein', 
    'Visceral_Fat_Rating_VFR', 'Bone_Mass_BM', 'Muscle_Mass_MM', 'Obesity', 
    'Total_Fat_Content_TFC', 'Visceral_Fat_Area_VFA', 'Visceral_Muscle_Area_VMA_Kg', 
    'Glucose', 'Total_Cholesterol_TC', 'Low_Density_Lipoprotein_LDL', 
    'High_Density_Lipoprotein_HDL', 'Triglyceride', 'Aspartat_Aminotransferaz_AST', 
    'Alanin_Aminotransferaz_ALT', 'Alkaline_Phosphatase_ALP', 'Creatinine', 
    'Glomerular_Filtration_Rate_GFR', 'C-Reactive_Protein_CRP', 'Hemoglobin_HGB', 
    'Vitamin_D', 'Coronary_Artery_Disease_CAD', 'Hypothyroidism', 'Hyperlipidemia', 
    'Diabetes_Mellitus_DM', 'Comorbidity', 'Hepatic_Fat_Accumulation_HFA'
]

# --- 6. Load Model and Make Predictions ---
try:
    print(f"Loading model from '{MODEL_FILENAME}'...")
    with open(MODEL_FILENAME, 'rb') as f:
        # This line will now work because Python knows what
        # AdaBoost, DecisionStump, and StumpNode are.
        model = pickle.load(f)
    print("✅ Model loaded successfully.")

    # Convert the mock samples into DataFrames
    # They *must* have the same columns in the same order as the training data
    mock_samples_df = pd.DataFrame([mock_sample_1, mock_sample_2], columns=COLUMN_ORDER)
    
    print("\n--- Mock Data for Prediction ---")
    print("Sample 1 (High Risk Profile):")
    print(f"  Age: {mock_sample_1['Age']}, Gender: {mock_sample_1['Gender']}")
    print(f"  BMI: {mock_sample_1['Body_Mass_Index_BMI']}, Obesity: {mock_sample_1['Obesity']}")
    print(f"  Diabetes: {mock_sample_1['Diabetes_Mellitus_DM']}, Hyperlipidemia: {mock_sample_1['Hyperlipidemia']}")
    print(f"  Comorbidity: {mock_sample_1['Comorbidity']}, Hepatic Fat: {mock_sample_1['Hepatic_Fat_Accumulation_HFA']}")
    
    print("\nSample 2 (Low Risk Profile):")
    print(f"  Age: {mock_sample_2['Age']}, Gender: {mock_sample_2['Gender']}")
    print(f"  BMI: {mock_sample_2['Body_Mass_Index_BMI']}, Obesity: {mock_sample_2['Obesity']}")
    print(f"  Diabetes: {mock_sample_2['Diabetes_Mellitus_DM']}, Hyperlipidemia: {mock_sample_2['Hyperlipidemia']}")
    print(f"  Comorbidity: {mock_sample_2['Comorbidity']}, Hepatic Fat: {mock_sample_2['Hepatic_Fat_Accumulation_HFA']}")

    # Make predictions
    # The 'model' is the full_pipeline, so it will:
    # 1. Take the raw DataFrame
    # 2. Run the internal preprocessor (impute, scale, one-hot encode)
    # 3. Run the internal AdaBoost's predict method
    # 4. Output the predicted labels
    predictions = model.predict(mock_samples_df)
    
    print("\n--- Prediction Results ---")
    for i, prediction in enumerate(predictions):
        sample_name = f"Sample {i+1}"
        risk_level = "High Risk" if i == 0 else "Low Risk"
        
        print(f"{sample_name} ({risk_level}):")
        print(f"  Predicted Gallstone Status: {prediction}")
        
        # Interpret the result
        if prediction == 'Yes':
            print(f"  🔴 HIGH RISK: Patient is predicted to have gallstones")
        else:
            print(f"  🟢 LOW RISK: Patient is predicted to NOT have gallstones")
        print()

    print("--- Summary ---")
    print(f"Total samples processed: {len(predictions)}")
    print(f"Predictions: {list(predictions)}")

except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{MODEL_FILENAME}'.")
    print("Make sure you have run main.py first to train and save the model.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
    import traceback
    traceback.print_exc()
