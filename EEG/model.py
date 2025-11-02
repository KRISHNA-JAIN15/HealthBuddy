import os
import sys
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle

# --- 1. Standard Library Imports ---
print("Loading libraries (sklearn, pandas, numpy, matplotlib)...")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, RocCurveDisplay
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 2. Configuration (NEEDS YOUR INPUT) ---

# ❗ **ACTION REQUIRED**: Update this path to the *single* folder containing your data.
# e.g., r'C:\Users\jaink\OneDrive\Desktop\ML_Project\EEG'
DATASET_DIR = r'C:\Users\Research PC\Desktop\ML\EEG' 

# --- 3. Helper: Find Data Files ---
def find_data_files(folder_path):
    """Finds train/test files, prioritizing 'features' files."""
    files = [f.lower() for f in os.listdir(folder_path)]
    
    # Priority 1: Look for *features* files
    train_feat_file = next((f for f in files if "train_features.csv" in f), None)
    test_feat_file = next((f for f in files if "test_features.csv" in f), None)
    if train_feat_file and test_feat_file:
        print(f"Found pre-computed feature files: {train_feat_file}, {test_feat_file}")
        return os.path.join(folder_path, train_feat_file), os.path.join(folder_path, test_feat_file), "features"

    # Priority 2: Look for *data* files
    train_data_file = next((f for f in files if re.search(r'(train_data|train)\.csv', f)), None)
    test_data_file = next((f for f in files if re.search(r'(test_data|test)\.csv', f)), None)
    if train_data_file and test_data_file:
        print(f"Found raw data files: {train_data_file}, {test_data_file}")
        return os.path.join(folder_path, train_data_file), os.path.join(folder_path, test_data_file), "raw"

    return None, None, None


# --- 4. Data Loading and Preprocessing Function ---
def load_and_preprocess_data(dataset_path):
    """Loads, cleans, and prepares data from the given folder."""
    print(f"--- Loading data from: {dataset_path} ---")
    train_file, test_file, mode = find_data_files(dataset_path)

    if mode is None:
        print(f"Error: No *features.csv or *data.csv files found in {dataset_path}.")
        return None
        
    imputer = None  # <-- **FIX 1**: Initialize imputer to None
    
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        print(f"Loaded train ({df_train.shape}) and test ({df_test.shape})")

        # --- Identify Target ---
        if 'Session' in df_train.columns:
            target_col = 'Session'
        else:
            target_col = df_train.columns[-1]
            print(f"Warning: Assuming last column '{target_col}' is the target variable.")

        # --- Preprocessing ---
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[target_col])
        y_test = le.transform(df_test[target_col])
        n_classes = len(le.classes_)
        print(f"Target variable '{target_col}' encoded into {n_classes} classes: {list(le.classes_)}")

        # Drop 'Subject' (identifier) and target column
        drop_cols = [target_col, 'Subject', 'Unnamed: 0']
        X_train_raw = df_train.drop(columns=drop_cols, errors='ignore')
        X_test_raw = df_test.drop(columns=drop_cols, errors='ignore')
        
        # Align columns just in case
        X_train_raw, X_test_raw = X_train_raw.align(X_test_raw, join='left', axis=1, fill_value=0)

        # Impute missing values (even in feature files, just in case)
        if X_train_raw.isnull().sum().sum() > 0 or X_test_raw.isnull().sum().sum() > 0:
            print("Imputing missing values (mean strategy)...")
            imputer = SimpleImputer(strategy='mean') # <-- **FIX 2**: Assign to imputer var
            X_train_imputed = imputer.fit_transform(X_train_raw)
            X_test_imputed = imputer.transform(X_test_raw)
        else:
            print("No missing values found.")
            X_train_imputed = X_train_raw.values
            X_test_imputed = X_test_raw.values

        # Get feature names *after* all column ops
        feature_names = X_train_raw.columns.tolist()

        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_imputed, columns=feature_names)
        X_test = pd.DataFrame(X_test_imputed, columns=feature_names)
        
        return {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "n_classes": n_classes, "feature_names": feature_names,
            "label_encoder": le,
            "imputer": imputer  # <-- **FIX 3**: Return the imputer in the dictionary
        }
    except Exception as e:
        print(f"Error processing data in {dataset_path}: {e}")
        return None

# --- 5. Plotting Function for Multi-Class ROC ---
def plot_multiclass_roc(model, X_test, y_test, n_classes, class_names):
    """Plots a One-vs-Rest (OvR) ROC curve for a multi-class model."""
    y_pred_proba = model.predict_proba(X_test)

    # Dictionary to hold FPR, TPR, and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    
    # Calculate ROC for each class
    for i in range(n_classes):
        # Treat class 'i' as positive, all others as negative
        y_test_binary = (y_test == i).astype(int)
        class_score = y_pred_proba[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_test_binary, class_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve for {class_names[i]} (area = {roc_auc[i]:0.3f})')

    # Plot the "random guess" line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    print("Displaying ROC AUC Plot...")
    plt.show()


# --- 6. Main Execution ---
if __name__ == "__main__":
    
    # --- Setup ---
    if DATASET_DIR == r'path/to/your/EEG_folder':
        print("="*60)
        print("🛑 ERROR: Please update the 'DATASET_DIR' variable (line 29)")
        print("       to point to your folder containing the data (e.g., .../EEG).")
        print("="*60)
        sys.exit(1)
    if not os.path.isdir(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        sys.exit(1)
    
    dataset_name = os.path.basename(DATASET_DIR)
    print(f"\n{'='*25} Processing Dataset: {dataset_name} {'='*25}")
    
    # --- Load Data ---
    data = load_and_preprocess_data(DATASET_DIR)
    
    if data is None:
        print("Could not load data. Exiting.")
        sys.exit(1)
        
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    le = data['label_encoder']
    class_names = le.classes_
    n_classes = data['n_classes']

    # --- 1. Train Random Forest ---
    print("\n--- 1. Training scikit-learn Random Forest ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training complete. (Time: {time.time() - start_time:.2f}s)")
    
    # --- 2. Get Predictions & Classification Report ---
    print("\n--- 2. Classification Report ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # --- 3. Plot ROC AUC Curve ---
    print("\n--- 3. Generating ROC AUC Plot ---")
    try:
        plot_multiclass_roc(model, X_test, y_test, n_classes, class_names)
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

    # --- 4. Save Model & Pipeline ---
    
    print("\n--- 4. Saving Model & Pipeline ---")
    
    # Retrieve the imputer (it will be None if no imputation was done)
    imputer = data['imputer'] # <-- This will now work correctly
    
    # Bundle all necessary components into a dictionary
    pipeline_components = {
        'model': model,
        'label_encoder': le,
        'imputer': imputer,
        'feature_names': data['feature_names'], # Save feature names for prediction
        'class_names': list(class_names)
    }
    
    # Define the output filename
    output_filename = "rf_eeg_pipeline.pkl"
    
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(pipeline_components, f)
        print(f"✅ Successfully saved pipeline components to '{output_filename}'")
    except Exception as e:
        print(f"--- 🛑 ERROR SAVING MODEL ---")
        print(f"An error occurred: {e}")

    # --- This print statement was already here ---
    print(f"\n{'='*25} 🏁 Analysis Complete 🏁 {'='*25}")