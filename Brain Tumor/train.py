import cv2
import os
import sys
import numpy as np
import pandas as pd
import warnings
import pickle  
from tqdm import tqdm # Added for a nice progress bar

# --- Sklearn/Skimage Imports ---
try:
    from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
except ImportError:
    print("Error: scikit-image not found. Please install it: pip install scikit-image")
    sys.exit(1)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 1. Import Your From-Scratch KNN ---

# ❗ **ACTION REQUIRED**: Update this to the FULL, ABSOLUTE path to your code folder.
# Example: CLASSIFIER_DIR = r'C:\Users\jaink\OneDrive\Desktop\ML_Project\Classifier_codes'
CLASSIFIER_DIR = r'/home/lightshadow/pseudoD/Sem 5/HealthBuddy/Classifier_codes'

if not os.path.isdir(CLASSIFIER_DIR):
    print(f"Error: Classifier directory '{CLASSIFIER_DIR}' not found.")
    print("Please update the path in this cell.")
    sys.exit(1)
else:
    # Add the folder to the system path to allow imports
    if CLASSIFIER_DIR not in sys.path:
        sys.path.append(os.path.abspath(CLASSIFIER_DIR))
    
    try:
        # Import only the KNN class
        from KNN import KNN
        print("✅ Successfully imported KNN classifier from 'Classifier_codes'.")
    except ImportError as e:
        print(f"--- 🛑 IMPORT ERROR ---")
        print(f"Error importing 'KNN' from 'KNN.py': {e}")
        print("Please check your file and class names.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during import: {e}")
        sys.exit(1)


# --- 2. Feature extraction functions ---
# (Exactly as you provided)

def extract_hog(img):
    hog_features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys',
                          visualize=True, transform_sqrt=True)
    return hog_features

def extract_glcm_features(img):
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return features

def extract_lbp_features(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 10),
                             range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_statistical_features(img):
    mean = np.mean(img)
    var = np.var(img)
    skew = np.mean((img - mean)**3) / (np.std(img)**3 + 1e-6)
    kurt = np.mean((img - mean)**4) / (np.std(img)**4 + 1e-6)
    entropy = -np.sum((img/255) * np.log2(img/255 + 1e-6))
    return [mean, var, skew, kurt, entropy]

# --- 3. Combine all feature extractors ---

def extract_features_from_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        return None
    
    img = cv2.resize(img, (128, 128))

    hog_f = extract_hog(img)
    glcm_f = extract_glcm_features(img)
    lbp_f = extract_lbp_features(img)
    stat_f = extract_statistical_features(img)

    combined = np.hstack([hog_f, glcm_f, lbp_f, stat_f])
    return combined

# --- 4. Dataset loader ---

def load_dataset(folder_path):
    features, labels = [], []
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at {folder_path}")
        return np.array([]), np.array([])
        
    for label in os.listdir(folder_path):
        subfolder = os.path.join(folder_path, label)
        if not os.path.isdir(subfolder):
            continue
        
        print(f"  Processing class: {label}")
        # Use TQDM for a progress bar
        for file in tqdm(os.listdir(subfolder), desc=f'  {label}'):
            img_path = os.path.join(subfolder, file)
            try:
                feature_vector = extract_features_from_image(img_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(features), np.array(labels)

# --- 5. Load Training & Testing ---

# ❗ **ACTION REQUIRED**: Set your data folder paths relative to this script
# (Or provide absolute paths)
train_path = "Training"
test_path = "Testing"

if not os.path.isdir(train_path) or not os.path.isdir(test_path):
    print(f"Error: '{train_path}' or '{test_path}' folders not found.")
    print("Please make sure these folders are in the same directory as the script, or provide full paths.")
else:
    print(f"Loading and extracting features from '{train_path}'...")
    X_train, y_train_labels = load_dataset(train_path)
    
    print(f"\nLoading and extracting features from '{test_path}'...")
    X_test, y_test_labels = load_dataset(test_path)

    print("\nFeature extraction completed.")
    print("Training samples:", X_train.shape)
    print("Testing samples:", X_test.shape)

    # --- 6. Feature scaling, Label Encoding, & PCA ---

    print("\nEncoding string labels to numbers...")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_labels)
    y_test = le.transform(y_test_labels)
    print(f"Classes found: {list(le.classes_)}")

    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    N_COMPONENTS = 100
    print(f"Applying PCA (n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Data preprocessed. Final feature shape: {X_train_pca.shape}")


    # --- 7. Train Model (KNN) ---
    
    # ❗ **PARAMETER**: You can change 'k=4' to any number you want to test
    K_VALUE = 4 
    
    print(f"\n--- Training KNN Model (k={K_VALUE}) ---")
    model = KNN(k=K_VALUE)
    model.fit(X_train_pca, y_train) # Use scaled, PCA-reduced data
    
    print("\n--- Predicting on Test Set ---")
    y_pred = model.predict(X_test_pca)

    # --- 8. Evaluate ---
    print("\n--- Evaluation Complete ---")
    
    accuracy = accuracy_score(y_test, y_pred)
    # --- 8. Evaluate ---
    print("\n--- Evaluation Complete ---")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")
    # Use target_names from the LabelEncoder to show full class names
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    
    # --- 9. Save the Model and Preprocessors ---
    
    print("\n--- Saving Model & Pipeline ---")
    
    # Bundle all necessary components into a dictionary
    # This is crucial, as you need the scaler and PCA to process new images
    pipeline_components = {
        'model': model,
        'label_encoder': le,
        'scaler': scaler,
        'pca': pca,
        'class_names': list(le.classes_) # Save class names for convenience
    }
    
    # Define the output filename
    output_filename = "knn_pipeline.pkl"
    
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(pipeline_components, f)
        print(f"✅ Successfully saved pipeline components to '{output_filename}'")
    except Exception as e:
        print(f"--- 🛑 ERROR SAVING MODEL ---")
        print(f"An error occurred: {e}")

# This should be the end of the 'else' blockconda