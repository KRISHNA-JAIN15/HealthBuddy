import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# Feature extraction functions
# ---------------------------

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

# ---------------------------
# Combine all feature extractors
# ---------------------------

def extract_features_from_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))

    hog_f = extract_hog(img)
    glcm_f = extract_glcm_features(img)
    lbp_f = extract_lbp_features(img)
    stat_f = extract_statistical_features(img)

    combined = np.hstack([hog_f, glcm_f, lbp_f, stat_f])
    return combined

# ---------------------------
# Dataset loader
# ---------------------------

def load_dataset(folder_path):
    features, labels = [], []
    for label in os.listdir(folder_path):
        subfolder = os.path.join(folder_path, label)
        if not os.path.isdir(subfolder):
            continue
        for file in os.listdir(subfolder):
            img_path = os.path.join(subfolder, file)
            try:
                feature_vector = extract_features_from_image(img_path)
                features.append(feature_vector)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return np.array(features), np.array(labels)

# ---------------------------
# Load Training & Testing
# ---------------------------

train_path = "Training"
test_path = "Testing"

X_train, y_train = load_dataset(train_path)
X_test, y_test = load_dataset(test_path)

print("Feature extraction completed.")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# ---------------------------
# Feature scaling & PCA
# ---------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=100)  # Adjust components as needed
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ---------------------------
# Train Model (Random Forest)
# ---------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

# ---------------------------
# Evaluate
# ---------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
