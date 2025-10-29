import numpy as np
import os
import sys
from collections import defaultdict

# --- Essential Sklearn imports ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score as sklearn_accuracy_score # Use sklearn's for reference

# --- Add Classifier_codes folder to Python path ---
# Assume the script is run from the directory containing Classifier_codes
classifier_folder = 'Classifier_codes'
if not os.path.isdir(classifier_folder):
    print(f"Error: Folder '{classifier_folder}' not found in the current directory.")
    print("Please ensure the classifier code files are inside this folder.")
    sys.exit(1)
sys.path.append(os.path.abspath(classifier_folder))

# --- Import your custom classifier classes ---
try:
    # (Make sure the file names match your actual files)
    from KNN import KNN
    from LogisticRegression import LogisticRegression
    from DecisionTree import DecisionTreeClassifier # Assumes Node is in here too
    from RandomForest import RandomForestClassifier # Assumes it imports DT correctly
    from SVM import LinearSVM
    from GaussianNaiveBayes import GaussianNaiveBayes
    from MultiLayerPerceptron import SimpleMLP
    print("Successfully imported custom classifiers.")
except ImportError as e:
    print(f"Error importing classifier classes: {e}")
    print("Please check file names and ensure all necessary classes (like Node) are defined correctly.")
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     sys.exit(1)

# --- 1. Load and Prepare Iris Dataset ---
print("\n--- Loading and Preparing Iris Dataset ---")
iris = load_iris()
X = iris.data
y = iris.target # Target labels are already 0, 1, 2

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Classes: {np.unique(y)} ({iris.target_names})")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")

# Scale features (important for SVM, Logistic Regression, KNN, MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")

# --- Helper for One-vs-Rest (OvR) ---
def train_ovr(X_train_data, y_train_data, binary_classifier_class, **kwargs):
    """Trains multiple binary classifiers for One-vs-Rest."""
    models = {}
    classes = np.unique(y_train_data)
    for cls in classes:
        # Create binary labels: 1 for current class, 0 for others
        y_train_binary = np.where(y_train_data == cls, 1, 0)
        print(f"  Training OvR model for class {cls}...")
        model = binary_classifier_class(**kwargs)
        model.fit(X_train_data, y_train_binary)
        models[cls] = model
    return models

def predict_ovr(X_test_data, models):
    """Makes predictions using One-vs-Rest models."""
    # Get scores/probabilities from each binary model for each sample
    scores = np.zeros((X_test_data.shape[0], len(models)))
    classes = sorted(models.keys()) # Ensure consistent order

    for i, cls in enumerate(classes):
        model = models[cls]
        # Use predict_proba or decision_function if available, otherwise just predict
        if hasattr(model, 'predict_proba'):
            # Assumes predict_proba returns probability of class 1
            scores[:, i] = model.predict_proba(X_test_data) # Might need [:, 1] if shape is (n, 2)
            # Handle potential shape issues if predict_proba gives (n,) or (n, 2)
            if scores[:, i].ndim > 1 and scores.shape[1] > 1 :
                 # Assuming the second column is probability of class '1'
                 scores[:, i] = scores[:, i][:, 1]
            elif scores[:, i].ndim == 0: # Handle scalar case if only one prediction
                 scores = scores.reshape(-1, len(models)) # Reshape if needed
                 # This case might indicate an issue, check predict_proba output shape


        elif hasattr(model, 'decision_function'): # For SVM
            scores[:, i] = model.decision_function(X_test_data)
        else:
             # Fallback if only predict is available (less ideal for OvR)
             # This means we take the binary 0/1 prediction as the 'score'
             print(f"Warning: Using binary predict output for OvR scoring (model for class {cls}). Probabilities preferred.")
             scores[:, i] = model.predict(X_test_data)

    # Predict the class with the highest score/probability
    predictions = np.array([classes[i] for i in np.argmax(scores, axis=1)])
    return predictions


# --- 2. Train and Evaluate Classifiers ---
results = {}

print("\n--- Training and Evaluating Models ---")

# --- KNN ---
print("\n1. K-Nearest Neighbors (KNN)")
knn = KNN(k=5)
knn.fit(X_train_scaled, y_train) # Use scaled data for distance-based models
y_pred_knn = knn.predict(X_test_scaled)
results['KNN'] = sklearn_accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {results['KNN']:.4f}")

# --- Logistic Regression (using OvR) ---
print("\n2. Logistic Regression (One-vs-Rest)")
# Need predict_proba for OvR, make sure your class has it
if hasattr(LogisticRegression, 'predict_proba'):
    lr_models_ovr = train_ovr(X_train_scaled, y_train, LogisticRegression,
                              learning_rate=0.1, n_iters=1000, verbose=False)
    y_pred_lr_ovr = predict_ovr(X_test_scaled, lr_models_ovr)
    results['Logistic Regression (OvR)'] = sklearn_accuracy_score(y_test, y_pred_lr_ovr)
    print(f"Logistic Regression (OvR) Accuracy: {results['Logistic Regression (OvR)']:.4f}")
else:
    print("  Skipping Logistic Regression (OvR): predict_proba method not found.")
    results['Logistic Regression (OvR)'] = 'Skipped'


# --- Decision Tree ---
print("\n3. Decision Tree")
# Trees are less sensitive to scaling, can use original or scaled
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=3)
dt.fit(X_train, y_train) # Using unscaled for variety
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = sklearn_accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {results['Decision Tree']:.4f}")

# --- Random Forest ---
print("\n4. Random Forest")
# As with DT, scaling isn't strictly necessary but doesn't hurt
rf = RandomForestClassifier(n_trees=10, max_depth=10, min_samples_split=3, n_feats=None) # n_feats=None -> sqrt(features)
rf.fit(X_train, y_train) # Using unscaled
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = sklearn_accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {results['Random Forest']:.4f}")

# --- Linear SVM (using OvR) ---
print("\n5. Linear SVM (One-vs-Rest)")
# SVM needs scaled data and ideally a decision_function or similar score for OvR
# Let's add a simple decision_function if it doesn't exist
if not hasattr(LinearSVM, 'decision_function'):
    print("  Adding basic decision_function to LinearSVM for OvR.")
    def decision_function(self, X):
        X = np.array(X)
        if self.weights is None or self.bias is None:
            raise RuntimeError("SVM Model has not been fitted yet.")
        if X.shape[1] != self.weights.shape[0]:
             raise ValueError(f"Input feature dimension mismatch in decision_function: Expected {self.weights.shape[0]}, got {X.shape[1]}")
        return np.dot(X, self.weights) - self.bias
    LinearSVM.decision_function = decision_function # Monkey-patching

svm_models_ovr = train_ovr(X_train_scaled, y_train, LinearSVM,
                           learning_rate=0.01, lambda_param=0.01, n_iters=1000, verbose=False)
y_pred_svm_ovr = predict_ovr(X_test_scaled, svm_models_ovr)
results['Linear SVM (OvR)'] = sklearn_accuracy_score(y_test, y_pred_svm_ovr)
print(f"Linear SVM (OvR) Accuracy: {results['Linear SVM (OvR)']:.4f}")


# --- Gaussian Naive Bayes ---
print("\n6. Gaussian Naive Bayes")
# GNB doesn't strictly require scaling, but can sometimes benefit if features have vastly different ranges
gnb = GaussianNaiveBayes() # Or GaussianNaiveBayes if you fixed the typo
gnb.fit(X_train, y_train) # Using unscaled
y_pred_gnb = gnb.predict(X_test)
results['Gaussian Naive Bayes'] = sklearn_accuracy_score(y_test, y_pred_gnb)
print(f"Gaussian Naive Bayes Accuracy: {results['Gaussian Naive Bayes']:.4f}")

# --- Multi-Layer Perceptron (using OvR) ---
print("\n7. Simple MLP (One-vs-Rest)")
# MLP requires scaled data and predict_proba for OvR
n_features = X_train_scaled.shape[1]
if hasattr(SimpleMLP, 'predict_proba'):
    mlp_models_ovr = train_ovr(X_train_scaled, y_train, SimpleMLP,
                               n_input=n_features, n_hidden=8, n_output=1, # n_output=1 for binary sigmoid
                               learning_rate=0.1, n_iters=1000, verbose=False, activation='relu') # Try ReLU
    y_pred_mlp_ovr = predict_ovr(X_test_scaled, mlp_models_ovr)
    results['Simple MLP (OvR)'] = sklearn_accuracy_score(y_test, y_pred_mlp_ovr)
    print(f"Simple MLP (OvR) Accuracy: {results['Simple MLP (OvR)']:.4f}")
else:
     print("  Skipping MLP (OvR): predict_proba method not found.")
     results['Simple MLP (OvR)'] = 'Skipped'


# --- 3. Display Results ---
print("\n--- Final Accuracy Summary ---")
for model_name, accuracy in results.items():
    if isinstance(accuracy, str): # Handle skipped models
        print(f"{model_name}: {accuracy}")
    else:
        print(f"{model_name}: {accuracy:.4f}")