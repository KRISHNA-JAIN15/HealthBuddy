import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- [Step 0: Import Your Custom Model] ---
# This script assumes 'RandomForest.py' is in the same folder.
try:
    from RandomForest import RandomForestRegressor
    print("✅ Successfully imported custom RandomForestRegressor from RandomForest.py")
except ImportError:
    print("❌ ERROR: Could not import RandomForestRegressor from RandomForest.py")
    print("Please make sure the file 'RandomForest.py' exists and contains your class.")
    # In a real script, you might exit here
    # exit()
except Exception as e:
    print(f"❌ An error occurred during import: {e}")
    # exit()

print("\n--- [Step 1: Load Data] ---")

try:
    data = pd.read_csv('bodyfat.csv')
    print(f"✅ Successfully loaded 'bodyfat.csv' with {len(data)} rows.")
except FileNotFoundError:
    print("❌ Error: 'bodyfat.csv' not found.")
    print("Please make sure the file is in the same directory.")
    exit() # Stop the script
except Exception as e:
    print(f"❌ An error occurred loading the CSV: {e}")
    exit()

print("\n--- [Step 2: Preprocessing] ---")

try:
    # Define Features (X) and Target (y)
    target_column = 'BodyFat'
    X = data.drop(columns=[target_column, 'Density']) 
    y = data[target_column]
    
    # Convert y to a 1D NumPy array (good practice for custom models)
    y_np = y.values.ravel()

    # Split the Data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_np, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Scale the Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled successfully.")

except KeyError as e:
    print(f"\n❌ Error: A required column is missing from the CSV. {e}")
    exit()
except Exception as e:
    print(f"\n❌ An unexpected error occurred during preprocessing: {e}")
    exit()


print("\n--- [Step 3: Train & Evaluate Your Model on Test Set] ---")
print("This step trains on the 80% split to check performance.")

# The best parameters you found from hyperparameter tuning
best_params = {
    'n_trees': 100, 
    'max_depth': 10, 
    'min_samples_split': 5,
    'random_state': 42
}

print(f"Using parameters: {best_params}")

try:
    # 1. Instantiate your model
    test_model = RandomForestRegressor(**best_params)

    # 2. Train on the training set
    start_time = time.time()
    test_model.fit(X_train_scaled, y_train)
    duration = time.time() - start_time
    print(f"✅ Model trained on 80% of data in {duration:.2f}s.")

    # 3. Evaluate on the test set
    y_pred = test_model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n--- Test Set Performance ---")
    print(f"  R-squared: {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f}")

except NameError:
    print("\n❌ ERROR: 'RandomForestRegressor' class not found.")
    print("Import in Step 0 failed. Cannot proceed.")
    exit()
except Exception as e:
    print(f"\n❌ An error occurred during model training or evaluation: {e}")
    exit()


print("\n--- [Step 4: Retrain Final Model on ALL Data] ---")
print("Retraining the model on 100% of the data for production use.")

# 1. Create and fit a new scaler on the *entire* X dataset
final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X)
print("Created and fitted a new scaler on the *entire* dataset.")

# 2. Instantiate a new, final model
final_model = RandomForestRegressor(**best_params)

# 3. Train the final model on the *entire* scaled dataset (X_full_scaled and y_np)
start_time = time.time()
final_model.fit(X_full_scaled, y_np)
duration = time.time() - start_time
print(f"✅ Final model trained on all data in {duration:.2f}s.")


print("\n--- [Step 5: Save Final Model and Scaler] ---")

# We must save BOTH the model and the final_scaler
# to make correct predictions on new data later.
model_filename = 'final_custom_rf_model.pkl'

model_payload = {
    'model': final_model,
    'scaler': final_scaler
}

try:
    with open(model_filename, 'wb') as file:
        pickle.dump(model_payload, file)
        
    print(f"✅ Model and scaler successfully saved to '{model_filename}'")
    
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("\n--- Script Complete ---")