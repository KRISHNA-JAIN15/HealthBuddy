import pickle
import numpy as np
import pandas as pd

# --- [Step 1: Load the Model and Scaler] ---
model_filename = 'final_custom_rf_model.pkl'

try:
    with open(model_filename, 'rb') as file:
        # Load the dictionary saved earlier
        model_payload = pickle.load(file)
    
    # Separate the model and the scaler
    loaded_model = model_payload['model']
    loaded_scaler = model_payload['scaler']
    
    print(f"✅ Successfully loaded model and scaler from '{model_filename}'")
    
except FileNotFoundError:
    print(f"❌ ERROR: The file '{model_filename}' was not found.")
    print("Please make sure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading the file: {e}")
    exit()


# --- [Step 2: Define New Data for Prediction] ---

# NOTE: The input data MUST have the same 13 features in the same order
# that the model was trained on.
# Features: ['Age', 'Weight', 'Height', 'Neck', 'Chest', 
#            'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 
#            'Biceps', 'Forearm', 'Wrist']

# Let's create two new example data points
new_data_dict = {
    'Sample 1': [
        30,     # Age
        160.0,  # Weight
        70.0,   # Height
        37.0,   # Neck
        95.0,   # Chest
        88.0,   # Abdomen
        96.0,   # Hip
        58.0,   # Thigh
        38.0,   # Knee
        22.0,   # Ankle
        33.0,   # Biceps
        29.0,   # Forearm
        18.0    # Wrist
    ],
    'Sample 2': [
        45,     # Age
        185.0,  # Weight
        68.0,   # Height
        40.0,   # Neck
        105.0,  # Chest
        100.0,  # Abdomen
        104.0,  # Hip
        62.0,   # Thigh
        40.0,   # Knee
        24.0,   # Ankle
        35.0,   # Biceps
        31.0,   # Forearm
        19.0    # Wrist
    ]
}

# Get the feature names in the correct order (from the scaler)
try:
    feature_names = loaded_scaler.feature_names_in_
except AttributeError:
    # If the scaler was on a numpy array, it won't have feature_names_in_
    # We must define them manually in the correct order.
    # This is the order from your CSV (minus BodyFat and Density)
    feature_names = ['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 
                     'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']
    print("\nWarning: Scaler did not store feature names. Assuming manual order.")


# Convert the dictionary data to a pandas DataFrame
# This ensures the columns are in the correct order for the scaler
new_data_df = pd.DataFrame.from_dict(new_data_dict, orient='index', columns=feature_names)

print("\n--- New Data to Predict ---")
print(new_data_df)


# --- [Step 3: Scale and Predict] ---
try:
    # 1. Scale the new data using the *loaded scaler*
    #    (The scaler expects a 2D array, which our DataFrame provides)
    new_data_scaled = loaded_scaler.transform(new_data_df)
    print("\n✅ New data scaled successfully.")

    # 2. Make predictions using the *loaded model*
    predictions = loaded_model.predict(new_data_scaled)
    print("✅ Predictions made.")

    # --- [Step 4: Show Results] ---
    print("\n--- Predicted BodyFat % ---")
    
    for (sample_name, prediction) in zip(new_data_df.index, predictions):
        print(f"  {sample_name}: {prediction:.2f} %")

except Exception as e:
    print(f"\n❌ An error occurred during scaling or prediction: {e}")
    print("This might happen if the new data's columns don't match the training data.")