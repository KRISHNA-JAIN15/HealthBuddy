import pickle
import numpy as np
import pandas as pd
import sys
import os

# --- [Step 1: Load the Model and Pipeline Components] ---
PIPELINE_FILENAME = 'rf_eeg_pipeline.pkl'

try:
    with open(PIPELINE_FILENAME, 'rb') as file:
        pipeline = pickle.load(file)
    
    loaded_model = pipeline['model']
    loaded_le = pipeline['label_encoder']
    loaded_imputer = pipeline['imputer']
    feature_names = pipeline['feature_names'] # This is crucial
    class_names = pipeline['class_names']
    
    print(f"✅ Successfully loaded pipeline from '{PIPELINE_FILENAME}'")
    print(f"   Model trained on {len(feature_names)} features.")
    
except FileNotFoundError:
    print(f"❌ ERROR: The file '{PIPELINE_FILENAME}' was not found.")
    print(f"   Make sure '{PIPELINE_FILENAME}' is in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An error occurred while loading the file: {e}")
    sys.exit(1)


# --- [Step 2: Get CSV File Path from Command Line] ---

# Check if a file path was provided
if len(sys.argv) != 2:
    print("\n--- Usage ---")
    print(f"python {sys.argv[0]} <path_to_your_features.csv>")
    print("\nExample:")
    print(f"python {sys.argv[0]} Subject_01_Session_1_features.csv")
    sys.exit(1)

csv_file_path = sys.argv[1]

if not os.path.exists(csv_file_path):
    print(f"❌ ERROR: File not found at '{csv_file_path}'")
    sys.exit(1)

print(f"\nLoading new data from '{csv_file_path}'...")

try:
    # Load the new data from the CSV
    new_data_df = pd.read_csv(csv_file_path)
    
    # Store sample identifiers (like a 'Subject' col or index)
    # This is just for printing the results nicely
    if 'Subject' in new_data_df.columns:
        sample_names = new_data_df['Subject']
    else:
        sample_names = new_data_df.index
        
    print(f"Loaded {len(new_data_df)} samples (epochs) for prediction.")

except Exception as e:
    print(f"❌ An error occurred while reading the CSV: {e}")
    sys.exit(1)


# --- [Step 3: Preprocess, Re-align, and Predict] ---
try:
    # **CRITICAL STEP: Align columns**
    # This ensures the DataFrame has the exact same columns in the
    # exact same order as the training data ('feature_names').
    # - It will drop any columns not in 'feature_names' (like 'Subject' or 'Session')
    # - It will add any missing columns from 'feature_names' and fill them with NaN
    data_for_processing = new_data_df.reindex(columns=feature_names, fill_value=np.nan)
    
    print(f"Data re-aligned to {len(feature_names)} features.")

    # 1. Apply the Imputer (if one was saved)
    #    This will fill any NaNs (including those from missing columns)
    if loaded_imputer:
        print("Applying loaded imputer...")
        data_to_predict = loaded_imputer.transform(data_for_processing)
    else:
        print("No imputer found. Using data as-is (and filling NaNs with 0).")
        data_to_predict = data_for_processing.fillna(0).values

    # 2. Make predictions
    predictions_idx = loaded_model.predict(data_to_predict)
    
    # 3. Decode predictions
    predictions_names = loaded_le.inverse_transform(predictions_idx)
    print("✅ Predictions made.")

    # --- [Step 4: Show Results] ---
    print("\n--- Prediction Results ---")
    
    results_df = pd.DataFrame({
        'Sample_Epoch_Number': new_data_df.index,
        'Predicted_Session': predictions_names
    })
    
    # Print the full list of predictions
    print(results_df.to_string(index=False))

    # Print a summary
    print("\n--- Summary ---")
    prediction_counts = results_df['Predicted_Session'].value_counts()
    print(prediction_counts)
    
    most_common_prediction = prediction_counts.idxmax()
    print(f"\nOverall Prediction for this file (most common): {most_common_prediction}")


except Exception as e:
    print(f"\n❌ An error occurred during processing or prediction: {e}")
    print("This might be due to a mismatch in expected data or column names.")


#python predict_eeg.py Subject_00_Session_1_features.csv