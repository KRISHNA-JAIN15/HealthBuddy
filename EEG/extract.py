import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import sys
import os
import warnings

# Suppress MNE info messages
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. Configuration ---

# ❗ **ACTION REQUIRED**: Update this to the name of your EDF file
EDF_FILE_PATH = r"C:\Users\Research PC\Desktop\ML\EEG\Subject00_1.edf" 

# ❗ **ACTION REQUIRED**: Update these to match your data
SUBJECT_ID = "Subject_00"
SESSION_ID = "Session_1" # e.g., 'Session 1', 'Session 2', etc.

# --- 2. Feature Extraction Parameters ---
EPOCH_DURATION_SEC = 2  # How long each "sample" or "window" should be
EEG_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45)  # Added Gamma
}

# --- 3. Feature Extraction Functions ---

def get_band_power(data, sfreq, band):
    """Calculate the average power in a specific frequency band."""
    band_low, band_high = band
    # Welch's method for power spectral density
    freqs, psd = welch(data, sfreq, nperseg=sfreq*EPOCH_DURATION_SEC)
    
    # Find frequencies within the band
    idx_band = np.logical_and(freqs >= band_low, freqs <= band_high)
    
    # Calculate average power in the band (using Simpson's rule for integration)
    avg_power = np.trapz(psd[idx_band], freqs[idx_band])
    return avg_power

def extract_features_from_epoch(epoch_data, sfreq, channel_names):
    """
    Extracts features from a single epoch (window) of data.
    Input: epoch_data (channels, timepoints)
    """
    features = {}
    for i, ch_name in enumerate(channel_names):
        ch_data = epoch_data[i, :]
        
        # 1. Time-domain (Statistical) features
        features[f'{ch_name}_mean'] = np.mean(ch_data)
        features[f'{ch_name}_var'] = np.var(ch_data)
        features[f'{ch_name}_skew'] = skew(ch_data)
        features[f'{ch_name}_kurt'] = kurtosis(ch_data)
        
        # 2. Frequency-domain (Band Power) features
        for band_name, band_freqs in EEG_BANDS.items():
            power = get_band_power(ch_data, sfreq, band_freqs)
            features[f'{ch_name}_{band_name}'] = power
            
    return features

# --- 4. Main Execution ---

def process_edf_file(file_path, subject_id, session_id):
    print(f"Processing {file_path} for {subject_id}, {session_id}...")
    
    try:
        # Load the EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Select only EEG channels (optional, but good practice)
        raw.pick_types(eeg=True)
        
        # Set a standard montage if missing (helps with channel names)
        # 'standard_1020' is common. Use 'standard_1005' for more channels.
        try:
            raw.set_montage('standard_1020', on_missing='ignore')
        except ValueError:
            print("Could not set montage. Using raw channel names.")

        channel_names = raw.ch_names
        sfreq = raw.info['sfreq']
        print(f"Found {len(channel_names)} channels. Fs = {sfreq} Hz.")
        
        # Create fixed-length epochs (windows)
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION_SEC, 
                                            overlap=0, preload=True)
        
        all_epoch_features = []
        
        # Iterate over each epoch and extract features
        for epoch_data in epochs.get_data():
            # epoch_data shape is (n_channels, n_times)
            features = extract_features_from_epoch(epoch_data, sfreq, channel_names)
            all_epoch_features.append(features)
        
        if not all_epoch_features:
            print("Error: No epochs were created. Check file duration and epoch size.")
            return

        # Convert list of feature dicts to a DataFrame
        features_df = pd.DataFrame(all_epoch_features)
        
        # Add the Subject and Session columns for your model
        features_df['Subject'] = subject_id
        features_df['Session'] = session_id
        
        # Re-order columns to put Subject and Session first
        cols_to_move = ['Subject', 'Session']
        features_df = features_df[cols_to_move + 
                                  [c for c in features_df.columns if c not in cols_to_move]]
        
        print(f"Successfully extracted {len(features_df)} samples (epochs).")
        return features_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(EDF_FILE_PATH):
        print(f"--- 🛑 ERROR ---")
        print(f"File not found: {EDF_FILE_PATH}")
        print("Please update the 'EDF_FILE_PATH' variable in this script.")
        sys.exit(1)
        
    # Process the file
    subject_features_df = process_edf_file(EDF_FILE_PATH, SUBJECT_ID, SESSION_ID)
    
    if subject_features_df is not None:
        # Save to a new CSV file
        output_csv = f"{SUBJECT_ID}_{SESSION_ID}_features.csv"
        subject_features_df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Success! Features saved to '{output_csv}'")
        print("\nDataFrame Head:")
        print(subject_features_df.head())