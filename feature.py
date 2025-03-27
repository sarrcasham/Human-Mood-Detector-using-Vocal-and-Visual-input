import numpy as np
import pandas as pd
import librosa

# Load your dataset CSV file
dataset_path = r"C:\Users\Asarv\Desktop\Dl project\dataset\speech_dataset.csv"
df = pd.read_csv(dataset_path)

# Initialize list to store features
features = []

# Iterate through each audio file in the dataset
for index, row in df.iterrows():
    filepath = row["Filepath"]
    try:
        # Load audio file
        y, sr = librosa.load(filepath, sr=None)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        # Append processed features to list
        features.append(mfccs_processed)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

# Check if features were extracted successfully
if len(features) == 0:
    raise ValueError("Feature extraction failed. Ensure audio files are valid.")

# Convert features list to NumPy array and save as .npy file
X_features = np.array(features)
np.save("X_features.npy", X_features)

print("Feature extraction complete. Saved to X_features.npy.")

