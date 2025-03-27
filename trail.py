# import pandas as pd

# dataset_path = r"C:\Users\Asarv\Desktop\Dl project\dataset\speech_dataset.csv"
# df = pd.read_csv(dataset_path)

# # print(df.head())
# # print("Dataset size:", len(df))

# print(df["Emotion"].unique())
# print(df["Emotion"].value_counts())

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical

# # Load your dataset CSV file into a DataFrame clearly:
# dataset_path = r"C:\Users\Asarv\Desktop\Dl project\dataset\speech_dataset.csv"
# df = pd.read_csv(dataset_path)

# # Verify dataset loaded correctly:
# print("Dataset loaded successfully!")
# print(df.head())

# # Extract labels from the 'Emotion' column:
# labels = df["Emotion"].values

# # Check if labels are loaded correctly:
# if len(labels) == 0:
#     raise ValueError("Labels array is empty. Check dataset loading.")

# # Encode labels:
# le = LabelEncoder()
# labels_encoded = le.fit_transform(labels)

# # Convert encoded labels to categorical (one-hot encoding):
# labels_categorical = to_categorical(labels_encoded)

# # Verify categorical labels shape:
# print("Categorical labels shape:", labels_categorical.shape)

# import numpy as np

# try:
#     X = np.load("X_features.npy")
#     print("Shape of X:", X.shape)
# except FileNotFoundError:
#     print("Error: 'X_features.npy' not found.")


import pandas as pd
import os

# Load the dataset
dataset_path = r"C:\Users\Asarv\Desktop\Dl project\dataset\speech_dataset.csv"
df = pd.read_csv(dataset_path)

# Check if all files exist
missing_files = []
for filepath in df["Filepath"]:
    if not os.path.exists(filepath):
        missing_files.append(filepath)

if missing_files:
    print("Missing audio files:", missing_files)
else:
    print("All audio files are accessible.")


# import numpy as np

# X = np.load("X_features.npy")
# print("Shape of X:", X.shape)
