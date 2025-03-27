import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load dataset CSV file
dataset_path = r"C:\Users\Asarv\Desktop\Dl project\dataset\speech_dataset.csv"
df = pd.read_csv(dataset_path)

# Load precomputed features from .npy file
X = np.load("X_features.npy")

# Reshape X if necessary (add time dimension for LSTM input)
if len(X.shape) == 2:  # If X has shape (num_samples, features)
    X = np.expand_dims(X, axis=1)  # Reshape to (num_samples, timesteps=1, features)

print("Shape of X after reshaping:", X.shape)

# Extract labels and encode them
labels = df["Emotion"].values

if len(labels) == 0:
    raise ValueError("Labels array is empty. Check dataset loading.")

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
y = to_categorical(labels_encoded)

# Define LSTM model structure
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))  # Adjust input shape dynamically
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model on extracted features and labels
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# Save trained model after successful training
model.save("lstm_emotion_model.h5")
print("Model training complete. Saved as lstm_emotion_model.h5.")
