import sounddevice as sd
import numpy as np
import whisper
import noisereduce as nr
import librosa
import cv2
from deepface import DeepFace
import mediapipe as mp
import pandas as pd

# Load Whisper model for speech-to-text transcription
model_whisper = whisper.load_model("base")

# MediaPipe setup for gesture detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load your speech dataset CSV file (adjust path as needed)
dataset_path = r"C:\Users\Asarv\Desktop\Dl project\dataset\speech_dataset.csv"
speech_df = pd.read_csv(dataset_path)
print("Dataset loaded successfully!")
print(speech_df.head())

# Audio settings
samplerate = 16000  # Whisper's required sample rate
duration = 5        # Duration of audio chunks in seconds

# Record audio and reduce noise
def record_audio(duration, samplerate):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    audio = np.squeeze(audio)
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=samplerate)
    return reduced_noise_audio

# Whisper transcription function
def transcribe(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model_whisper, mel, options)
    return result.text

# Extract MFCC features (for future LSTM model integration)
def extract_audio_features(audio, sr):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        return None

# Real-time facial emotion detection using webcam
def detect_facial_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion, frame  # Return emotion and frame for gesture detection
    else:
        return "No face detected", None

# Real-time hand gesture detection using MediaPipe Hands
def detect_hand_gestures(frame):
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    gestures_detected = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gestures_detected.append("Hand detected")
    
    hands.close()
    return gestures_detected

print("\n🎙️ Real-time Emotion-Aware Voice Assistant Started (Press Ctrl+C to stop):")

try:
    transcript_list = []  # Store transcripts and emotions
    
    while True:
        # Audio processing pipeline
        audio_chunk = record_audio(duration, samplerate)
        transcript = transcribe(audio_chunk)
        audio_features = extract_audio_features(audio_chunk, samplerate)

        # TODO: Replace this placeholder with actual LSTM model prediction later.
        predicted_audio_emotion = "neutral"  # Placeholder

        # Visual processing pipeline (webcam-based facial emotion detection and gesture detection)
        predicted_visual_emotion, frame = detect_facial_emotion()
        
        if frame is not None:
            gestures_detected = detect_hand_gestures(frame)  # Detect gestures from webcam feed
        
        # Simple fusion logic for combining emotions from voice and face:
        if predicted_audio_emotion == predicted_visual_emotion:
            combined_emotion = predicted_audio_emotion
        else:
            combined_emotion = f"{predicted_audio_emotion}/{predicted_visual_emotion}"

        # Append results to transcript list for final output after stopping recording
        transcript_list.append({"Transcript": transcript, "Emotion": combined_emotion})

        # Output results clearly:
        print(f"\n📝 Transcript: {transcript}")
        print(f"🔊 Audio Emotion: {predicted_audio_emotion}")
        print(f"📷 Visual Emotion: {predicted_visual_emotion}")
        
        if frame is not None and gestures_detected:
            print(f"✋ Gestures Detected: {gestures_detected}")
        
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    
    # Final output of all transcripts with emotions after stopping recording
    print("\n📜 Full Transcript:")
    for entry in transcript_list:
        print(f"{entry['Transcript']} - Emotion: {entry['Emotion']}")
