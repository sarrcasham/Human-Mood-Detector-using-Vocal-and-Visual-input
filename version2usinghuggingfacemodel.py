import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load pre-trained Hugging Face model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

# Audio settings
samplerate = 16000  # Required sample rate for Wav2Vec2
duration = 5  # Recording duration in seconds per transcription chunk

def transcribe(audio):
    inputs = tokenizer(audio, return_tensors='pt', padding='longest').input_values
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription.lower()

print("Real-time Voice-to-Text Transcription (Press Ctrl+C to stop):")

try:
    while True:
        print("\nRecording...")
        audio_chunk = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        audio_chunk = np.squeeze(audio_chunk)
        print("Transcribing...")
        text_output = transcribe(audio_chunk)
        print(f"Transcript: {text_output}")

except KeyboardInterrupt:
    print("\nStopped transcription.")
