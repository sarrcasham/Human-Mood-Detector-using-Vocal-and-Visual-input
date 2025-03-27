import sounddevice as sd
import numpy as np
import whisper

# Load Whisper model (use 'base' for limited computational resources)
model = whisper.load_model("base")

# Audio settings
samplerate = 16000  # Whisper's required sample rate
duration = 5        # Duration of audio chunks in seconds

def record_audio(duration, samplerate):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    audio = np.squeeze(audio)
    return audio

def transcribe(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

print("Real-time transcription with Whisper (Press Ctrl+C to stop):")

try:
    while True:
        audio_chunk = record_audio(duration, samplerate)
        transcript = transcribe(audio_chunk)
        print(f"Transcript: {transcript}")

except KeyboardInterrupt:
    print("\nTranscription stopped.")
