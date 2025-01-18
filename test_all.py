from models import build_model
import torch
import numpy as np
from onnxruntime import InferenceSession
import soundfile as sf
from kokoro import phonemize, tokenize

# Configure paths
model_path = 'kokoro-v0_19.pth'
onnx_path = 'kokoro-v0_19.onnx'
voice_dir = 'voices'

# List of available voices
VOICE_NAME = [
    'af', 'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky'
]

# Input text
text = input("Enter the text to generate audio for all voices: ")

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(model_path, device)

# Tokenize text
tokens = tokenize(phonemize(text, 'a'))
if len(tokens) > 510:
    tokens = tokens[:510]  # Truncate tokens if too long
    print("Warning: Input text truncated to fit token limit (510 tokens).")

# Add padding to tokens
tokens = [[0, *tokens, 0]]

# Loop through all voices and generate audio
for voice_name in VOICE_NAME:
    try:
        # Load voicepack
        voicepack_path = f"{voice_dir}/{voice_name}.pt"
        print(f"Using voice: {voice_name}")
        voicepack = torch.load(voicepack_path)[len(tokens[0]) - 2].numpy()  # Adjust for padding

        # Run ONNX inference
        sess = InferenceSession(onnx_path)
        audio = sess.run(None, dict(
            tokens=tokens,
            style=voicepack,
            speed=np.ones(1, dtype=np.float32)
        ))[0]

        # Save audio
        output_path = f"output_audio_{voice_name}.wav"
        sf.write(output_path, audio, 24000)
        print(f"Audio saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate audio for {voice_name}: {e}")

print("Audio generation completed for all voices.")

