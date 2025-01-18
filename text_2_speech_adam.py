from models import build_model
import torch
import numpy as np
from onnxruntime import InferenceSession
import soundfile as sf
from kokoro import phonemize, tokenize

# Configure paths
model_path = 'kokoro-v0_19.pth'
onnx_path = 'kokoro-v0_19.onnx'
voice_name = 'am_adam'  # Voice: Adam
voicepack_path = f'voices/{voice_name}.pt'

# Input text
text = input("Enter the text to generate audio for Adam: ")

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

# Load Adam's voicepack
try:
    voicepack = torch.load(voicepack_path)[len(tokens[0]) - 2].numpy()  # Adjust for padding
except Exception as e:
    print(f"Failed to load voicepack for {voice_name}: {e}")
    exit()

# Run ONNX inference
try:
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
    print(f"Failed to generate audio: {e}")

