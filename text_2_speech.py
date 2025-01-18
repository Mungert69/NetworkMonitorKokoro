import argparse
from models import build_model
import torch
import numpy as np
from onnxruntime import InferenceSession
import soundfile as sf
from kokoro import phonemize, tokenize

# Parse arguments
parser = argparse.ArgumentParser(description="Generate speech from text")
parser.add_argument("--text", required=True, help="Text to convert to speech")
parser.add_argument("--output", required=True, help="Output file path")
args = parser.parse_args()

# Paths
model_path = 'kokoro-v0_19.pth'
onnx_path = 'kokoro-v0_19.onnx'
voice_name = 'am_adam'  # Voice: Adam
voicepack_path = f'voices/{voice_name}.pt'

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(model_path, device)

# Tokenize text
tokens = tokenize(phonemize(args.text, 'a'))
if len(tokens) > 510:
    tokens = tokens[:510]
tokens = [[0, *tokens, 0]]

# Load voicepack
voicepack = torch.load(voicepack_path)[len(tokens[0]) - 2].numpy()

# Run ONNX inference
sess = InferenceSession(onnx_path)
audio = sess.run(None, dict(
    tokens=tokens,
    style=voicepack,
    speed=np.ones(1, dtype=np.float32)
))[0]

# Save audio
sf.write(args.output, audio, 24000)
print(f"Audio saved to {args.output}")

