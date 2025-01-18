from flask import Flask, request, jsonify
from models import build_model
import torch
import numpy as np
from onnxruntime import InferenceSession
import soundfile as sf
from kokoro import phonemize, tokenize
import os

# Initialize Flask app
app = Flask(__name__)

# Configure paths
model_path = 'kokoro-v0_19.pth'
onnx_path = 'kokoro-v0_19.onnx'
voice_name = 'am_adam'
voicepack_path = f'voices/{voice_name}.pt'
output_dir = 'output_audio'  # Directory to save generated audio
os.makedirs(output_dir, exist_ok=True)

# Load model and voicepack
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(model_path, device)
voicepack = torch.load(voicepack_path)

# Function to generate audio
def generate_audio(text, output_path):
    try:
        # Tokenize text
        tokens = tokenize(phonemize(text, 'a'))
        if len(tokens) > 510:
            tokens = tokens[:510]  # Truncate tokens if too long
            print("Warning: Input text truncated to fit token limit (510 tokens).")
        
        # Add padding
        tokens = [[0, *tokens, 0]]
        
        # Load voicepack embedding
        style_vector = voicepack[len(tokens[0]) - 2].numpy()  # Adjust for padding

        # Run ONNX inference
        sess = InferenceSession(onnx_path)
        audio = sess.run(None, dict(
            tokens=tokens,
            style=style_vector,
            speed=np.ones(1, dtype=np.float32)
        ))[0]

        # Save audio
        sf.write(output_path, audio, 24000)
        return True, None
    except Exception as e:
        return False, str(e)

# Define API route
@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from request
        data = request.json
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Generate audio
        output_file = os.path.join(output_dir, f"output_audio_{voice_name}.wav")
        success, error = generate_audio(text, output_file)

        if success:
            return jsonify({"message": f"Audio saved to {output_file}", "output_path": output_file}), 200
        else:
            return jsonify({"error": error}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

