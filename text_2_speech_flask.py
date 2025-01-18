from flask import Flask, request, jsonify
from models import build_model
import torch
import numpy as np
from onnxruntime import InferenceSession
import soundfile as sf
from kokoro import phonemize, tokenize
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths
model_path = 'kokoro-v0_19.pth'
onnx_path = 'kokoro-v0_19.onnx'
voice_name = 'am_adam'  # Voice: Adam
voicepack_path = f'voices/{voice_name}.pt'

# Load model once during startup
try:
    logger.info("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug(f"Using device: {device}")
    model = build_model(model_path, device)
    logger.info(f"Model loaded successfully from {model_path}")

    logger.info("Loading voicepack...")
    voicepack = torch.load(voicepack_path)
    logger.info(f"Voicepack loaded successfully from {voicepack_path}")

    logger.info("Loading ONNX session...")
    sess = InferenceSession(onnx_path)
    logger.info(f"ONNX session loaded successfully from {onnx_path}")
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
    raise

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    try:
        # Log the incoming request
        logger.debug("Received request to /generate_audio")
        data = request.json
        text = data['text']
        output_path = data.get('output', 'output_audio.wav')
        logger.debug(f"Text: {text}")
        logger.debug(f"Output path: {output_path}")

        # Tokenize text
        logger.debug("Tokenizing text...")
        tokens = tokenize(phonemize(text, 'a'))
        logger.debug(f"Initial tokens: {tokens}")
        if len(tokens) > 510:
            logger.warning(f"Text too long; truncating to 510 tokens.")
            tokens = tokens[:510]
        tokens = [[0, *tokens, 0]]
        logger.debug(f"Final tokens: {tokens}")

        # Get voicepack style
        logger.debug("Fetching voicepack style...")
        style = voicepack[len(tokens[0]) - 2].numpy()
        logger.debug(f"Voicepack style shape: {style.shape}")

        # Run ONNX inference
        logger.debug("Running ONNX inference...")
        audio = sess.run(None, dict(
            tokens=tokens,
            style=style,
            speed=np.ones(1, dtype=np.float32)
        ))[0]
        logger.debug(f"Audio generated with shape: {audio.shape}")

        # Save audio
        logger.debug(f"Saving audio to {output_path}...")
        sf.write(output_path, audio, 24000)
        logger.info(f"Audio saved successfully to {output_path}")
        return jsonify({"status": "success", "output_path": output_path})
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)

