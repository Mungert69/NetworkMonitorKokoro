from flask import Flask, request, jsonify
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import numpy as np
from onnxruntime import InferenceSession
import soundfile as sf
import os
import sys
import uuid
import logging
from flask_cors import CORS
import threading
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global lock to ensure one method runs at a time
global_lock = threading.Lock()

# Repository ID and paths
kokoro_model_hexgrad_id = 'hexgrad/Kokoro-82M'
model_path_hexgrad = 'kokoro_model_hexgrad'
voice_name = 'am_adam'  # Voice: Adam

# Download and initialize models
def initialize_models():
    global sess, voicepack, processor, whisper_model

    try:
        # Download Hexgrad model if not already downloaded
        if not os.path.exists(model_path_hexgrad):
            logger.info("Downloading and loading Hexgrad model...")
            kokoro_dir_hexgrad = snapshot_download(kokoro_model_hexgrad_id, cache_dir=model_path_hexgrad)
            logger.info(f"Hexgrad model directory: {kokoro_dir_hexgrad}")
        else:
            kokoro_dir_hexgrad = model_path_hexgrad
            logger.info(f"Using cached Hexgrad model directory: {kokoro_dir_hexgrad}")

        # Validate ONNX file path
        onnx_path = None
        for root, _, files in os.walk(kokoro_dir_hexgrad):
            if 'kokoro-v0_19.onnx' in files:
                onnx_path = os.path.join(root, 'kokoro-v0_19.onnx')
                break

        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found after redownload at {kokoro_dir_hexgrad}")

        logger.info("Loading ONNX session...")
        sess = InferenceSession(onnx_path)
        logger.info(f"ONNX session loaded successfully from {onnx_path}")

        # Add Hexgrad directory containing models.py to Python path
        models_path = None
        for root, _, files in os.walk(kokoro_dir_hexgrad):
            if 'models.py' in files:
                models_path = root
                break

        if not models_path:
            raise FileNotFoundError(f"models.py not found in {kokoro_dir_hexgrad}")
        sys.path.append(models_path)
        logger.info(f"Added models directory to Python path: {models_path}")

        from models import build_model  # Import after adding path
        logger.info("Hexgrad model loaded successfully")

        # Load voicepack
        logger.info("Loading voicepack...")
        voicepack_path = os.path.join(models_path, 'voices', f'{voice_name}.pt')
        if not os.path.exists(voicepack_path):
            raise FileNotFoundError(f"Voicepack file not found at {voicepack_path}")
        voicepack = torch.load(voicepack_path)
        logger.info(f"Voicepack loaded successfully from {voicepack_path}")

        # Initialize Whisper model for S2T
        logger.info("Downloading and loading Whisper model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        whisper_model.config.forced_decoder_ids = None
        logger.info("Whisper model loaded successfully")

    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise


# Initialize models
initialize_models()

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    """Text-to-Speech (T2S) Endpoint"""
    with global_lock:  # Acquire global lock to ensure only one instance runs
        try:
            logger.debug("Received request to /generate_audio")
            data = request.json
            text = data['text']
            output_path = data.get('output', 'output_audio.wav')
            logger.debug(f"Text: {text}")
            logger.debug(f"Output path: {output_path}")

            # Tokenize text
            logger.debug("Tokenizing text...")
            from kokoro import phonemize, tokenize  # Import dynamically
            tokens = tokenize(phonemize(text, 'a'))
            logger.debug(f"Initial tokens: {tokens}")
            if len(tokens) > 510:
                logger.warning("Text too long; truncating to 510 tokens.")
                tokens = tokens[:510]
            tokens = [[0, *tokens, 0]]
            logger.debug(f"Final tokens: {tokens}")

            # Get voicepack style
            logger.debug("Fetching voicepack style...")
            style = voicepack[len(tokens[0]) - 2].numpy()
            logger.debug(f"Voicepack style shape: {style.shape}")

            # Run ONNX inference
            logger.debug("Running ONNX inference...")
            audio = sess.run(None, {
                'tokens': tokens,
                'style': style,
                'speed': np.ones(1, dtype=np.float32)
            })[0]
            logger.debug(f"Audio generated with shape: {audio.shape}")

            # Save audio
            logger.debug(f"Saving audio to {output_path}...")
            sf.write(output_path, audio, 24000)
            logger.info(f"Audio saved successfully to {output_path}")
            return jsonify({"status": "success", "output_path": output_path})
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """Speech-to-Text (S2T) Endpoint"""
    with global_lock:  # Acquire global lock to ensure only one instance runs
        audio_path = None
        try:
            logger.debug("Received request to /transcribe_audio")
            file = request.files['file']
            # Generate a unique filename using uuid
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            audio_path = os.path.join("/tmp", unique_filename)
            file.save(audio_path)
            logger.debug(f"Audio file saved to {audio_path}")

            # Load and preprocess audio
            logger.debug("Processing audio for transcription...")
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            input_features = processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features

            # Generate transcription
            logger.debug("Generating transcription...")
            predicted_ids = whisper_model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info(f"Transcription: {transcription}")

            return jsonify({"status": "success", "transcription": transcription})
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            # Ensure temporary file is removed
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"Temporary file {audio_path} removed")

if __name__ == '__main__':
    logger.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)

