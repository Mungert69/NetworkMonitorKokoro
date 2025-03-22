from flask import Flask, request, jsonify, send_from_directory, abort
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
import werkzeug
import tempfile
from huggingface_hub import snapshot_download
from tts_processor import preprocess_all
import hashlib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global lock to ensure one method runs at a time
global_lock = threading.Lock()

# Repository ID and paths
kokoro_model_id = 'onnx-community/Kokoro-82M-v1.0-ONNX'
model_path = 'kokoro_model'
voice_name = 'am_adam'  # Example voice: af (adjust as needed)

# Directory to serve files from
SERVE_DIR = os.environ.get("SERVE_DIR", "./files")  # Default to './files' if not provided

os.makedirs(SERVE_DIR, exist_ok=True)
def validate_audio_file(file):
    if not isinstance(file, werkzeug.datastructures.FileStorage):
        raise ValueError("Invalid file type")
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
        raise ValueError("Unsupported file type")
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    if file_size > 10 * 1024 * 1024:  # 10 MB limit
        raise ValueError("File is too large (max 10 MB)")

def validate_text_input(text):
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    if len(text.strip()) == 0:
        raise ValueError("Text input cannot be empty")
    if len(text) > 1024:  # Limit to 1024 characters
        raise ValueError("Text input is too long (max 1024 characters)")

file_cache = {}

def is_cached(cached_file_path):
    """
    Check if a file exists in the cache.
    If the file is not in the cache, perform a disk check and update the cache.
    """
    if cached_file_path in file_cache:
        return file_cache[cached_file_path]  # Return cached result
    exists = os.path.exists(cached_file_path)  # Perform disk check
    file_cache[cached_file_path] = exists  # Update the cache
    return exists

# Initialize models
def initialize_models():
    global sess, voice_style, processor, whisper_model

    try:
        # Download the ONNX model if not already downloaded
        if not os.path.exists(model_path):
            logger.info("Downloading and loading Kokoro model...")
            kokoro_dir = snapshot_download(kokoro_model_id, cache_dir=model_path)
            logger.info(f"Kokoro model directory: {kokoro_dir}")
        else:
            kokoro_dir = model_path
            logger.info(f"Using cached Kokoro model directory: {kokoro_dir}")

        # Validate ONNX file path
        onnx_path = None
        for root, _, files in os.walk(kokoro_dir):
            if 'model.onnx' in files:
                onnx_path = os.path.join(root, 'model.onnx')
                break

        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found after redownload at {kokoro_dir}")

        logger.info("Loading ONNX session...")
        sess = InferenceSession(onnx_path)
        logger.info(f"ONNX session loaded successfully from {onnx_path}")

        # Load the voice style vector
        voice_style_path = None
        for root, _, files in os.walk(kokoro_dir):
            if f'{voice_name}.bin' in files:
                voice_style_path = os.path.join(root, f'{voice_name}.bin')
                break

        if not voice_style_path or not os.path.exists(voice_style_path):
            raise FileNotFoundError(f"Voice style file not found at {voice_style_path}")

        logger.info("Loading voice style vector...")
        voice_style = np.fromfile(voice_style_path, dtype=np.float32).reshape(-1, 1, 256)
        logger.info(f"Voice style vector loaded successfully from {voice_style_path}")

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

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy"}), 500

# Text-to-Speech (T2S) Endpoint
@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    """Text-to-Speech (T2S) Endpoint"""
    with global_lock:  # Acquire global lock to ensure only one instance runs
        try:
            logger.debug("Received request to /generate_audio")
            data = request.json
            text = data['text']
            output_dir = data.get('output_dir')

            validate_text_input(text)
            logger.debug(f"Text: {text}")
            if not output_dir:
                raise ValueError("Output directory is required but not provided")

            # Ensure output_dir is an absolute path and valid
            if not os.path.isabs(output_dir):
                raise ValueError("Output directory must be an absolute path")
            if not os.path.exists(output_dir):
                raise ValueError(f"Output directory does not exist: {output_dir}")

            # Generate a unique hash for the text
            text = preprocess_all(text)
            logger.debug(f"Processed Text {text}")
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            hashed_file_name = f"{text_hash}.wav"
            cached_file_path = os.path.join(output_dir, hashed_file_name)
            logger.debug(f"Generated hash for processed text: {text_hash}")
            logger.debug(f"Output directory: {output_dir}")
            logger.debug(f"Cached file path: {cached_file_path}")

            # Check if cached file exists
            if is_cached(cached_file_path):
                logger.info(f"Returning cached audio for text: {text}")
                return jsonify({"status": "success", "output_path": cached_file_path})

            # Tokenize text
            logger.debug("Tokenizing text...")
            from kokoro import phonemize, tokenize  # Import dynamically
            tokens = tokenize(phonemize(text, 'a'))
            logger.debug(f"Initial tokens: {tokens}")
            if len(tokens) > 510:
                logger.warning("Text too long; truncating to 510 tokens.")
                tokens = tokens[:510]
            tokens = [[0, *tokens, 0]]  # Add pad tokens
            logger.debug(f"Final tokens: {tokens}")

            # Get style vector based on token length
            logger.debug("Fetching style vector...")
            ref_s = voice_style[len(tokens[0]) - 2]  # Shape: (1, 256)
            logger.debug(f"Style vector shape: {ref_s.shape}")

            # Run ONNX inference
            logger.debug("Running ONNX inference...")
            audio = sess.run(None, dict(
                input_ids=np.array(tokens, dtype=np.int64),
                style=ref_s,
                speed=np.ones(1, dtype=np.float32),
            ))[0]
            logger.debug(f"Audio generated with shape: {audio.shape}")

            # Fix audio data for saving
            audio = np.squeeze(audio)  # Remove extra dimension
            audio = audio.astype(np.float32)  # Ensure correct data type

            # Save audio
            logger.debug(f"Saving audio to {cached_file_path}...")
            sf.write(cached_file_path, audio, 24000)  # Save with 24 kHz sample rate
            logger.info(f"Audio saved successfully to {cached_file_path}")
            return jsonify({"status": "success", "output_path": cached_file_path})
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

# Speech-to-Text (S2T) Endpoint
@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """Speech-to-Text (S2T) Endpoint"""
    with global_lock:  # Acquire global lock to ensure only one instance runs
        audio_path = None
        try:
            logger.debug("Received request to /transcribe_audio")
            file = request.files['file']
            validate_audio_file(file)
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

@app.route('/files/<filename>', methods=['GET'])
def serve_wav_file(filename):
    """
    Serve a .wav file from the configured directory.
    Only serves files ending with '.wav'.
    """
    # Ensure only .wav files are allowed
    if not filename.lower().endswith('.wav'):
        abort(400, "Only .wav files are allowed.")
    
    # Check if the file exists in the directory
    file_path = os.path.join(SERVE_DIR, filename)
    logger.debug(f"Looking for file at: {file_path}")
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        abort(404, "File not found.")
    
    # Serve the file
    return send_from_directory(SERVE_DIR, filename)

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors."""
    return {"error": "Bad Request", "message": str(error)}, 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return {"error": "Not Found", "message": str(error)}, 404

@app.errorhandler(500)
def internal_error(error):
    """Handle unexpected errors."""
    return {"error": "Internal Server Error", "message": "An unexpected error occurred."}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
