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
import os
import torch
import numpy as np
import onnxruntime as ort

# ---------------------------
# THREAD LIMIT CONFIG
# ---------------------------
MAX_THREADS = 2  # <-- change this number to control all thread usage

# ---------------------------
# ---------------------------
# STORAGE ROOT
# ---------------------------
SERVE_DIR = "/home/user/app/files"
os.makedirs(SERVE_DIR, exist_ok=True)

# Limit NumPy / BLAS / MKL threads
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(MAX_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_THREADS)

# Torch thread limits
torch.set_num_threads(MAX_THREADS)
torch.set_num_interop_threads(1)  # keep inter-op small to avoid overhead

# ONNXRuntime session options (use when creating the session)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = MAX_THREADS
sess_options.inter_op_num_threads = 1


# Configure logging
logging.basicConfig(level=logging.INFO)
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
    """Validates audio files including WebM/Opus format"""
    if not isinstance(file, werkzeug.datastructures.FileStorage):
        raise ValueError("Invalid file type")
    
    # Supported MIME types (add WebM/Opus)
    supported_types = [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/webm",
        "audio/ogg"  # For Opus in Ogg container
    ]
    
    # Check MIME type
    if file.content_type not in supported_types:
        raise ValueError(f"Unsupported file type. Must be one of: {', '.join(supported_types)}")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    max_size = 10 * 1024 * 1024  # 10 MB
    if file_size > max_size:
        raise ValueError(f"File is too large (max {max_size//(1024*1024)} MB)")
    
    # Optional: Verify file header matches content_type
    if not verify_audio_header(file):
        raise ValueError("File header doesn't match declared content type")
def verify_audio_header(file):
    """Quickly checks if file headers match the declared audio format"""
    header = file.read(4)
    file.seek(0)  # Rewind after reading
    
    if file.content_type in ["audio/webm", "audio/ogg"]:
        # WebM starts with \x1aE\xdf\xa3, Ogg with OggS
        return (
            (file.content_type == "audio/webm" and header.startswith(b'\x1aE\xdf\xa3')) or
            (file.content_type == "audio/ogg" and header.startswith(b'OggS'))
        )
    elif file.content_type in ["audio/wav", "audio/x-wav"]:
        return header.startswith(b'RIFF')
    elif file.content_type in ["audio/mpeg", "audio/mp3"]:
        return header.startswith(b'\xff\xfb')  # MP3 frame sync
    return True  # Skip verification for other types

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
        sess = InferenceSession(onnx_path, sess_options)
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
    with global_lock:
        try:
            logger.debug("Received request to /generate_audio")
            data = request.json
            text = data['text']

            validate_text_input(text)

            # Preprocess & stable hash
            text = preprocess_all(text)
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            filename = f"{text_hash}.wav"
            cached_file_path = os.path.join(SERVE_DIR, filename)

            # Cache hit
            if is_cached(cached_file_path):
                logger.info("Returning cached audio")
                return jsonify({"status": "success", "filename": filename})

            # Tokenize
            from kokoro import phonemize, tokenize  # lazy import is fine
            tokens = tokenize(phonemize(text, 'a'))
            if len(tokens) > 510:
                logger.warning("Text too long; truncating to 510 tokens.")
                tokens = tokens[:510]
            tokens = [[0, *tokens, 0]]

            # Style vector
            ref_s = voice_style[len(tokens[0]) - 2]  # (1,256)

            # ONNX inference
            audio = sess.run(None, dict(
                input_ids=np.array(tokens, dtype=np.int64),
                style=ref_s,
                speed=np.ones(1, dtype=np.float32),
            ))[0]

            # Save
            audio = np.squeeze(audio).astype(np.float32)
            sf.write(cached_file_path, audio, 24000)

            logger.info(f"Audio saved: {cached_file_path}")
            return jsonify({"status": "success", "filename": filename})
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

# Speech-to-Text (S2T) Endpoint
# Add these imports at the top with the other imports
import subprocess
import tempfile
from pathlib import Path

# Then update the transcribe_audio function:
@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """Speech-to-Text (S2T) Endpoint with automatic format conversion"""
    with global_lock:  # Acquire global lock to ensure only one instance runs
        input_audio_path = None
        converted_audio_path = None
        try:
            logger.debug("Received request to /transcribe_audio")
            file = request.files['file']
            
            # Create temporary files for both input and output
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as input_temp:
                input_audio_path = input_temp.name
                file.save(input_audio_path)
                logger.debug(f"Original audio file saved to {input_audio_path}")
            
            # Create a temporary file for the converted WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_temp:
                converted_audio_path = output_temp.name
            
            # Convert to WAV with ffmpeg (16kHz, mono)
            logger.debug(f"Converting audio to 16kHz mono WAV format...")
            conversion_command = [
                'ffmpeg',
                '-y',                  # Force overwrite without prompting
                '-i', input_audio_path,
                '-acodec', 'pcm_s16le', # 16-bit PCM
                '-ac', '1',             # mono
                '-ar', '16000',         # 16kHz sample rate
                '-af', 'highpass=f=80,lowpass=f=7500,afftdn=nr=10:nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11',  # Audio cleanup filters
                converted_audio_path
            ]
            result = subprocess.run(
                conversion_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg conversion error: {result.stderr}")
                raise Exception(f"Audio conversion failed: {result.stderr}")
            
            logger.debug(f"Audio successfully converted to {converted_audio_path}")
            
            # Load and process the converted audio
            logger.debug("Processing audio for transcription...")
            audio_array, sampling_rate = librosa.load(converted_audio_path, sr=16000)

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
            # Clean up temporary files
            for path in [input_audio_path, converted_audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.debug(f"Temporary file {path} removed")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {path}: {e}")

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
    app.run(host="0.0.0.0", port=7860, threaded=False, processes=1)

