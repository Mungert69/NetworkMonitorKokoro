from flask import Flask, request, jsonify, send_from_directory, abort
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor
import librosa
import torch
import numpy as np
from onnxruntime import InferenceSession
import os
import sys
import logging
from flask_cors import CORS
import threading
import tempfile
import subprocess
import shutil
import base64
from pathlib import Path
from huggingface_hub import snapshot_download
import hashlib
import onnxruntime as ort

# ---------------------------
# THREAD LIMIT CONFIG
# ---------------------------
MAX_THREADS = 2  # <-- change this number to control all thread usage

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

# Repository ID and paths (LFM2.5 Audio ONNX)
TTS_BACKEND = os.environ.get("TTS_BACKEND", "lfm_onnx").lower()  # lfm_onnx | gguf
ASR_BACKEND = ""  # resolved after USE_WAV2VEC2
GGUF_SERVER_URL = os.environ.get("GGUF_SERVER_URL", "http://127.0.0.1:8080/v1")
GGUF_SYSTEM_TTS = os.environ.get("GGUF_SYSTEM_TTS", "Perform TTS. Use the UK female voice.")
GGUF_SYSTEM_ASR = os.environ.get("GGUF_SYSTEM_ASR", "Perform ASR.")
GGUF_MAX_TOKENS = int(os.environ.get("GGUF_MAX_TOKENS", "1024"))
GGUF_AUDIO_SAMPLE_RATE = int(os.environ.get("GGUF_AUDIO_SAMPLE_RATE", "24000"))

LFM_MODEL_ID = os.environ.get("LFM_MODEL_ID", "LiquidAI/LFM2.5-Audio-1.5B-ONNX")
LFM_MODEL_DIR = os.environ.get("LFM_MODEL_DIR", "lfm_audio_model")
LFM_PRECISION = os.environ.get("LFM_PRECISION", "q4")
LFM_SYSTEM_PROMPT = os.environ.get("LFM_SYSTEM_PROMPT", "Perform TTS. Use the UK female voice.")
LFM_RUNNER_REPO = os.environ.get("LFM_RUNNER_REPO", "https://github.com/Liquid4All/onnx-export.git")
LFM_RUNNER_DIR = os.environ.get("LFM_RUNNER_DIR", "lfm_runner")
LFM_PYTHONPATH = os.environ.get("LFM_PYTHONPATH", os.path.join(LFM_RUNNER_DIR, "src"))
LFM_BOOTSTRAP = os.environ.get("LFM_BOOTSTRAP", "1").lower() in {"1", "true", "yes", "on"}
LFM_AUDIO_TEMPERATURE = os.environ.get("LFM_AUDIO_TEMPERATURE", "0.8")
LFM_AUDIO_TOP_K = os.environ.get("LFM_AUDIO_TOP_K", "64")
LFM_MAX_TOKENS = os.environ.get("LFM_MAX_TOKENS", "1024")
LFM_TEXT_TEMPERATURE = os.environ.get("LFM_TEXT_TEMPERATURE", "0.7")
LFM_LOAD_AUDIO_ENCODER = os.environ.get("LFM_LOAD_AUDIO_ENCODER", "0").lower() in {"1", "true", "yes", "on"}
LFM_DECODER = os.environ.get("LFM_DECODER", "")
LFM_AUDIO_EMBEDDING = os.environ.get("LFM_AUDIO_EMBEDDING", "")
LFM_AUDIO_ENCODER = os.environ.get("LFM_AUDIO_ENCODER", "")
LFM_AUDIO_DETOKENIZER = os.environ.get("LFM_AUDIO_DETOKENIZER", "")
LFM_VOCODER_DEPTHFORMER = os.environ.get("LFM_VOCODER_DEPTHFORMER", "")
lfm_model_dir = None
lfm_onnx_dir = None
lfm_files = {}
lfm_model = None
LFM2AudioInference = None
audio_codes_to_wav = None
resolve_precision_files = None
gguf_client = None

# Directory to serve files from
default_serve_dir = os.path.join(os.path.expanduser("~"), "app", "files")
SERVE_DIR = os.environ.get("SERVE_DIR", default_serve_dir)

os.makedirs(SERVE_DIR, exist_ok=True)

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

def resolve_lfm_model_root(base_dir):
    if os.path.isfile(os.path.join(base_dir, "config.json")):
        return base_dir
    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    return base_dir

def resolve_lfm_onnx_dir(base_dir):
    if os.path.isdir(os.path.join(base_dir, "onnx")):
        return os.path.join(base_dir, "onnx")
    for root, dirs, _ in os.walk(base_dir):
        if "onnx" in dirs:
            return os.path.join(root, "onnx")
    return base_dir

def ensure_lfm_python():
    global LFM2AudioInference, audio_codes_to_wav, resolve_precision_files

    if LFM2AudioInference is not None:
        return

    if LFM_PYTHONPATH and os.path.isdir(LFM_PYTHONPATH) and LFM_PYTHONPATH not in sys.path:
        sys.path.insert(0, LFM_PYTHONPATH)

    try:
        from liquidonnx.lfm2_audio.infer import (  # type: ignore
            LFM2AudioInference,
            audio_codes_to_wav,
            resolve_precision_files,
        )
        return
    except Exception:
        if not LFM_BOOTSTRAP:
            raise

    if shutil.which("git") is None:
        raise RuntimeError("git is required to bootstrap LFM runner source")
    if not os.path.exists(LFM_RUNNER_DIR):
        logger.info(f"Cloning LFM ONNX runner repo to {LFM_RUNNER_DIR}...")
        subprocess.run(
            ["git", "clone", LFM_RUNNER_REPO, LFM_RUNNER_DIR],
            check=True,
        )
    if LFM_PYTHONPATH and os.path.isdir(LFM_PYTHONPATH) and LFM_PYTHONPATH not in sys.path:
        sys.path.insert(0, LFM_PYTHONPATH)

    from liquidonnx.lfm2_audio.infer import (  # type: ignore
        LFM2AudioInference,
        audio_codes_to_wav,
        resolve_precision_files,
    )

def get_gguf_client():
    global gguf_client
    if gguf_client is not None:
        return gguf_client
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is required for GGUF backend") from e
    gguf_client = OpenAI(base_url=GGUF_SERVER_URL, api_key="dummy")
    return gguf_client

def write_wav_from_float32(samples, output_path, sample_rate):
    import wave
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

def gguf_tts(text, output_path):
    client = get_gguf_client()
    messages = [
        {"role": "system", "content": GGUF_SYSTEM_TTS},
        {"role": "user", "content": text},
    ]
    stream = client.chat.completions.create(
        model="",
        messages=messages,
        max_tokens=GGUF_MAX_TOKENS,
        stream=True,
        extra_body={"reset_context": True},
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        audio_chunk = getattr(delta, "audio_chunk", None)
        if audio_chunk and isinstance(audio_chunk, dict):
            data = audio_chunk.get("data")
            if data:
                raw = base64.b64decode(data)
                chunks.append(np.frombuffer(raw, dtype=np.float32))
    if not chunks:
        raise RuntimeError("GGUF TTS produced no audio")
    audio = np.concatenate(chunks, axis=0)
    write_wav_from_float32(audio, output_path, GGUF_AUDIO_SAMPLE_RATE)

def gguf_asr(wav_path):
    client = get_gguf_client()
    with open(wav_path, "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode("utf-8")
    messages = [
        {"role": "system", "content": GGUF_SYSTEM_ASR},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}
            ],
        },
    ]
    stream = client.chat.completions.create(
        model="",
        messages=messages,
        max_tokens=GGUF_MAX_TOKENS,
        stream=True,
        extra_body={"reset_context": True},
    )
    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            parts.append(delta.content)
    return "".join(parts).strip()

use_wav2vec2 = os.environ.get("USE_WAV2VEC2", "").lower() in {"1", "true", "yes", "on"}
ASR_BACKEND = os.environ.get("ASR_BACKEND", "")
if not ASR_BACKEND:
    ASR_BACKEND = "wav2vec2_onnx" if use_wav2vec2 else "whisper_pt"
ASR_BACKEND = ASR_BACKEND.lower()
ASR_MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "facebook/wav2vec2-base-960h")
ASR_ONNX_REPO = os.environ.get("ASR_ONNX_REPO", "onnx-community/wav2vec2-base-960h-ONNX")

# Initialize models
def initialize_models():
    global processor, whisper_model, asr_session, asr_processor
    global lfm_model_dir, lfm_onnx_dir, lfm_files, lfm_model

    try:
        if TTS_BACKEND == "lfm_onnx":
            # Download the LFM2.5 Audio ONNX model into a real local directory (no symlinks)
            config_path = os.path.join(LFM_MODEL_DIR, "config.json")
            if not os.path.isfile(config_path):
                logger.info("Downloading LFM2.5 Audio ONNX model (materialized, no symlinks)...")
                lfm_model_dir = snapshot_download(
                    LFM_MODEL_ID,
                    local_dir=LFM_MODEL_DIR,
                    local_dir_use_symlinks=False,
                )
                logger.info(f"LFM model directory: {lfm_model_dir}")
            else:
                lfm_model_dir = LFM_MODEL_DIR
                logger.info(f"Using cached LFM model directory: {lfm_model_dir}")

            # Resolve model root (config.json) and onnx dir
            lfm_model_dir = resolve_lfm_model_root(lfm_model_dir)
            lfm_onnx_dir = resolve_lfm_onnx_dir(lfm_model_dir)
            logger.info(f"Resolved LFM model directory: {lfm_model_dir}")
            logger.info(f"Resolved LFM ONNX directory: {lfm_onnx_dir}")

            # Ensure python module for in-process inference is available
            ensure_lfm_python()

            # Resolve ONNX filenames for selected precision
            lfm_files = resolve_precision_files(LFM_PRECISION)
            if LFM_DECODER:
                lfm_files["decoder"] = LFM_DECODER
            if LFM_AUDIO_EMBEDDING:
                lfm_files["audio_embedding"] = LFM_AUDIO_EMBEDDING
            if LFM_AUDIO_ENCODER:
                lfm_files["audio_encoder"] = LFM_AUDIO_ENCODER
            if LFM_AUDIO_DETOKENIZER:
                lfm_files["audio_detokenizer"] = LFM_AUDIO_DETOKENIZER
            if LFM_VOCODER_DEPTHFORMER:
                lfm_files["vocoder_depthformer"] = LFM_VOCODER_DEPTHFORMER
            if not LFM_LOAD_AUDIO_ENCODER:
                lfm_files["audio_encoder"] = "audio_encoder.DISABLED.onnx"

            lfm_model = LFM2AudioInference(
                Path(lfm_model_dir),
                decoder_file=lfm_files.get("decoder"),
                audio_embedding_file=lfm_files.get("audio_embedding"),
                audio_encoder_file=lfm_files.get("audio_encoder"),
                audio_detokenizer_file=lfm_files.get("audio_detokenizer"),
                vocoder_depthformer_file=lfm_files.get("vocoder_depthformer"),
            )

        # Initialize ASR engine
        if ASR_BACKEND == "wav2vec2_onnx":
            logger.info(f"Loading Wav2Vec2 ONNX ASR model ({ASR_MODEL_NAME})...")
            # Load processor for feature extraction + CTC labels
            asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)

            # Try to locate/download ONNX model; if not present, download a ready-made ONNX repo.
            default_onnx_path = f"asr_onnx/{ASR_MODEL_NAME.replace('/', '_')}.onnx"
            asr_onnx_path_env = os.environ.get("ASR_ONNX_PATH", default_onnx_path)
            if not os.path.exists(asr_onnx_path_env):
                logger.info(f"ASR ONNX not found at {asr_onnx_path_env}. Attempting to download from {ASR_ONNX_REPO}...")
                try:
                    cache_dir = os.environ.get("ASR_ONNX_CACHE_DIR", "asr_onnx_cache")
                    repo_dir = snapshot_download(ASR_ONNX_REPO, cache_dir=cache_dir)
                    # Look for common ONNX filenames
                    onnx_path = None
                    for root, _, files in os.walk(repo_dir):
                        for cand in ["model.onnx", "wav2vec2.onnx", "onnx/model.onnx"]:
                            if cand in files:
                                onnx_path = os.path.join(root, cand if cand != "onnx/model.onnx" else "model.onnx")
                                break
                        if onnx_path:
                            break
                    if not onnx_path:
                        # Fallback: pick first .onnx file found
                        for root, _, files in os.walk(repo_dir):
                            for f in files:
                                if f.endswith(".onnx"):
                                    onnx_path = os.path.join(root, f)
                                    break
                            if onnx_path:
                                break
                    if not onnx_path:
                        raise FileNotFoundError("No .onnx file found in downloaded repo")
                    os.makedirs(os.path.dirname(asr_onnx_path_env), exist_ok=True)
                    # Copy to stable location
                    shutil.copyfile(onnx_path, asr_onnx_path_env)
                    logger.info(f"Downloaded ASR ONNX to {asr_onnx_path_env}")
                except Exception as de:
                    logger.error(f"Failed to download ASR ONNX: {de}")
                    logger.warning("Falling back to Whisper PT engine.")
                    raise
            asr_session = InferenceSession(asr_onnx_path_env, sess_options)
            logger.info("Wav2Vec2 ONNX ASR model loaded")
        elif ASR_BACKEND == "whisper_pt":
            logger.info("ASR_BACKEND set to whisper_pt; loading Whisper model...")
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            whisper_model.config.forced_decoder_ids = None
            logger.info("Whisper model loaded successfully")
        else:
            logger.info("ASR_BACKEND set to gguf; using GGUF server for ASR")

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

            # No preprocessing needed for LFM2.5
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            filename = f"{text_hash}.wav"
            cached_file_path = os.path.join(SERVE_DIR, filename)

            # Cache hit
            if is_cached(cached_file_path):
                logger.info("Returning cached audio")
                return jsonify({"status": "success", "filename": filename})

            if TTS_BACKEND == "gguf":
                gguf_tts(text, cached_file_path)
            else:
                if lfm_model is None:
                    raise Exception("LFM model not initialized")

                # Reset cache to avoid cross-request context
                lfm_model.reset()

                audio_codes = lfm_model.synthesize(
                    text,
                    max_new_tokens=int(LFM_MAX_TOKENS),
                    audio_temperature=float(LFM_AUDIO_TEMPERATURE),
                    audio_top_k=int(LFM_AUDIO_TOP_K),
                    text_temperature=float(LFM_TEXT_TEMPERATURE),
                    system_prompt=LFM_SYSTEM_PROMPT,
                )
                if not audio_codes:
                    raise Exception("LFM inference produced no audio codes")

                ok = audio_codes_to_wav(
                    audio_codes,
                    cached_file_path,
                    model_dir=Path(lfm_model_dir),
                    audio_detokenizer_file=lfm_files.get("audio_detokenizer"),
                )
                if not ok or not os.path.exists(cached_file_path):
                    raise Exception("LFM inference produced no audio output")


            logger.info(f"Audio saved: {cached_file_path}")
            return jsonify({"status": "success", "filename": filename})
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

# Speech-to-Text (S2T) Endpoint
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

            if ASR_BACKEND == "gguf":
                transcription = gguf_asr(converted_audio_path)
                logger.info(f"Transcription (GGUF): {transcription}")
            elif ASR_BACKEND == "wav2vec2_onnx" and 'asr_session' in globals() and asr_session is not None:
                # Prepare input for Wav2Vec2 ONNX: float32 PCM, shape (batch, samples)
                inputs = asr_processor(audio_array, sampling_rate=16000, return_tensors="np")
                # Some exports expect input as (batch, sequence); adjust key as needed
                ort_inputs = {}
                # Common input name variants
                for name in ["input_values", "input_features", "inputs"]:
                    if name in [i.name for i in asr_session.get_inputs()]:
                        ort_inputs[name] = inputs["input_values"].astype(np.float32)
                        break
                else:
                    # Fall back to first input name
                    first_name = asr_session.get_inputs()[0].name
                    ort_inputs[first_name] = inputs["input_values"].astype(np.float32)

                logits = asr_session.run(None, ort_inputs)[0]  # (batch, time, vocab)
                # Greedy CTC decode
                pred_ids = np.argmax(logits, axis=-1)
                # Collapse repeats and remove CTC blank (id 0 for many models; rely on processor)
                transcription = asr_processor.batch_decode(pred_ids)[0]
                transcription = transcription.strip()
                logger.info(f"Transcription (Wav2Vec2 ONNX): {transcription}")
            else:
                # Whisper fallback
                input_features = processor(
                    audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                ).input_features

                logger.debug("Generating transcription (Whisper)...")
                predicted_ids = whisper_model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                logger.info(f"Transcription (Whisper): {transcription}")

            # No extra text post-processing for ASR output

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
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, threaded=False, processes=1)
