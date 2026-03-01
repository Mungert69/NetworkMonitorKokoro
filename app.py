from flask import Flask, request, jsonify, send_from_directory, abort
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor, AutoTokenizer, AutoModelForTokenClassification
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
import re
import threading
import time
import werkzeug
import tempfile
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download
from tts_processor import preprocess_all
import hashlib
import shutil
import os
import torch
import numpy as np
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
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
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
default_serve_dir = os.path.join(os.path.expanduser("~"), "app", "files")
SERVE_DIR = os.environ.get("SERVE_DIR", default_serve_dir)

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

use_wav2vec2 = os.environ.get("USE_WAV2VEC2", "").lower() in {"1", "true", "yes", "on"}
ASR_ENGINE = os.environ.get("ASR_ENGINE", "wav2vec2_onnx" if use_wav2vec2 else "whisper_pt").lower()
ASR_MODEL_NAME = os.environ.get("ASR_MODEL_NAME", "facebook/wav2vec2-base-960h")
ASR_ONNX_REPO = os.environ.get("ASR_ONNX_REPO", "onnx-community/wav2vec2-base-960h-ONNX")
PUNCTUATE_TEXT = os.environ.get("PUNCTUATE_TEXT", "0").lower() in {"1", "true", "yes", "on"}
TECH_NORMALIZE = os.environ.get("TECH_NORMALIZE", "0").lower() in {"1", "true", "yes", "on"}
PUNCTUATION_MODEL = os.environ.get("PUNCTUATION_MODEL", "kredor/punctuate-all")
KOKORO_ONNX_MODE = os.environ.get("KOKORO_ONNX_MODE", "auto").lower()  # auto | legacy | stts2 | piper
KOKORO_SPEAKER_ID = int(os.environ.get("KOKORO_SPEAKER_ID", "0"))
KOKORO_MAX_PHONEME_TOKENS = int(os.environ.get("KOKORO_MAX_PHONEME_TOKENS", "510"))
KOKORO_STTS2_NOISE_SCALE = float(os.environ.get("KOKORO_STTS2_NOISE_SCALE", "0.667"))
KOKORO_STTS2_LENGTH_SCALE = float(os.environ.get("KOKORO_STTS2_LENGTH_SCALE", "1.0"))
KOKORO_STTS2_NOISE_W = float(os.environ.get("KOKORO_STTS2_NOISE_W", "0.8"))
PIPER_BIN = os.environ.get("PIPER_BIN", "piper")
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", "")
PIPER_CONFIG_PATH = os.environ.get("PIPER_CONFIG_PATH", "")
PIPER_AUTO_DOWNLOAD = os.environ.get("PIPER_AUTO_DOWNLOAD", "1").lower() in {"1", "true", "yes", "on"}
PIPER_REPO_ID = os.environ.get("PIPER_REPO_ID", "campwill/HAL-9000-Piper-TTS")
PIPER_REPO_MODEL_FILE = os.environ.get("PIPER_REPO_MODEL_FILE", "hal.onnx")
PIPER_REPO_CONFIG_FILE = os.environ.get("PIPER_REPO_CONFIG_FILE", "hal.onnx.json")
PIPER_DOWNLOAD_DIR = os.environ.get("PIPER_DOWNLOAD_DIR", os.path.join(model_path, "piper_repo"))
MODEL_INIT_MODE = os.environ.get("MODEL_INIT_MODE", "lazy").lower()  # lazy | startup
PIPER_PREFETCH_ON_STARTUP = os.environ.get("PIPER_PREFETCH_ON_STARTUP", "0").lower() in {"1", "true", "yes", "on"}
REQUEST_LOCK_TIMEOUT_SEC = float(os.environ.get("REQUEST_LOCK_TIMEOUT_SEC", "15"))
PIPER_TIMEOUT_SEC = float(os.environ.get("PIPER_TIMEOUT_SEC", "120"))
FFMPEG_TIMEOUT_SEC = float(os.environ.get("FFMPEG_TIMEOUT_SEC", "60"))

sess = None
voice_style = None
processor = None
whisper_model = None
asr_session = None
asr_processor = None
punctuation_model = None
punctuation_tokenizer = None
kokoro_tts_mode = None
kokoro_input_names = set()
model_init_lock = threading.Lock()

# Initialize models
def initialize_models(init_tts=True, init_asr=True):
    global sess, voice_style, processor, whisper_model, asr_session, asr_processor
    global kokoro_tts_mode, kokoro_input_names
    global punctuation_model, punctuation_tokenizer

    try:
        if os.path.exists(model_path) and not os.path.isdir(model_path):
            raise FileExistsError(f"Model path exists but is not a directory: {model_path}")

        def find_onnx(search_root):
            for root, _, files in os.walk(search_root):
                if "model.onnx" in files:
                    return os.path.join(root, "model.onnx")
            return None

        def download_kokoro():
            allow_patterns_env = os.environ.get("KOKORO_ALLOW_PATTERNS", "")
            if allow_patterns_env.strip():
                allow_patterns = [p.strip() for p in allow_patterns_env.split(",") if p.strip()]
            else:
                allow_patterns = [
                    "**/model.onnx",
                    f"**/{voice_name}.bin",
                    "config.json",
                ]

            logger.info("Downloading and loading Kokoro model...")
            try:
                import inspect
                sig = inspect.signature(snapshot_download)
                if "local_dir" in sig.parameters:
                    return snapshot_download(
                        kokoro_model_id,
                        local_dir=model_path,
                        local_dir_use_symlinks=False,
                        allow_patterns=allow_patterns,
                        resume_download=True,
                    )
                return snapshot_download(
                    kokoro_model_id,
                    cache_dir=model_path,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )
            except Exception:
                return snapshot_download(
                    kokoro_model_id,
                    cache_dir=model_path,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )

        if init_tts and kokoro_tts_mode is None:
            if KOKORO_ONNX_MODE == "piper":
                # Piper mode performs synthesis through the Piper CLI and does not use ORT here.
                sess = None
                kokoro_input_names = set()
                kokoro_tts_mode = "piper"
            else:
                kokoro_dir = model_path if os.path.exists(model_path) else None
                onnx_path = find_onnx(kokoro_dir) if kokoro_dir else None

                if not onnx_path:
                    kokoro_dir = download_kokoro()
                    logger.info(f"Kokoro model directory: {kokoro_dir}")
                    onnx_path = find_onnx(kokoro_dir)

                if not onnx_path or not os.path.exists(onnx_path):
                    raise FileNotFoundError(f"ONNX file not found after redownload at {kokoro_dir}")

                logger.info("Loading ONNX session...")
                sess = InferenceSession(onnx_path, sess_options)
                logger.info(f"ONNX session loaded successfully from {onnx_path}")
                kokoro_input_names = {i.name for i in sess.get_inputs()}
                if KOKORO_ONNX_MODE == "auto":
                    if {"input_ids", "style", "speed"}.issubset(kokoro_input_names):
                        kokoro_tts_mode = "legacy"
                    elif {"input", "input_lengths"}.issubset(kokoro_input_names):
                        kokoro_tts_mode = "stts2"
                    else:
                        raise RuntimeError(
                            f"Could not auto-detect ONNX input mode from inputs: {sorted(kokoro_input_names)}. "
                            "Set KOKORO_ONNX_MODE=legacy or KOKORO_ONNX_MODE=stts2."
                        )
                elif KOKORO_ONNX_MODE in {"legacy", "stts2"}:
                    kokoro_tts_mode = KOKORO_ONNX_MODE
                else:
                    raise ValueError("KOKORO_ONNX_MODE must be one of: auto, legacy, stts2, piper")

            logger.info(f"Kokoro ONNX input names: {sorted(kokoro_input_names) if kokoro_input_names else []}")
            logger.info(f"Kokoro ONNX mode: {kokoro_tts_mode}")

            if kokoro_tts_mode == "legacy":
                # Legacy Kokoro ONNX expects a style embedding from voice .bin
                voice_style_path = None
                for root, _, files in os.walk(kokoro_dir):
                    if f'{voice_name}.bin' in files:
                        voice_style_path = os.path.join(root, f'{voice_name}.bin')
                        break

                if not voice_style_path or not os.path.exists(voice_style_path):
                    kokoro_dir = download_kokoro()
                    for root, _, files in os.walk(kokoro_dir):
                        if f'{voice_name}.bin' in files:
                            voice_style_path = os.path.join(root, f'{voice_name}.bin')
                            break
                    if not voice_style_path or not os.path.exists(voice_style_path):
                        raise FileNotFoundError(f"Voice style file not found at {voice_style_path}")

                logger.info("Loading voice style vector...")
                voice_style = np.fromfile(voice_style_path, dtype=np.float32).reshape(-1, 1, 256)
                logger.info(f"Voice style vector loaded successfully from {voice_style_path}")

        # Initialize ASR engine
        if init_asr and ASR_ENGINE == "wav2vec2_onnx" and asr_session is None:
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
                    import shutil
                    shutil.copyfile(onnx_path, asr_onnx_path_env)
                    logger.info(f"Downloaded ASR ONNX to {asr_onnx_path_env}")
                except Exception as de:
                    logger.error(f"Failed to download ASR ONNX: {de}")
                    logger.warning("Falling back to Whisper PT engine.")
                    raise
            asr_session = InferenceSession(asr_onnx_path_env, sess_options)
            logger.info("Wav2Vec2 ONNX ASR model loaded")
        elif init_asr and ASR_ENGINE != "wav2vec2_onnx" and (processor is None or whisper_model is None):
            logger.info("ASR_ENGINE set to whisper_pt; loading Whisper model...")
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            whisper_model.config.forced_decoder_ids = None
            logger.info("Whisper model loaded successfully")

        if init_asr and PUNCTUATE_TEXT and punctuation_model is None:
            logger.info(f"Loading punctuation model ({PUNCTUATION_MODEL})...")
            punctuation_tokenizer = AutoTokenizer.from_pretrained(PUNCTUATION_MODEL)
            punctuation_model = AutoModelForTokenClassification.from_pretrained(PUNCTUATION_MODEL)
            punctuation_model.eval()
            logger.info("Punctuation model loaded successfully")

    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

def ensure_models_for_tts():
    if kokoro_tts_mode is not None:
        return
    with model_init_lock:
        if kokoro_tts_mode is None:
            initialize_models(init_tts=True, init_asr=False)

def ensure_models_for_asr():
    asr_ready = (ASR_ENGINE == "wav2vec2_onnx" and asr_session is not None) or (
        ASR_ENGINE != "wav2vec2_onnx" and processor is not None and whisper_model is not None
    )
    punct_ready = (not PUNCTUATE_TEXT) or (punctuation_model is not None and punctuation_tokenizer is not None)
    if asr_ready and punct_ready:
        return
    with model_init_lock:
        asr_ready = (ASR_ENGINE == "wav2vec2_onnx" and asr_session is not None) or (
            ASR_ENGINE != "wav2vec2_onnx" and processor is not None and whisper_model is not None
        )
        punct_ready = (not PUNCTUATE_TEXT) or (punctuation_model is not None and punctuation_tokenizer is not None)
        if not (asr_ready and punct_ready):
            initialize_models(init_tts=False, init_asr=True)

def _resolve_piper_model_and_config():
    model_candidate = PIPER_MODEL_PATH.strip()
    if not model_candidate:
        model_candidate = os.path.join(model_path, "onnx", "model.onnx")

    config_candidate = PIPER_CONFIG_PATH.strip()
    if not config_candidate:
        for cand in [
            f"{model_candidate}.json",
            os.path.join(os.path.dirname(model_candidate), "config.json"),
            os.path.join(model_path, "config.json"),
        ]:
            if os.path.isfile(cand):
                config_candidate = cand
                break

    missing_model = not os.path.isfile(model_candidate)
    missing_config = not config_candidate or not os.path.isfile(config_candidate)

    if (missing_model or missing_config) and PIPER_AUTO_DOWNLOAD:
        logger.info(
            f"Missing Piper assets (model_missing={missing_model}, config_missing={missing_config}). "
            f"Downloading from {PIPER_REPO_ID}..."
        )
        allow_patterns = [PIPER_REPO_MODEL_FILE, PIPER_REPO_CONFIG_FILE]
        try:
            import inspect
            sig = inspect.signature(snapshot_download)
            if "local_dir" in sig.parameters:
                repo_dir = snapshot_download(
                    PIPER_REPO_ID,
                    local_dir=PIPER_DOWNLOAD_DIR,
                    local_dir_use_symlinks=False,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )
            else:
                repo_dir = snapshot_download(
                    PIPER_REPO_ID,
                    cache_dir=PIPER_DOWNLOAD_DIR,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )
        except Exception:
            repo_dir = snapshot_download(
                PIPER_REPO_ID,
                cache_dir=PIPER_DOWNLOAD_DIR,
                allow_patterns=allow_patterns,
                resume_download=True,
            )

        src_model = None
        src_config = None
        for root, _, files in os.walk(repo_dir):
            for f in files:
                if f == PIPER_REPO_MODEL_FILE:
                    src_model = os.path.join(root, f)
                elif f == PIPER_REPO_CONFIG_FILE:
                    src_config = os.path.join(root, f)

        if missing_model:
            if not src_model:
                raise FileNotFoundError(
                    f"Failed to find {PIPER_REPO_MODEL_FILE} in downloaded repo {PIPER_REPO_ID}"
                )
            os.makedirs(os.path.dirname(model_candidate), exist_ok=True)
            shutil.copyfile(src_model, model_candidate)
            logger.info(f"Downloaded Piper model to {model_candidate}")

        if not config_candidate:
            config_candidate = f"{model_candidate}.json"

        if not os.path.isfile(config_candidate):
            if not src_config:
                raise FileNotFoundError(
                    f"Failed to find {PIPER_REPO_CONFIG_FILE} in downloaded repo {PIPER_REPO_ID}"
                )
            os.makedirs(os.path.dirname(config_candidate), exist_ok=True)
            shutil.copyfile(src_config, config_candidate)
            logger.info(f"Downloaded Piper config to {config_candidate}")

    if not os.path.isfile(model_candidate):
        raise FileNotFoundError(
            f"Piper model not found at {model_candidate}. "
            f"Set PIPER_MODEL_PATH or enable PIPER_AUTO_DOWNLOAD from {PIPER_REPO_ID}."
        )

    if config_candidate and not os.path.isfile(config_candidate):
        raise FileNotFoundError(
            f"Piper config not found at {config_candidate}. "
            "Set PIPER_CONFIG_PATH to the .onnx.json file (for HAL use hal.onnx.json)."
        )
    return model_candidate, config_candidate

if MODEL_INIT_MODE == "startup":
    initialize_models(init_tts=True, init_asr=True)
    if KOKORO_ONNX_MODE == "piper" and PIPER_PREFETCH_ON_STARTUP:
        _resolve_piper_model_and_config()

def synthesize_with_piper(text, output_path):
    model_file, config_file = _resolve_piper_model_and_config()

    cmd = [PIPER_BIN, "--model", model_file, "--output_file", output_path]
    if config_file:
        cmd.extend(["--config", config_file])
    if KOKORO_SPEAKER_ID >= 0:
        cmd.extend(["--speaker", str(KOKORO_SPEAKER_ID)])

    cmd.extend([
        "--noise_scale", str(KOKORO_STTS2_NOISE_SCALE),
        "--length_scale", str(KOKORO_STTS2_LENGTH_SCALE),
        "--noise_w", str(KOKORO_STTS2_NOISE_W),
    ])

    try:
        result = subprocess.run(
            cmd,
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=PIPER_TIMEOUT_SEC,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Piper binary not found: {PIPER_BIN}. Install Piper or set PIPER_BIN."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Piper synthesis timed out after {PIPER_TIMEOUT_SEC:.0f}s. "
            f"Command: {' '.join(cmd)}"
        ) from e

    if result.returncode != 0:
        raise RuntimeError(f"Piper synthesis failed: {result.stderr.strip() or result.stdout.strip()}")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Piper did not produce an output wav file.")

def restore_punctuation(text, max_words=120):
    if not PUNCTUATE_TEXT:
        return text
    if "punctuation_model" not in globals() or punctuation_model is None:
        return text
    words = text.strip().lower().split()
    if not words:
        return text

    label_to_punct = {
        "O": "",
        "COMMA": ",",
        "PERIOD": ".",
        "QUESTION": "?",
        "EXCLAMATION": "!",
        "COLON": ":",
        "SEMICOLON": ";",
    }

    def process_chunk(chunk_words, capitalize_next):
        inputs = punctuation_tokenizer(
            chunk_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        with torch.no_grad():
            logits = punctuation_model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        word_ids = inputs.word_ids()
        last_word = -1
        word_end_labels = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != last_word:
                last_word = word_id
            word_end_labels[word_id] = pred_ids[idx]

        decoded = []
        for i, word in enumerate(chunk_words):
            label_id = word_end_labels.get(i)
            label = punctuation_model.config.id2label.get(label_id, "O")
            punct = label_to_punct.get(label, "")
            if capitalize_next and word:
                word = word[0].upper() + word[1:]
                capitalize_next = False
            decoded.append(word + punct)
            if punct in {".", "?", "!"}:
                capitalize_next = True
        return " ".join(decoded), capitalize_next

    out_parts = []
    capitalize_next = True
    for i in range(0, len(words), max_words):
        chunk = words[i:i + max_words]
        chunk_text, capitalize_next = process_chunk(chunk, capitalize_next)
        out_parts.append(chunk_text)
    return " ".join(out_parts).strip()

def normalize_tech_text(text):
    """
    Normalize spoken "tech" tokens (dot/com/slash/etc.) into symbols.
    Intended for wav2vec2 output; Whisper already handles this better.
    """
    normalized = text

    # Common domain suffixes
    normalized = re.sub(r"\bdot com\b", ".com", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot come\b", ".com", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot comm\b", ".com", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot net\b", ".net", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot org\b", ".org", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot io\b", ".io", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot ai\b", ".ai", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot co\b", ".co", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot uk\b", ".uk", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot dev\b", ".dev", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdot local\b", ".local", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\.\\s+(com|net|org|io|ai|co|uk|dev|local)\\b", r".\\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(\\w)\\s+\\.(com|net|org|io|ai|co|uk|dev|local)\\b", r"\\1.\\2", normalized, flags=re.IGNORECASE)

    # Symbols between tokens
    normalized = re.sub(r"(?<=\\w)\\s+dot\\s+(?=\\w)", ".", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\\w)\\s+at\\s+(?=\\w)", "@", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\\w)\\s+colon\\s+(?=\\w)", ":", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\\w)\\s+dash\\s+(?=\\w)", "-", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<=\\w)\\s+hyphen\\s+(?=\\w)", "-", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bhyphen\\b", "-", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bunderscore\\b", "_", normalized, flags=re.IGNORECASE)

    # Slashes
    normalized = re.sub(r"\\bback\\s+slash\\b", r"\\\\", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bbackslash\\b", r"\\\\", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bbash\\b", r"\\\\", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bforward\\s+slash\\b", "/", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bslash\\b", "/", normalized, flags=re.IGNORECASE)

    # Spoken punctuation tokens
    normalized = re.sub(r"\\bcomma\\b", ",", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bperiod\\b", ".", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bquestion\\s+mark\\b", "?", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bexclamation\\s+point\\b", "!", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bexclamation\\s+mark\\b", "!", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\bhash\\b", "#", normalized, flags=re.IGNORECASE)

    # Collapse sequences of spoken digits into numbers (useful for IPs/ports).
    num_map = {
        "zero": "0",
        "oh": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    parts = normalized.split()
    out = []
    buffer = []
    for token in parts:
        lower = token.lower()
        if lower in num_map:
            buffer.append(num_map[lower])
            continue
        if lower == ".":
            buffer.append(".")
            continue
        if lower == "dot":
            buffer.append(".")
            continue
        if buffer:
            out.append("".join(buffer))
            buffer = []
        out.append(token)
    if buffer:
        out.append("".join(buffer))
    normalized = " ".join(out)

    return normalized

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
    request_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()
    logger.info(f"[{request_id}] /generate_audio request received")
    acquired = global_lock.acquire(timeout=REQUEST_LOCK_TIMEOUT_SEC)
    if not acquired:
        logger.warning(
            f"[{request_id}] /generate_audio lock wait exceeded {REQUEST_LOCK_TIMEOUT_SEC:.0f}s"
        )
        return jsonify({
            "status": "error",
            "message": "Service busy: previous request still running",
        }), 503
    try:
        logger.info(f"[{request_id}] /generate_audio lock acquired")
        try:
            ensure_models_for_tts()
            data = request.get_json(silent=True) or {}
            text = data.get("text")

            validate_text_input(text)

            # Use legacy preprocessing for Kokoro ONNX, but raw text for Piper.
            text_for_tts = text if kokoro_tts_mode == "piper" else preprocess_all(text)
            if kokoro_tts_mode == "piper":
                cache_fingerprint = f"mode=piper|model={PIPER_MODEL_PATH}|config={PIPER_CONFIG_PATH}|speaker={KOKORO_SPEAKER_ID}"
            elif kokoro_tts_mode == "legacy":
                cache_fingerprint = f"mode=legacy|voice={voice_name}"
            else:
                cache_fingerprint = (
                    f"mode=stts2|speaker={KOKORO_SPEAKER_ID}|"
                    f"noise={KOKORO_STTS2_NOISE_SCALE}|len={KOKORO_STTS2_LENGTH_SCALE}|noisew={KOKORO_STTS2_NOISE_W}"
                )
            text_hash = hashlib.sha256(f"{cache_fingerprint}|{text_for_tts}".encode('utf-8')).hexdigest()
            filename = f"{text_hash}.wav"
            cached_file_path = os.path.join(SERVE_DIR, filename)

            # Cache hit
            if is_cached(cached_file_path):
                logger.info(f"[{request_id}] Returning cached audio: {filename}")
                return jsonify({"status": "success", "filename": filename})

            if kokoro_tts_mode == "piper":
                synthesize_with_piper(text_for_tts, cached_file_path)
            else:
                # Tokenize
                from kokoro import phonemize, tokenize  # lazy import is fine
                tokens = tokenize(phonemize(text_for_tts, 'a'))
                if len(tokens) > KOKORO_MAX_PHONEME_TOKENS:
                    logger.warning(f"Text too long; truncating to {KOKORO_MAX_PHONEME_TOKENS} tokens.")
                    tokens = tokens[:KOKORO_MAX_PHONEME_TOKENS]

                if kokoro_tts_mode == "legacy":
                    tokens = [[0, *tokens, 0]]
                    ref_s = voice_style[len(tokens[0]) - 2]  # (1,256)
                    ort_inputs = {
                        "input_ids": np.array(tokens, dtype=np.int64),
                        "style": ref_s,
                        "speed": np.ones(1, dtype=np.float32),
                    }
                else:
                    token_ids = np.array([tokens], dtype=np.int64)
                    ort_inputs = {}
                    if "input" in kokoro_input_names:
                        ort_inputs["input"] = token_ids
                    elif "input_ids" in kokoro_input_names:
                        ort_inputs["input_ids"] = token_ids
                    else:
                        first_name = sess.get_inputs()[0].name
                        ort_inputs[first_name] = token_ids

                    if "input_lengths" in kokoro_input_names:
                        ort_inputs["input_lengths"] = np.array([token_ids.shape[1]], dtype=np.int64)
                    if "ids" in kokoro_input_names:
                        ort_inputs["ids"] = np.array([KOKORO_SPEAKER_ID], dtype=np.int64)
                    if "sid" in kokoro_input_names:
                        ort_inputs["sid"] = np.array([KOKORO_SPEAKER_ID], dtype=np.int64)
                    if "scales" in kokoro_input_names:
                        ort_inputs["scales"] = np.array(
                            [KOKORO_STTS2_NOISE_SCALE, KOKORO_STTS2_LENGTH_SCALE, KOKORO_STTS2_NOISE_W],
                            dtype=np.float32,
                        )
                    if "speed" in kokoro_input_names:
                        ort_inputs["speed"] = np.ones(1, dtype=np.float32)

                audio = sess.run(None, ort_inputs)[0]

                # Save
                audio = np.squeeze(audio).astype(np.float32)
                sf.write(cached_file_path, audio, 24000)

            elapsed = time.monotonic() - start_time
            logger.info(f"[{request_id}] Audio saved: {cached_file_path} ({elapsed:.2f}s)")
            return jsonify({"status": "success", "filename": filename})
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.exception(f"[{request_id}] Error generating audio after {elapsed:.2f}s: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        global_lock.release()
        elapsed = time.monotonic() - start_time
        logger.info(f"[{request_id}] /generate_audio completed ({elapsed:.2f}s)")

# Speech-to-Text (S2T) Endpoint
@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """Speech-to-Text (S2T) Endpoint with automatic format conversion"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()
    logger.info(f"[{request_id}] /transcribe_audio request received")
    acquired = global_lock.acquire(timeout=REQUEST_LOCK_TIMEOUT_SEC)
    if not acquired:
        logger.warning(
            f"[{request_id}] /transcribe_audio lock wait exceeded {REQUEST_LOCK_TIMEOUT_SEC:.0f}s"
        )
        return jsonify({
            "status": "error",
            "message": "Service busy: previous request still running",
        }), 503
    input_audio_path = None
    converted_audio_path = None
    try:
        logger.info(f"[{request_id}] /transcribe_audio lock acquired")
        try:
            ensure_models_for_asr()
            if 'file' not in request.files:
                logger.warning(f"[{request_id}] No audio file part in request")
                return jsonify({"status": "error", "message": "Missing file field 'file'"}), 400
            file = request.files['file']
            validate_audio_file(file)
            logger.info(
                f"[{request_id}] STT input accepted: name={file.filename!r} "
                f"content_type={file.content_type!r}"
            )
            
            # Create temporary files for both input and output
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as input_temp:
                input_audio_path = input_temp.name
                file.save(input_audio_path)
                input_size = os.path.getsize(input_audio_path)
                logger.info(f"[{request_id}] Input audio saved to {input_audio_path} ({input_size} bytes)")
            
            # Create a temporary file for the converted WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_temp:
                converted_audio_path = output_temp.name
            
            # Convert to WAV with ffmpeg (16kHz, mono)
            logger.info(f"[{request_id}] FFmpeg conversion started (timeout={FFMPEG_TIMEOUT_SEC:.0f}s)")
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
            try:
                ffmpeg_start = time.monotonic()
                result = subprocess.run(
                    conversion_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=FFMPEG_TIMEOUT_SEC,
                )
                ffmpeg_elapsed = time.monotonic() - ffmpeg_start
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"FFmpeg conversion timed out after {FFMPEG_TIMEOUT_SEC:.0f}s"
                ) from e

            if result.returncode != 0:
                logger.error(f"FFmpeg conversion error: {result.stderr}")
                raise Exception(f"Audio conversion failed: {result.stderr}")
            
            converted_size = os.path.getsize(converted_audio_path)
            logger.info(
                f"[{request_id}] FFmpeg conversion complete in {ffmpeg_elapsed:.2f}s "
                f"-> {converted_audio_path} ({converted_size} bytes)"
            )
            
            # Load and process the converted audio
            logger.info(f"[{request_id}] Loading converted audio for ASR")
            audio_array, sampling_rate = librosa.load(converted_audio_path, sr=16000)
            duration_sec = (len(audio_array) / sampling_rate) if sampling_rate else 0.0
            logger.info(
                f"[{request_id}] ASR input ready: samples={len(audio_array)} "
                f"rate={sampling_rate} duration={duration_sec:.2f}s engine={ASR_ENGINE}"
            )

            if ASR_ENGINE == "wav2vec2_onnx" and 'asr_session' in globals() and asr_session is not None:
                asr_start = time.monotonic()
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
                asr_elapsed = time.monotonic() - asr_start
                logger.info(
                    f"[{request_id}] ASR (Wav2Vec2 ONNX) complete in {asr_elapsed:.2f}s "
                    f"(chars={len(transcription)})"
                )
            else:
                asr_start = time.monotonic()
                # Whisper fallback
                input_features = processor(
                    audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                ).input_features

                logger.debug("Generating transcription (Whisper)...")
                predicted_ids = whisper_model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                asr_elapsed = time.monotonic() - asr_start
                logger.info(
                    f"[{request_id}] ASR (Whisper) complete in {asr_elapsed:.2f}s "
                    f"(chars={len(transcription)})"
                )

            if PUNCTUATE_TEXT:
                try:
                    transcription = restore_punctuation(transcription)
                    logger.info(f"Transcription (Punctuated): {transcription}")
                except Exception as pe:
                    logger.warning(f"Punctuation restore failed: {pe}")

            if TECH_NORMALIZE:
                try:
                    transcription = normalize_tech_text(transcription)
                    logger.info(f"Transcription (Normalized): {transcription}")
                except Exception as ne:
                    logger.warning(f"Tech normalization failed: {ne}")

            preview = transcription[:160].replace("\n", " ").strip()
            if len(transcription) > 160:
                preview += "..."
            logger.info(f"[{request_id}] STT transcription preview: {preview}")
            return jsonify({"status": "success", "transcription": transcription})
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.exception(f"[{request_id}] Error transcribing audio after {elapsed:.2f}s: {str(e)}")
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
    finally:
        global_lock.release()
        elapsed = time.monotonic() - start_time
        logger.info(f"[{request_id}] /transcribe_audio completed ({elapsed:.2f}s)")

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
