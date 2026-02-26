# NetworkMonitorKokoro

## Overview

NetworkMonitorKokoro is a Flask-based service that provides advanced text-to-speech (T2S) and speech-to-text (S2T) functionalities using multiple model backends:

- **Text-to-Speech (T2S)**:
  - Kokoro ONNX (legacy and StyleTTS2-like ONNX signatures)
  - Piper-compatible ONNX voices (for example HAL-9000)
- **Speech-to-Text (S2T)**: Transcribes audio files into text using OpenAI's Whisper model.

This repository leverages ONNX for efficient inference and Hugging Face's model hub for seamless model downloads.

You can see the script in action with the Quantum Network Monitor Assistant at [https://freenetworkmonitor.click](https://freenetworkmonitor.click).

---

## Features

- **T2S (Text-to-Speech)**
  - High-quality voice synthesis using Kokoro ONNX (`KOKORO_ONNX_MODE=legacy` or `stts2`).
  - Piper voice synthesis mode (`KOKORO_ONNX_MODE=piper`) for models like HAL-9000.
  - Configurable speaker/style controls depending on backend.

- **S2T (Speech-to-Text)**
  - Default: CPU-friendly ONNX CTC pipeline using Facebook Wav2Vec2 (facebook/wav2vec2-base-960h).
  - Fallback: OpenAI Whisper (PyTorch) when enabled.
  - Handles a wide range of audio inputs.

- **Automatic Model Management**
  - Models are automatically downloaded from the Hugging Face Hub if not present locally.

- **Flask API Endpoints**
  - `/generate_audio`: Convert text into speech and cache by text hash.
  - `/transcribe_audio`: Transcribe uploaded audio; auto-converts to 16kHz WAV via ffmpeg.
  - `/files/<filename>`: Serve generated `.wav` files from the serve directory.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+ (Check with `python3 --version` or `python --version`)
- Pip (Check with `pip --version`)
- A CUDA-enabled GPU (optional, for faster inference)
- **System Dependencies** (Debian/Ubuntu):
  ```bash
  sudo apt-get install libsndfile1 espeak-ng
  ```
- **System Dependencies** (Windows) choco install libsndfile espeak-ng -y
- **System Dependencies** (Mac) brew install libsndfile espeak-ng

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NetworkMonitorKokoro.git
   cd NetworkMonitorKokoro
   ```

2. **Create and activate a virtual environment**:
   - **On Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **On Windows**:
     ```bash
     python3 -m venv venv
     venv\Scripts\activate
     ```

   Once activated, you should see `(venv)` at the start of your command prompt, indicating the virtual environment is active.

3. **Install dependencies (choose one path)**:
   - **Path A: you have root/sudo access (recommended)**:
     ```bash
     ./install.sh
     ```
     - Installs Linux system packages (apt), optionally installs Piper globally, then installs Python dependencies.
   - **Path B: no root/sudo access**:
     - Ensure system dependencies from `install.sh` are already present on the machine (`espeak`, `libsndfile1`, `ffmpeg`, `curl`, `tar` on Linux).
     - Then do Python-only setup:
     ```bash
     cd ~/code/services/kokoro
     python3 -m venv venv
     source venv/bin/activate
     python3 install_dependencies.py
     ```

4. **Set up the models**:
   - The Kokoro T2S model is downloaded automatically on first run.
   - In Piper mode, model/config files are also auto-downloaded on first run (default: `campwill/HAL-9000-Piper-TTS`) if missing.
   - For ASR (S2T), the app auto-downloads a ready-made Wav2Vec2 ONNX model on first run. You can override the repo and path via env vars. See "ASR Engines" below.

5. **Install Piper runtime if using `KOKORO_ONNX_MODE=piper`**:
   - `install.sh` prompts to install Piper globally on Linux.
   - Or install manually and set `PIPER_BIN` (default expected path: `/usr/local/bin/piper`).

6. **Configure file serving directory (optional)**:
   - The app writes generated audio to a serve directory. By default it uses `./files`.
   - To customize, set env var before start:
     ```bash
     export SERVE_DIR=/absolute/path/for/generated/audio
     mkdir -p "$SERVE_DIR"
     ```

7. **Start the Flask server directly**:
   ```bash
   python3 app.py
   ```
   - Or set up systemd service for background startup (see "Running as a Linux Service").

8. **Deactivate the virtual environment** (optional):
   ```bash
   deactivate
   ```

---

## Running as a Linux Service

To run **NetworkMonitorKokoro** as a systemd service on Linux, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NetworkMonitorKokoro.git
   cd NetworkMonitorKokoro
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   python3 install_dependencies.py
   ```

4. **Create a systemd service file**:
   ```bash
   sudo nano /etc/systemd/system/networkmonitor-kokoro.service
   ```

   Add the following content:
   ```ini
   [Unit]
   Description=NetworkMonitorKokoro Service
   After=network.target

   [Service]
   User=yourusername
   WorkingDirectory=/path/to/NetworkMonitorKokoro
   ExecStart=/path/to/NetworkMonitorKokoro/venv/bin/python3 /path/to/NetworkMonitorKokoro/app.py
   Restart=always
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   ```

   Replace `/path/to/NetworkMonitorKokoro` with the full path to the directory where the repository was cloned and the virtual environment was created. Replace `yourusername` with your Linux username.

5. **Set proper permissions**:
   ```bash
   sudo chmod 644 /etc/systemd/system/networkmonitor-kokoro.service
   ```

6. **Reload systemd**:
   ```bash
   sudo systemctl daemon-reload
   ```

7. **Start the service**:
   ```bash
   sudo systemctl start networkmonitor-kokoro
   ```

8. **Enable the service to start on boot**:
   ```bash
   sudo systemctl enable networkmonitor-kokoro
   ```

9. **Check the service status**:
   ```bash
   sudo systemctl status networkmonitor-kokoro
   ```

---

## Usage

### API Endpoints

#### 1. **Generate Audio**
- **Endpoint**: `/generate_audio`
- **Method**: `POST`
- **Request Body (JSON)**:
  - `text` (string, required): text to synthesize. Max 1024 chars. Input is preprocessed and hashed for caching.
  ```json
  { "text": "Your text here" }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "filename": "<sha256(text)>.wav"
  }
  ```
  - The audio file is saved under the serve directory (`$SERVE_DIR` or `./files`).
  - Fetch the audio via `GET /files/<filename>`.

#### 2. **Transcribe Audio**
- **Endpoint**: `/transcribe_audio`
- **Method**: `POST`
- **Request**:
  - `multipart/form-data` with file field name `file`.
  - Supported containers: WAV, MP3, WebM, Ogg/Opus, etc. The server converts input to 16kHz mono WAV using `ffmpeg` and applies light denoise/normalization filters.
- **Response**:
  ```json
  {
    "status": "success",
    "transcription": "Your transcription here"
  }
  ```

---

## ASR Engines

- Engine selection:
  - Default is Whisper (`whisper_pt`).
  - Set `USE_WAV2VEC2=1` to use the ONNX Wav2Vec2 pipeline.
  - You can also override directly with `ASR_ENGINE=wav2vec2_onnx` or `ASR_ENGINE=whisper_pt`.

- Wav2Vec2 model selection:
  - Default model: `facebook/wav2vec2-base-960h`
  - Override the processor/model with `ASR_MODEL_NAME` if you want a different Wav2Vec2 model.

- Wav2Vec2 ONNX model provisioning:
  - On first run, if `ASR_ONNX_PATH` does not exist, the app downloads a ready-made ONNX model from `ASR_ONNX_REPO` and stores it under `asr_onnx/<model_name>.onnx`.
  - Default: `ASR_ONNX_REPO=onnx-community/wav2vec2-base-960h-ONNX`
  - You can override the source repo via `ASR_ONNX_REPO` or the destination path via `ASR_ONNX_PATH`.
  - The model should take `input_values` (batch, samples) float32 at 16 kHz and output CTC logits `(batch, time, vocab)`.
  - Tokenizer: this app uses `Wav2Vec2Processor` from the matching Facebook model for feature extraction and decoding.

- Optional punctuation restoration:
  - Enable with `PUNCTUATE_TEXT=1` (default off).
  - Default model: `PUNCTUATION_MODEL=kredor/punctuate-all`.
  - This runs after ASR to add punctuation and casing to the raw transcript.

- Optional tech normalization:
  - Enable with `TECH_NORMALIZE=1` (default off).
  - Converts spoken tokens like "dot/com/slash/at/colon" to symbols.

- Example run with Wav2Vec2 ONNX:
  ```bash
  export USE_WAV2VEC2=1
  # optional: override processor/model or ONNX repo/path
  # export ASR_MODEL_NAME=facebook/wav2vec2-base-960h
  # optional override of the source repo or ONNX path
  # export ASR_ONNX_REPO=onnx-community/wav2vec2-base-960h-ONNX
  # export ASR_ONNX_PATH=asr_onnx/custom.onnx
  # optional: post-processing
  # export PUNCTUATE_TEXT=1
  # export TECH_NORMALIZE=1
  python3 app.py
  ```

- Example run with Whisper fallback:
  ```bash
  # default is whisper, so no flags needed
  python3 app.py
  ```

## TTS Modes

- `KOKORO_ONNX_MODE=auto` (default):
  - Auto-detects ONNX signature and selects `legacy` or `stts2`.
- `KOKORO_ONNX_MODE=legacy`:
  - Uses classic Kokoro inputs (`input_ids`, `style`, `speed`) and voice `.bin` vectors.
- `KOKORO_ONNX_MODE=stts2`:
  - Uses StyleTTS2-like ONNX inputs (`input`/`input_lengths`, optional `ids`/`sid`, optional `scales`).
- `KOKORO_ONNX_MODE=piper`:
  - Uses Piper CLI to synthesize from a Piper ONNX voice model.
  - Requires Piper binary (`PIPER_BIN`, default `piper`).
  - Uses `PIPER_MODEL_PATH` and `PIPER_CONFIG_PATH`.

### Example: Kokoro mode

```bash
export KOKORO_ONNX_MODE=legacy
python3 app.py
```

### Example: HAL Piper mode

```bash
export KOKORO_ONNX_MODE=piper
export PIPER_BIN=/usr/local/bin/piper
export PIPER_MODEL_PATH=kokoro_model/onnx/model.onnx
export PIPER_CONFIG_PATH=kokoro_model/onnx/model.onnx.json
python3 app.py
```

#### 3. **Serve Audio Files**
- **Endpoint**: `/files/<filename>`
- **Method**: `GET`
- **Notes**:
  - Only serves `.wav` files located in the serve directory.
  - Returns 404 if the file does not exist.

---

## Operational Details

- Concurrency: the app uses a global lock so only one synthesis/transcription runs at a time. Internal math libraries and ONNX Runtime are also constrained to a small thread count (`MAX_THREADS=2`).
- Models:
  - T2S supports both Kokoro ONNX and Piper voices:
    - Kokoro default source: `onnx-community/Kokoro-82M-v1.0-ONNX`
    - Piper default source (auto-download when missing): `campwill/HAL-9000-Piper-TTS`
  - S2T defaults to Whisper and can be switched to Wav2Vec2 ONNX with `USE_WAV2VEC2=1`.
- Lazy model loading:
  - Default is lazy (`MODEL_INIT_MODE=lazy`): TTS assets load on first `/generate_audio`, ASR assets load on first `/transcribe_audio`.
  - Set `MODEL_INIT_MODE=startup` to preload everything at app start.
- Caching: generated audio filenames are content-addressed and include backend/mode fingerprinting to avoid collisions across TTS backends.
- File location: generated WAV files are saved to `$SERVE_DIR` (default `./files`).

---

## Example Requests

- Generate audio:
  ```bash
  curl -sX POST http://localhost:7860/generate_audio \
    -H 'Content-Type: application/json' \
    -d '{"text":"Hello from Kokoro."}'
  # => {"status":"success","filename":"<hash>.wav"}
  curl -O http://localhost:7860/files/<hash>.wav
  ```

- Transcribe audio:
  ```bash
  curl -sX POST http://localhost:7860/transcribe_audio \
    -F file=@sample.webm
  # => {"status":"success","transcription":"..."}
  ```

### Example Requests (alternate form)

#### Generate Audio
```bash
curl -sX POST \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, world!"}' \
  http://127.0.0.1:7860/generate_audio
```

#### Transcribe Audio
```bash
curl -sX POST \
  -F "file=@sample_audio.wav" \
  http://127.0.0.1:7860/transcribe_audio
```

---

## Dependencies

- [Transformers](https://huggingface.co/transformers)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Librosa](https://librosa.org/)
- [SoundFile](https://pysoundfile.readthedocs.io/)
- [Flask](https://flask.palletsprojects.com/)
- [Flask-CORS](https://flask-cors.readthedocs.io/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Hugging Face for providing pre-trained models.
- OpenAI for the Whisper model.
- ONNX for efficient inference.

---

## Contact

For questions or support, please open an issue or contact [support@mahadeva.co.uk](mailto:support@mahadeva.co.uk).
