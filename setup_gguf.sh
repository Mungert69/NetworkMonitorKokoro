#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/audioservice/code/services/kokoro"
VENV_PY="$APP_DIR/venv/bin/python3"
MODEL_DIR="/home/audioservice/models/LFM2.5-Audio-1.5B-GGUF"
RUNNER_DIR="/home/audioservice/gguf-runners"
SERVICE_DIR="/etc/systemd/system"
GGUF_HOST="127.0.0.1"
GGUF_PORT="8080"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ZIP_NAME="llama-liquid-audio-ubuntu-x64.zip" ;;
  aarch64|arm64) ZIP_NAME="llama-liquid-audio-ubuntu-arm64.zip" ;;
  *)
    echo "Unsupported arch: $ARCH" >&2
    exit 1
    ;;
esac

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "Please run as root: sudo $0" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y unzip
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

# Install huggingface-cli
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  python3 -m pip install -U --break-system-packages huggingface_hub
fi

if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  HF_CLI="python3 -m huggingface_hub.cli.hf"
fi

# Install OpenAI client in app venv (for GGUF backend)
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" -m pip install -U openai
else
  python3 -m pip install -U --break-system-packages openai
fi

mkdir -p "$MODEL_DIR" "$RUNNER_DIR"

# Download GGUF model files
${HF_CLI} download LiquidAI/LFM2.5-Audio-1.5B-GGUF \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False \
  --include "LFM2.5-Audio-1.5B-Q4_0.gguf" \
  --include "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  --include "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  --include "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"

# Download runner zip
${HF_CLI} download LiquidAI/LFM2.5-Audio-1.5B-GGUF \
  --local-dir "$RUNNER_DIR" \
  --local-dir-use-symlinks False \
  --include "runners/${ZIP_NAME}"

# Extract runner
unzip -o "$RUNNER_DIR/runners/${ZIP_NAME}" -d "$RUNNER_DIR"

# Locate server binary
GGUF_BIN="$(find "$RUNNER_DIR" -type f -name 'llama-liquid-audio-server' | head -n 1)"
if [ -z "$GGUF_BIN" ]; then
  echo "llama-liquid-audio-server not found after extraction" >&2
  exit 1
fi

# Write gguf_server.service
cat > "$SERVICE_DIR/gguf_server.service" <<EOF_SERVICE
[Unit]
Description=GGUF Audio Server (llama-liquid-audio-server)
After=network.target

[Service]
Type=simple
User=audioservice
WorkingDirectory=/home/audioservice
Environment=GGUF_MODEL=$MODEL_DIR/LFM2.5-Audio-1.5B-Q4_0.gguf
Environment=GGUF_MMPROJ=$MODEL_DIR/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf
Environment=GGUF_VOCODER=$MODEL_DIR/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf
Environment=GGUF_TOKENIZER=$MODEL_DIR/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf
Environment=GGUF_HOST=$GGUF_HOST
Environment=GGUF_PORT=$GGUF_PORT
ExecStart=/bin/sh -lc '$GGUF_BIN -m "${GGUF_MODEL}" -mm "${GGUF_MMPROJ}" -mv "${GGUF_VOCODER}" --tts-speaker-file "${GGUF_TOKENIZER}" --host "${GGUF_HOST}" --port "${GGUF_PORT}"'
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF_SERVICE

# Write audio_server_gguf.service
cat > "$SERVICE_DIR/audio_server_gguf.service" <<EOF_SERVICE
[Unit]
Description=Audio Server (GGUF)
After=network.target gguf_server.service
Wants=gguf_server.service

[Service]
ExecStart=$APP_DIR/venv/bin/python3 $APP_DIR/app.py
WorkingDirectory=$APP_DIR
Restart=always
User=audioservice
Environment=PYTHONUNBUFFERED=1
Nice=-10
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=1
Environment=SERVE_DIR=/home/audioservice/app/files
Environment=USE_WAV2VEC2=0
Environment=TTS_BACKEND=gguf
Environment=ASR_BACKEND=gguf
Environment=GGUF_SERVER_URL=http://$GGUF_HOST:$GGUF_PORT/v1

[Install]
WantedBy=multi-user.target
EOF_SERVICE

systemctl daemon-reload
systemctl enable --now gguf_server.service
systemctl enable --now audio_server_gguf.service

echo "GGUF setup complete."
