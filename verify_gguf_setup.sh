#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/audioservice/code/services/kokoro"
MODEL_DIR="/home/audioservice/models/LFM2.5-Audio-1.5B-GGUF"
RUNNER_DIR="/home/audioservice/gguf-runners"
SERVICE_DIR="/etc/systemd/system"
GGUF_HOST="127.0.0.1"
GGUF_PORT="8080"

fail() { echo "ERROR: $*" >&2; exit 1; }

# Check systemd units
[ -f "$SERVICE_DIR/gguf_server.service" ] || fail "gguf_server.service missing"
[ -f "$SERVICE_DIR/audio_server_gguf.service" ] || fail "audio_server_gguf.service missing"

# Verify unit file syntax
systemd-analyze verify "$SERVICE_DIR/gguf_server.service" "$SERVICE_DIR/audio_server_gguf.service"

# Check model files
[ -f "$MODEL_DIR/LFM2.5-Audio-1.5B-Q4_0.gguf" ] || fail "model gguf missing"
[ -f "$MODEL_DIR/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" ] || fail "mmproj gguf missing"
[ -f "$MODEL_DIR/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" ] || fail "vocoder gguf missing"
[ -f "$MODEL_DIR/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" ] || fail "tokenizer gguf missing"

# Check runner binary
BIN=$(find "$RUNNER_DIR" -type f -name 'llama-liquid-audio-server' | head -n 1 || true)
[ -n "$BIN" ] || fail "llama-liquid-audio-server not found"
[ -x "$BIN" ] || fail "llama-liquid-audio-server not executable"

echo "All checks passed."
