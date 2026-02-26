#!/usr/bin/env bash
set -euo pipefail

run_root() {
  if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Need root privileges for: $*" >&2
    return 1
  fi
}

install_system_deps_linux() {
  echo "Installing Linux system dependencies (apt)..."
  run_root apt-get update
  run_root apt-get install -y espeak libsndfile1 ffmpeg curl tar
}

install_piper_linux() {
  local arch url tarball tmpdir
  arch="$(uname -m)"
  case "$arch" in
    x86_64) url="https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz" ;;
    aarch64|arm64) url="https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz" ;;
    *)
      echo "Unsupported architecture for Piper: $arch"
      return 1
      ;;
  esac

  echo "Installing Piper runtime from $url"
  tarball="$(mktemp /tmp/piper.XXXXXX.tar.gz)"
  tmpdir="$(mktemp -d /tmp/piper_extract.XXXXXX)"
  trap 'rm -f "$tarball"; rm -rf "$tmpdir"' RETURN

  curl -fL "$url" -o "$tarball"
  tar -xzf "$tarball" -C "$tmpdir"
  test -d "$tmpdir/piper"

  run_root rm -rf /opt/piper
  run_root cp -a "$tmpdir/piper" /opt/piper
  run_root chown -R root:root /opt/piper
  run_root bash -lc 'cat >/usr/local/bin/piper << "EOF"
#!/usr/bin/env bash
exec /opt/piper/piper "$@"
EOF
chmod 0755 /usr/local/bin/piper'

  /usr/local/bin/piper --help >/dev/null
  echo "Piper installed at /usr/local/bin/piper"
}

OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
  read -r -p "Install Linux system dependencies (apt)? [Y/n]: " install_sys
  install_sys="${install_sys:-Y}"
  case "${install_sys,,}" in
    y|yes) install_system_deps_linux ;;
    *) echo "Skipping system package install." ;;
  esac

  read -r -p "Install Piper globally (/opt/piper + /usr/local/bin/piper)? [Y/n]: " install_piper
  install_piper="${install_piper:-Y}"
  case "${install_piper,,}" in
    y|yes) install_piper_linux ;;
    *) echo "Skipping Piper install." ;;
  esac
else
  echo "Non-Linux OS detected; skipping apt and Piper global install."
fi

python3 install_dependencies.py
