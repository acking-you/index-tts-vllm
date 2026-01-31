#!/usr/bin/env bash
set -euo pipefail

# Install system dependencies used by this project.
# This script modifies the system via apt and therefore requires sudo.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sudo >/dev/null 2>&1; then
  echo "Error: sudo not found." >&2
  exit 1
fi

echo "[sys] Installing system dependencies via apt (requires sudo)..." >&2
sudo -v

sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  libsndfile1 \
  build-essential \
  ca-certificates \
  curl

echo "[sys] Done." >&2
