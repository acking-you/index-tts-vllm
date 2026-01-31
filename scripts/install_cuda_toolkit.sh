#!/usr/bin/env bash
set -euo pipefail

# Install/upgrade CUDA Toolkit (nvcc) to support modern architectures like compute_120.
# This script modifies the system via apt and therefore requires sudo.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-13-0}"

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null || [[ -n "${WSL_DISTRO_NAME:-}" ]]
}

ubuntu_repo_slug() {
  if is_wsl; then
    echo "wsl-ubuntu"
    return 0
  fi

  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    case "${VERSION_ID:-}" in
      24.04) echo "ubuntu2404" ;;
      22.04) echo "ubuntu2204" ;;
      20.04) echo "ubuntu2004" ;;
      *) echo "ubuntu2404" ;;
    esac
    return 0
  fi

  echo "ubuntu2404"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' not found." >&2
    exit 1
  fi
}

require_cmd sudo

echo "[cuda] This will install CUDA Toolkit via apt (requires sudo)." >&2
sudo -v

echo "[cuda] Installing prerequisites..." >&2
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  build-essential

REPO_SLUG="$(ubuntu_repo_slug)"
KEYRING_DEB_URL="https://developer.download.nvidia.com/compute/cuda/repos/${REPO_SLUG}/x86_64/cuda-keyring_1.1-1_all.deb"

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

echo "[cuda] Adding NVIDIA CUDA apt repo: ${REPO_SLUG}" >&2
curl -fsSL "$KEYRING_DEB_URL" -o "$tmpdir/cuda-keyring.deb"
sudo dpkg -i "$tmpdir/cuda-keyring.deb"

echo "[cuda] apt-get update..." >&2
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y

echo "[cuda] Installing: ${CUDA_TOOLKIT_PACKAGE}" >&2
if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$CUDA_TOOLKIT_PACKAGE"; then
  echo "[cuda] Failed to install '${CUDA_TOOLKIT_PACKAGE}', trying 'cuda-toolkit'..." >&2
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit
fi

echo "[cuda] Done." >&2
echo "[cuda] Tip: use /usr/local/cuda/bin/nvcc (or export PATH=/usr/local/cuda/bin:\$PATH)." >&2

