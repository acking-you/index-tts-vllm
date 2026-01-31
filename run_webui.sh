#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
One-click run (uv) for IndexTTS WebUI.

Usage:
  ./run_webui.sh [--version 1.5] [--host 0.0.0.0] [--port 6006] [--model-dir /path/to/model] [--no-download] [--no-cuda-kernel]
  ./run_webui.sh --version 2 --qwenemo-gpu-memory-utilization 0.20
  ./run_webui.sh --cuda-kernel --install-cuda-toolkit
  ./run_webui.sh --install-system-deps

Env (optional):
  VERSION=1.5
  HOST=0.0.0.0
  PORT=6006
  GPU_MEMORY_UTILIZATION=0.25
  QWENEMO_GPU_MEMORY_UTILIZATION=0.20
  MODEL_DIR=/abs/or/relative/path
  DOWNLOAD_MODEL=1        # 1=auto download via modelscope if missing
  INDEXTTS_USE_CUDA_KERNEL=0|1  # enable/disable BigVGAN CUDA extension
  CUDA_TOOLKIT_PACKAGE=cuda-toolkit-13-0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

VERSION="${VERSION:-1.5}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
QWENEMO_GPU_MEMORY_UTILIZATION="${QWENEMO_GPU_MEMORY_UTILIZATION:-}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
MODEL_DIR="${MODEL_DIR:-}"
INDEXTTS_USE_CUDA_KERNEL="${INDEXTTS_USE_CUDA_KERNEL:-}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-13-0}"
INSTALL_CUDA_TOOLKIT=0
INSTALL_SYSTEM_DEPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:?missing value for --version}"; shift 2 ;;
    --host)
      HOST="${2:?missing value for --host}"; shift 2 ;;
    --port)
      PORT="${2:?missing value for --port}"; shift 2 ;;
    --model-dir)
      MODEL_DIR="${2:?missing value for --model-dir}"; shift 2 ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="${2:?missing value for --gpu-memory-utilization}"; shift 2 ;;
    --qwenemo-gpu-memory-utilization)
      QWENEMO_GPU_MEMORY_UTILIZATION="${2:?missing value for --qwenemo-gpu-memory-utilization}"; shift 2 ;;
    --no-download)
      DOWNLOAD_MODEL=0; shift ;;
    --no-cuda-kernel)
      INDEXTTS_USE_CUDA_KERNEL=0; shift ;;
    --cuda-kernel)
      INDEXTTS_USE_CUDA_KERNEL=1; shift ;;
    --install-cuda-toolkit)
      INSTALL_CUDA_TOOLKIT=1; shift ;;
    --install-system-deps)
      INSTALL_SYSTEM_DEPS=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "[bootstrap] uv not found, installing..." >&2
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Error: cannot install uv (need curl or wget)." >&2
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo "Error: uv install finished but uv not in PATH." >&2; exit 1; }
}

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return 0
  fi
  echo "[warn] ffmpeg not found. Some audio features may not work." >&2
  echo "[hint] Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
}

model_id_for_version() {
  case "$1" in
    1|1.0|v1|v1.0)
      echo "kusuriuri/Index-TTS-vLLM" ;;
    1.5|v1.5)
      echo "kusuriuri/Index-TTS-1.5-vLLM" ;;
    2|2.0|v2|v2.0)
      echo "kusuriuri/IndexTTS-2-vLLM" ;;
    *)
      echo "" ;;
  esac
}

default_model_dir_for_version() {
  case "$1" in
    1|1.0|v1|v1.0)
      echo "$ROOT_DIR/checkpoints/Index-TTS-vLLM" ;;
    1.5|v1.5)
      echo "$ROOT_DIR/checkpoints/Index-TTS-1.5-vLLM" ;;
    2|2.0|v2|v2.0)
      echo "$ROOT_DIR/checkpoints/IndexTTS-2-vLLM" ;;
    *)
      echo "" ;;
  esac
}

ensure_uv

if (( INSTALL_SYSTEM_DEPS == 1 )); then
  if [[ -x "$ROOT_DIR/scripts/install_system_deps.sh" ]]; then
    "$ROOT_DIR/scripts/install_system_deps.sh"
  else
    echo "Error: missing $ROOT_DIR/scripts/install_system_deps.sh" >&2
    exit 1
  fi
fi

ensure_ffmpeg

echo "[bootstrap] uv sync..." >&2
if [[ -f "uv.lock" ]]; then
  uv sync --frozen
else
  uv sync
fi

prefer_cuda_home() {
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:$PATH"
  fi
}

nvcc_supports_sm120() {
  local nvcc_bin="$1"
  local tmp_cu tmp_o
  tmp_cu="$(mktemp --suffix=.cu)"
  tmp_o="$(mktemp --suffix=.o)"
  cat >"$tmp_cu" <<'CU'
__global__ void k() {}
int main() { k<<<1,1>>>(); return 0; }
CU
  if "$nvcc_bin" -gencode arch=compute_120,code=sm_120 -c "$tmp_cu" -o "$tmp_o" >/dev/null 2>&1; then
    rm -f "$tmp_cu" "$tmp_o"
    return 0
  fi
  rm -f "$tmp_cu" "$tmp_o"
  return 1
}

install_cuda_toolkit_if_needed() {
  if (( INSTALL_CUDA_TOOLKIT != 1 )); then
    return 0
  fi
  if [[ ! -x "$ROOT_DIR/scripts/install_cuda_toolkit.sh" ]]; then
    echo "Error: missing $ROOT_DIR/scripts/install_cuda_toolkit.sh" >&2
    exit 1
  fi
  echo "[cuda] Installing/upgrading CUDA Toolkit (requires sudo)..." >&2
  CUDA_TOOLKIT_PACKAGE="$CUDA_TOOLKIT_PACKAGE" "$ROOT_DIR/scripts/install_cuda_toolkit.sh"
  prefer_cuda_home

  # CUDA upgrades often change supported arch flags; force a clean rebuild for the BigVGAN extension.
  rm -rf "$ROOT_DIR/indextts/BigVGAN/alias_free_activation/cuda/build" 2>/dev/null || true
  rm -rf "/tmp/BigVGAN/cuda/build" 2>/dev/null || true
}

# The BigVGAN CUDA extension is optional. On new GPUs (e.g. compute capability 12.0),
# an older CUDA Toolkit (nvcc) may not support the required arch and will fail with:
#   nvcc fatal: Unsupported gpu architecture 'compute_120'
#
# By default, we disable the extension in that case to keep WebUI working.
# Use `--install-cuda-toolkit` to upgrade nvcc and keep the extension enabled.
prefer_cuda_home
if [[ -z "$INDEXTTS_USE_CUDA_KERNEL" || "$INDEXTTS_USE_CUDA_KERNEL" == "1" ]]; then
  cap_major="$(uv run python -c "import torch; print(torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"
  if [[ "$cap_major" =~ ^[0-9]+$ ]] && (( cap_major >= 12 )); then
    if command -v nvcc >/dev/null 2>&1; then
      if ! nvcc_supports_sm120 "$(command -v nvcc)"; then
        install_cuda_toolkit_if_needed
      fi
    else
      install_cuda_toolkit_if_needed
    fi

    if command -v nvcc >/dev/null 2>&1; then
      if ! nvcc_supports_sm120 "$(command -v nvcc)"; then
        if [[ -z "$INDEXTTS_USE_CUDA_KERNEL" ]]; then
          INDEXTTS_USE_CUDA_KERNEL=0
          echo "[warn] nvcc does not support compute_120; disabling BigVGAN CUDA extension (fallback stays correct)." >&2
          echo "[hint] To enable it, run: ./run_webui.sh --cuda-kernel --install-cuda-toolkit" >&2
        elif [[ "$INDEXTTS_USE_CUDA_KERNEL" == "1" ]]; then
          echo "[warn] nvcc does not support compute_120; BigVGAN CUDA extension build will likely fail." >&2
          echo "[hint] Run with: ./run_webui.sh --cuda-kernel --install-cuda-toolkit" >&2
        fi
      fi
    else
      if [[ -z "$INDEXTTS_USE_CUDA_KERNEL" ]]; then
        INDEXTTS_USE_CUDA_KERNEL=0
        echo "[warn] nvcc not found; disabling BigVGAN CUDA extension (fallback stays correct)." >&2
        echo "[hint] To install nvcc, run: ./run_webui.sh --cuda-kernel --install-cuda-toolkit" >&2
      fi
    fi
  fi
fi

if [[ -n "$INDEXTTS_USE_CUDA_KERNEL" ]]; then
  export INDEXTTS_USE_CUDA_KERNEL
  echo "[bootstrap] INDEXTTS_USE_CUDA_KERNEL=$INDEXTTS_USE_CUDA_KERNEL" >&2
fi

if [[ -z "$MODEL_DIR" ]]; then
  MODEL_DIR="$(default_model_dir_for_version "$VERSION")"
fi

MODEL_ID="$(model_id_for_version "$VERSION")"
if [[ -z "$MODEL_DIR" || -z "$MODEL_ID" ]]; then
  echo "Error: unsupported --version '$VERSION' (supported: 1.0 / 1.5 / 2.0)" >&2
  exit 2
fi

if [[ "$DOWNLOAD_MODEL" != "0" ]]; then
  if [[ ! -f "$MODEL_DIR/config.yaml" ]]; then
    mkdir -p "$ROOT_DIR/checkpoints"
    echo "[model] downloading weights: $MODEL_ID -> $MODEL_DIR" >&2
    uv run modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
  fi
fi

if [[ ! -f "$MODEL_DIR/config.yaml" ]]; then
  echo "Error: model not found in '$MODEL_DIR' (missing config.yaml)." >&2
  echo "Hint: re-run with auto-download enabled, or set --model-dir." >&2
  exit 1
fi

if [[ "$VERSION" == "2" || "$VERSION" == "2.0" || "$VERSION" == "v2" || "$VERSION" == "v2.0" ]]; then
  if [[ -z "$QWENEMO_GPU_MEMORY_UTILIZATION" ]]; then
    QWENEMO_GPU_MEMORY_UTILIZATION=0.20
  fi
  echo "[run] webui_v2.py --model_dir \"$MODEL_DIR\" --host \"$HOST\" --port \"$PORT\"" >&2
  uv run python webui_v2.py \
    --model_dir "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --qwenemo_gpu_memory_utilization "$QWENEMO_GPU_MEMORY_UTILIZATION"
else
  echo "[run] webui.py --version \"$VERSION\" --model_dir \"$MODEL_DIR\" --host \"$HOST\" --port \"$PORT\"" >&2
  uv run python webui.py \
    --version "$VERSION" \
    --model_dir "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
fi
