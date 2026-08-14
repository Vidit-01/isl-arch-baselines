#!/usr/bin/env bash
# Cloud GPU pipeline: clone repo → 8-word HF dataset → landmarks → train/eval
# every arch.md baseline → write comparison report.
#
# Linux VM (RunPod / Lambda / Colab terminal):
#   git clone https://github.com/Vidit-01/isl-arch-baselines.git
#   cd isl-arch-baselines
#   bash scripts/run_pipeline_baselines.sh
#
# Optional env:
#   REPO_URL   HF_DATASET  HF_TOKEN  WORKDIR  MODELS  EPOCHS  SMOKE=1
#   SKIP_CLONE=1  SKIP_HF_DOWNLOAD=1  SKIP_LANDMARKS=1  SKIP_TRAIN=1

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Vidit-01/isl-arch-baselines.git}"
HF_DATASET="${HF_DATASET:-vidit031/isl-isolated-8words}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_FRAMES="${NUM_FRAMES:-30}"
GIT_BRANCH="${GIT_BRANCH:-main}"

echo "=== ISL arch.md baselines pipeline ==="

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found (CPU training will be slow)"
fi

# System libs MediaPipe / OpenCV need on a bare Ubuntu image
if command -v apt-get >/dev/null 2>&1 && [[ "${SKIP_APT:-0}" != "1" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update -y
    apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0
  else
    sudo apt-get update -y || true
    sudo apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 || true
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If we are already inside a checkout, stay there unless WORKDIR is set.
if [[ -n "${WORKDIR:-}" ]]; then
  :
elif [[ -f "$REPO_ROOT/models/baselines/train.py" ]]; then
  WORKDIR="$REPO_ROOT"
  SKIP_CLONE="${SKIP_CLONE:-1}"
else
  WORKDIR="${HOME}/isl-arch-baselines"
fi

PY_ARGS=(--repo-url "$REPO_URL" --workdir "$WORKDIR" --hf-dataset "$HF_DATASET" --num-frames "$NUM_FRAMES")

if [[ -n "${GIT_BRANCH:-}" ]]; then
  PY_ARGS+=(--branch "$GIT_BRANCH")
fi
if [[ "${SKIP_CLONE:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-clone)
fi
if [[ "${SKIP_HF_DOWNLOAD:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-download)
fi
if [[ "${SKIP_LANDMARKS:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-landmarks)
fi
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  PY_ARGS+=(--skip-train)
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  PY_ARGS+=(--smoke)
fi
if [[ -n "${EPOCHS:-}" ]]; then
  PY_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_ARR=($MODELS)
  PY_ARGS+=(--models "${MODEL_ARR[@]}")
fi

# Prefer the copy inside WORKDIR after clone; fall back to this tree
RUNNER="$REPO_ROOT/scripts/run_pipeline_baselines.py"
if [[ -f "$WORKDIR/scripts/run_pipeline_baselines.py" ]]; then
  RUNNER="$WORKDIR/scripts/run_pipeline_baselines.py"
fi

"$PYTHON_BIN" "$RUNNER" "${PY_ARGS[@]}"
