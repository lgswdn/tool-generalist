#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STEREOVLA_ROOT="${STEREOVLA_ROOT:-$REPO_ROOT/thirdparty/StereoVLA}"
PROJECT_MODEL_DIR="${PROJECT_MODEL_DIR:-/mnt/project/world_model/tool_generalist/stereovla_model}"
PORT="${PORT:-6666}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCHING_DELAY="${BATCHING_DELAY:-80}"
COMPILE="${COMPILE:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ ! -d "$STEREOVLA_ROOT" ]]; then
  echo "[ERROR] Missing StereoVLA repo: $STEREOVLA_ROOT" >&2
  exit 1
fi

checkpoint_candidates=()
if [[ -n "${STEREOVLA_MODEL_PATH:-}" ]]; then
  checkpoint_candidates+=("$STEREOVLA_MODEL_PATH")
fi
checkpoint_candidates+=(
  "$PROJECT_MODEL_DIR/stereovla/checkpoint/model.safetensors"
  "$REPO_ROOT/storage/stereovla/checkpoint/model.safetensors"
  "$STEREOVLA_ROOT/../storage/stereovla/checkpoint/model.safetensors"
)

hf_snapshot=""
hf_ref="$HOME/.cache/huggingface/hub/models--shengliangd--StereoVLA/refs/main"
if [[ -f "$hf_ref" ]]; then
  hf_revision="$(cat "$hf_ref")"
  hf_snapshot="$HOME/.cache/huggingface/hub/models--shengliangd--StereoVLA/snapshots/$hf_revision"
  checkpoint_candidates+=("$hf_snapshot/stereovla/checkpoint/model.safetensors")
fi

model_path=""
for candidate in "${checkpoint_candidates[@]}"; do
  if [[ -f "$candidate" ]]; then
    model_path="$(readlink -f "$candidate")"
    break
  fi
done

if [[ -z "$model_path" ]]; then
  echo "[ERROR] Could not find StereoVLA checkpoint model.safetensors." >&2
  echo "[ERROR] Expected one of:" >&2
  for candidate in "${checkpoint_candidates[@]}"; do
    echo "  - $candidate" >&2
  done
  echo "[ERROR] Download it into the default HF cache with:" >&2
  echo "  hf download shengliangd/StereoVLA" >&2
  echo "[ERROR] Or clone it into project storage with:" >&2
  echo "  MODEL_DIR=$PROJECT_MODEL_DIR $REPO_ROOT/download_stereovla_model.sh" >&2
  echo "[ERROR] Or download it into this repo with:" >&2
  echo "  hf download shengliangd/StereoVLA --local-dir $REPO_ROOT/storage" >&2
  echo "[ERROR] Or set STEREOVLA_MODEL_PATH=/path/to/stereovla/checkpoint/model.safetensors" >&2
  exit 1
fi

storage_candidates=()
if [[ -n "${STORAGE_PATH:-}" ]]; then
  storage_candidates+=("$STORAGE_PATH")
fi
storage_candidates+=(
  "$(dirname "$(dirname "$model_path")")"
  "$(dirname "$(dirname "$(dirname "$model_path")")")"
  "$PROJECT_MODEL_DIR"
  "$REPO_ROOT/storage"
  "$STEREOVLA_ROOT/../storage"
)
if [[ -n "$hf_snapshot" ]]; then
  storage_candidates+=("$hf_snapshot")
fi

resolved_storage=""
for candidate in "${storage_candidates[@]}"; do
  if [[ -f "$candidate/ckpt/pretrained/foundation_stereo/model_best_bp2.pth" ]]; then
    resolved_storage="$(readlink -f "$candidate")"
    break
  fi
done

if [[ -z "$resolved_storage" ]]; then
  echo "[ERROR] Could not find STORAGE_PATH with ckpt/pretrained/foundation_stereo/model_best_bp2.pth." >&2
  echo "[ERROR] Download the full StereoVLA repo assets into the default HF cache with:" >&2
  echo "  hf download shengliangd/StereoVLA" >&2
  echo "[ERROR] Or clone it into project storage with:" >&2
  echo "  MODEL_DIR=$PROJECT_MODEL_DIR $REPO_ROOT/download_stereovla_model.sh" >&2
  echo "[ERROR] Or into this repo with:" >&2
  echo "  hf download shengliangd/StereoVLA --local-dir $REPO_ROOT/storage" >&2
  echo "[ERROR] Or set STORAGE_PATH=/path/to/downloaded/StereoVLA/assets" >&2
  exit 1
fi

required_storage_files=(
  "ckpt/pretrained/foundation_stereo/model_best_bp2.pth"
  "ckpt/pretrained/internlm/internlm2-1_8b/config.json"
  "ckpt/pretrained/internlm/internlm2-1_8b/pytorch_model.bin"
)
missing_storage_files=()
for rel_path in "${required_storage_files[@]}"; do
  if [[ ! -f "$resolved_storage/$rel_path" ]]; then
    missing_storage_files+=("$resolved_storage/$rel_path")
  fi
done
if (( ${#missing_storage_files[@]} > 0 )); then
  echo "[ERROR] STORAGE_PATH is incomplete: $resolved_storage" >&2
  echo "[ERROR] Missing required files:" >&2
  for missing in "${missing_storage_files[@]}"; do
    echo "  - $missing" >&2
  done
  echo "[ERROR] Finish the full model download with:" >&2
  echo "  hf download shengliangd/StereoVLA" >&2
  echo "[ERROR] Or clone it into project storage with:" >&2
  echo "  MODEL_DIR=$PROJECT_MODEL_DIR $REPO_ROOT/download_stereovla_model.sh" >&2
  echo "[ERROR] Or use a complete local-dir download:" >&2
  echo "  hf download shengliangd/StereoVLA --local-dir $REPO_ROOT/storage" >&2
  exit 1
fi

compile_args=()
if [[ "$COMPILE" == "1" || "$COMPILE" == "true" || "$COMPILE" == "TRUE" ]]; then
  compile_args+=(--compile)
fi

cd "$STEREOVLA_ROOT"
export PYTHONPATH="$STEREOVLA_ROOT:${PYTHONPATH:-}"
export STORAGE_PATH="$resolved_storage"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/stereo/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/stereo/bin/python"
  elif [[ -x "$STEREOVLA_ROOT/env/bin/python" ]]; then
    PYTHON_BIN="$STEREOVLA_ROOT/env/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] Could not find python. Set PYTHON_BIN=/path/to/python for the StereoVLA environment." >&2
    exit 1
  fi
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 1
fi

echo "[INFO] Starting StereoVLA server"
echo "[INFO] repo=$STEREOVLA_ROOT"
echo "[INFO] storage=$STORAGE_PATH"
echo "[INFO] model=$model_path"
echo "[INFO] port=$PORT"
echo "[INFO] python=$PYTHON_BIN"

exec "$PYTHON_BIN" -m vla_network.scripts.serve \
  --path "$model_path" \
  --port "$PORT" \
  --batch-size "$BATCH_SIZE" \
  --batching-delay "$BATCHING_DELAY" \
  "${compile_args[@]}"
