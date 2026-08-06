#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DGN_ASSET_ROOT="${DGN_ASSET_ROOT:-/mnt/project/world_model/tool_generalist/assets/DGN}"
DGN_MANIFEST="${DGN_MANIFEST:-$DGN_ASSET_ROOT/full_yes.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/world_model/tool_generalist/grasp_result_dgn_visibility_tool_generalist}"
SCALE="${DGN_VISIBILITY_SCALE:-0.10}"

cd "$REPO_ROOT"
source "${HOME}/.bashrc"
conda activate "${CONDA_ENV:-isaac}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/source/IsaacLab_nonPrehensile${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
  echo "[ERROR] Set PYTHON_BIN=/path/to/python if this machine uses a different IsaacLab environment." >&2
  exit 1
fi
if [[ ! -d "$DGN_ASSET_ROOT" ]]; then
  echo "[ERROR] Missing DGN asset root: $DGN_ASSET_ROOT" >&2
  exit 1
fi

exec "$PYTHON_BIN" scripts/visualize_dgn_tool_generalist.py \
  --dgn-root "$DGN_ASSET_ROOT" \
  --manifest "$DGN_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --scale "$SCALE" \
  "$@"
