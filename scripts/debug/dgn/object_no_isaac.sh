#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DGN_ASSET_ROOT="${DGN_ASSET_ROOT:-/mnt/project/world_model/tool_generalist/assets/DGN}"
DGN_MANIFEST="${DGN_MANIFEST:-$DGN_ASSET_ROOT/full_yes.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/world_model/tool_generalist/grasp_result_dgn_obj_debug}"

cd "$REPO_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$DGN_ASSET_ROOT" ]]; then
  echo "[ERROR] Missing DGN asset root: $DGN_ASSET_ROOT" >&2
  exit 1
fi
if [[ ! -f "$DGN_MANIFEST" ]]; then
  echo "[ERROR] Missing DGN manifest: $DGN_MANIFEST" >&2
  exit 1
fi

exec "$PYTHON_BIN" thirdparty/graspsim-eval-stereovla/misc/debug_dgn_obj_no_isaac.py \
  --dgn-root "$DGN_ASSET_ROOT" \
  --manifest "$DGN_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
