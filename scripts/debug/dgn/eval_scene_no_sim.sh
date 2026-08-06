#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/mnt/home/zhengyixin/.conda/envs/data_convert/bin/python" ]]; then
    PYTHON_BIN="/mnt/home/zhengyixin/.conda/envs/data_convert/bin/python"
  elif [[ -x "/isaac-sim/python.sh" ]]; then
    PYTHON_BIN="/isaac-sim/python.sh"
  else
    PYTHON_BIN="python"
  fi
fi
SIMVLA_DATA="${SIMVLA_DATA:-/mnt/project/simvla/data}"
EVAL_SAMPLE_PATH="${EVAL_SAMPLE_PATH:-$SIMVLA_DATA/deploy-data/test_set/pick_all_v2/graspsim_test}"
DGN_ASSET_ROOT="${DGN_ASSET_ROOT:-/mnt/project/world_model/tool_generalist/assets/DGN}"
DGN_MANIFEST="${DGN_MANIFEST:-$DGN_ASSET_ROOT/full_yes.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/world_model/tool_generalist/grasp_result_dgn_scene_debug}"

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
  echo "[ERROR] Set PYTHON_BIN to an env with tensorflow_datasets installed." >&2
  exit 1
fi
if [[ ! -e "$EVAL_SAMPLE_PATH" ]]; then
  echo "[ERROR] Missing eval sample path: $EVAL_SAMPLE_PATH" >&2
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

exec "$PYTHON_BIN" thirdparty/graspsim-eval-stereovla/misc/debug_dgn_eval_scene_no_sim.py \
  --eval-sample-path "$EVAL_SAMPLE_PATH" \
  --simvla-data "$SIMVLA_DATA" \
  --dgn-root "$DGN_ASSET_ROOT" \
  --manifest "$DGN_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
