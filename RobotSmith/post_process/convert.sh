#!/bin/bash
# ──────────────────────────────────────────────────────────────
# convert.sh  –  Run mesh processing pipeline (manifold → normalize → decompose → meta → adjust)
#
# Usage:
#   bash convert.sh [EEF_DIR] [DGN_DIR]
#   bash convert.sh ../eef /path/to/DexGraspNet
#   bash convert.sh                    # defaults: ../eef and repo-local DexGraspNet if present
#
# Environment variables (optional):
#   DGN_DIR        – path to DexGraspNet repo
#   POOL_WORKERS   – parallel workers for DexGraspNet poolrun.py (default: 32)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Help ──────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    head -n 12 "$0" | tail -n +2 | sed 's/^# *//'
    exit 0
fi

# ── Resolve script directory ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Positional args ───────────────────────────────────────────
EEF_DIR="${1:-${SCRIPT_DIR}/../eef}"
if [ ! -d "$EEF_DIR" ]; then
    echo "ERROR: EEF directory does not exist: $EEF_DIR"
    exit 1
fi
EEF_DIR="$(cd "$EEF_DIR" && pwd)"   # canonicalize

# ── External tools ────────────────────────────────────────────
DEFAULT_DGN_DIR="${SCRIPT_DIR}/../../../DexGraspNet"
if [ ! -d "$DEFAULT_DGN_DIR/asset_process" ]; then
    DEFAULT_DGN_DIR="$HOME/DexGraspNet"
fi
DGN_DIR="${2:-${DGN_DIR:-$DEFAULT_DGN_DIR}}"
if [ ! -d "$DGN_DIR" ]; then
    echo "ERROR: DexGraspNet directory does not exist: $DGN_DIR"
    exit 1
fi
DGN_DIR="$(cd "$DGN_DIR" && pwd)"
MANIFOLD_BIN="${DGN_DIR}/thirdparty/ManifoldPlus/build/manifold"
COACD_BIN="${DGN_DIR}/thirdparty/CoACD/build/main"
POOL_WORKERS="${POOL_WORKERS:-32}"

# ── Sanity checks ─────────────────────────────────────────────
if [ ! -d "$DGN_DIR/asset_process" ]; then
    echo "ERROR: DexGraspNet not found at $DGN_DIR"
    echo "       Set DGN_DIR to the directory containing asset_process/"
    exit 1
fi
if ! compgen -G "$EEF_DIR/objects/*.obj" >/dev/null; then
    echo "ERROR: no OBJ files found in $EEF_DIR/objects"
    echo "       Run pamodet.sh first, or pass the correct EEF_DIR."
    exit 1
fi
if ! compgen -G "$EEF_DIR/objects_metadata/*_metadata.json" >/dev/null; then
    echo "ERROR: no metadata JSON files found in $EEF_DIR/objects_metadata"
    echo "       Run pamodet.sh first, or pass the correct EEF_DIR."
    exit 1
fi
if [ ! -x "$MANIFOLD_BIN" ]; then
    echo "ERROR: ManifoldPlus executable not found or not executable: $MANIFOLD_BIN"
    echo "       Build ManifoldPlus under DexGraspNet/thirdparty/ManifoldPlus first."
    exit 1
fi
if [ ! -x "$COACD_BIN" ]; then
    echo "ERROR: CoACD executable not found or not executable: $COACD_BIN"
    echo "       Build CoACD under DexGraspNet/thirdparty/CoACD first."
    exit 1
fi

echo "──────────────────────────────────────────"
echo "EEF_DIR      = $EEF_DIR"
echo "DGN_DIR      = $DGN_DIR"
echo "MANIFOLD_BIN = $MANIFOLD_BIN"
echo "COACD_BIN    = $COACD_BIN"
echo "POOL_WORKERS = $POOL_WORKERS"
echo "──────────────────────────────────────────"

# ── Conda ─────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate isaac

# ── DexGraspNet mesh processing ───────────────────────────────
cd "$DGN_DIR/asset_process"

#python "$DGN_DIR/asset_process/manifold.py" \
#    --src "$EEF_DIR/objects" \
#    --dst "$EEF_DIR/manifolds" \
#    --manifold_path "$MANIFOLD_BIN"

#python "$DGN_DIR/asset_process/poolrun.py" -p "$POOL_WORKERS"

python "$DGN_DIR/asset_process/normalize.py" \
    --src "$EEF_DIR/objects" \
    --dst "$EEF_DIR/normalized_models"

python "$DGN_DIR/asset_process/decompose_list.py" \
    --src "$EEF_DIR/normalized_models" \
    --dst "$EEF_DIR/meshdata" \
    --coacd_path "$COACD_BIN"

python "$DGN_DIR/asset_process/poolrun.py" -p "$POOL_WORKERS"

# ── Post-processing (convert_meta + adjust_meshes) ────────────
cd "$SCRIPT_DIR"

python convert_meta.py --eef-dir "$EEF_DIR"

python adjust_meshes.py --eef-dir "$EEF_DIR"
