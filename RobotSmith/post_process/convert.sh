#!/bin/bash
# ──────────────────────────────────────────────────────────────
# convert.sh  –  Run mesh processing pipeline (manifold → normalize → decompose → meta → adjust)
#
# Usage:
#   bash convert.sh /path/to/eef       # explicit eef directory
#   bash convert.sh                     # auto-detect ../eef relative to this script
#
# Environment variables (optional):
#   DGN_DIR    – path to DexGraspNet repo  (default: $HOME/DexGraspNet)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Help ──────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    head -n 11 "$0" | tail -n +2 | sed 's/^# *//'
    exit 0
fi

# ── Resolve script directory ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── EEF directory (first arg, or ../eef relative to script) ──
EEF_DIR="${1:-${SCRIPT_DIR}/../eef}"
EEF_DIR="$(cd "$EEF_DIR" && pwd)"   # canonicalize

# ── External tools ────────────────────────────────────────────
DGN_DIR="${DGN_DIR:-$HOME/DexGraspNet}"

# ── Sanity checks ─────────────────────────────────────────────
if [ ! -d "$EEF_DIR" ]; then
    echo "ERROR: EEF directory does not exist: $EEF_DIR"
    exit 1
fi
if [ ! -d "$DGN_DIR/asset_process" ]; then
    echo "ERROR: DexGraspNet not found at $DGN_DIR"
    echo "       Set DGN_DIR to the directory containing asset_process/"
    exit 1
fi

echo "──────────────────────────────────────────"
echo "EEF_DIR = $EEF_DIR"
echo "DGN_DIR = $DGN_DIR"
echo "──────────────────────────────────────────"

# ── Conda ─────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate isaac

# ── DexGraspNet mesh processing ───────────────────────────────
cd "$DGN_DIR/asset_process"

python "$DGN_DIR/asset_process/manifold.py" \
    --src "$EEF_DIR/objects" \
    --dst "$EEF_DIR/manifolds" \
    --manifold_path "$DGN_DIR/thirdparty/ManifoldPlus/build/manifold"

python "$DGN_DIR/asset_process/poolrun.py" -p 32

python "$DGN_DIR/asset_process/normalize.py" \
    --src "$EEF_DIR/manifolds" \
    --dst "$EEF_DIR/normalized_models"

python "$DGN_DIR/asset_process/decompose_list.py" \
    --src "$EEF_DIR/normalized_models" \
    --dst "$EEF_DIR/meshdata" \
    --coacd_path "$DGN_DIR/thirdparty/CoACD/build/main" \
    --t 0.05 --k 0.3

python "$DGN_DIR/asset_process/poolrun.py" -p 32

# ── Post-processing (convert_meta + adjust_meshes) ────────────
cd "$SCRIPT_DIR"

python convert_meta.py --eef-dir "$EEF_DIR"

python adjust_meshes.py --eef-dir "$EEF_DIR"