#!/bin/bash
# ──────────────────────────────────────────────────────────────
# pamo.sh  –  Run PAMO decimation on generated tool OBJ files
#
# Usage:
#   bash pamo.sh /path/to/eef          # explicit eef directory
#   bash pamo.sh                       # auto-detect ../eef relative to this script
#
# Environment variables (optional):
#   PAMO_DIR   – path to pamo repo   (default: $HOME/project/pamo)
#   MAX_JOBS   – max parallel workers (default: 8)
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
PAMO_DIR="${PAMO_DIR:-$HOME/project/pamo}"

# ── GPU detection & parallelism ───────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
MAX_JOBS="${MAX_JOBS:-$NUM_GPUS}"   # default: 1 job per GPU
GPU_IDX=0                           # round-robin counter

# ── Sanity checks ─────────────────────────────────────────────
if [ ! -d "$EEF_DIR/tmp_trial" ]; then
    echo "ERROR: $EEF_DIR/tmp_trial does not exist. Nothing to process."
    exit 1
fi
if [ ! -f "$PAMO_DIR/example.py" ]; then
    echo "ERROR: pamo not found at $PAMO_DIR/example.py"
    echo "       Set PAMO_DIR to the directory containing example.py"
    exit 1
fi

echo "──────────────────────────────────────────"
echo "EEF_DIR  = $EEF_DIR"
echo "PAMO_DIR = $PAMO_DIR"
echo "NUM_GPUS = $NUM_GPUS"
echo "MAX_JOBS = $MAX_JOBS  (round-robin across GPUs)"
echo "──────────────────────────────────────────"

# ── Conda ─────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pamo

# ── Define output directories ─────────────────────────────────
OUT_OBJ_DIR="${EEF_DIR}/objects"
OUT_META_DIR="${EEF_DIR}/objects_metadata"

mkdir -p "$OUT_OBJ_DIR"
mkdir -p "$OUT_META_DIR"

# ── Track failures ────────────────────────────────────────────
FAIL_COUNT=0

for trial_dir in "$EEF_DIR"/tmp_trial/*; do
    # Skip if it's not a directory
    [ -d "$trial_dir" ] || continue

    # Extract 'x' from the directory name (e.g., tmp_trial_5 -> 5)
    dir_name=$(basename "$trial_dir")
    x="${dir_name#tmp_trial/}"

    # Loop through all matching .obj files in this directory
    for obj_file in "$trial_dir"/*_var_*.obj; do
        # Skip if no matching files are found
        [ -f "$obj_file" ] || continue

        # Extract the filename without the path (e.g., apple_var_002.obj)
        filename=$(basename "$obj_file")
        
        # Remove the .obj extension (e.g., apple_var_002)
        base_name="${filename%.obj}"
        
        # Extract 'i' (everything after the last '_var_')
        i_padded="${base_name##*_var_}"
        
        # Extract 'name' (everything before the last '_var_')
        name="${base_name%_var_*}"

        # Construct the output paths
        OUTPUT_OBJ="${OUT_OBJ_DIR}/${x}_${name}_var_${i_padded}.obj"
        INPUT_JSON="${trial_dir}/${name}_var_${i_padded}_metadata.json"
        OUTPUT_JSON="${OUT_META_DIR}/${x}_${name}_var_${i_padded}_metadata.json"

        # ── Throttle: wait if we have MAX_JOBS running ────────
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
            wait -n 2>/dev/null || true
        done

        # ── Assign GPU (round-robin) ──────────────────────────
        GPU_ID=$(( GPU_IDX % NUM_GPUS ))
        GPU_IDX=$(( GPU_IDX + 1 ))

        # ── Launch pamo + metadata copy in background ─────────
        (
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            echo "[GPU $GPU_ID] [START] $obj_file"
            python "$PAMO_DIR/example.py" \
                --input "$obj_file" \
                --output "$OUTPUT_OBJ" \
                --ratio 0.1

            # Copy the metadata file if it exists
            if [ -f "$INPUT_JSON" ]; then
                cp "$INPUT_JSON" "$OUTPUT_JSON"
            else
                echo "Warning: Metadata JSON not found for $obj_file"
            fi
            echo "[GPU $GPU_ID] [DONE]  $OUTPUT_OBJ"
        ) &

    done
done

# ── Wait for all remaining background jobs ────────────────────
wait

echo "──────────────────────────────────────────"
echo "All pamo jobs finished."