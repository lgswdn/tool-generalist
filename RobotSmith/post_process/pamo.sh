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

# ── GPU detection ─────────────────────────────────────────────
AVAIL_GPUS=( $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null) )
NUM_AVAIL=${#AVAIL_GPUS[@]}

if [ "$NUM_AVAIL" -eq 0 ]; then
    echo "ERROR: nvidia-smi found no GPUs."
    exit 1
fi

MAX_JOBS="${MAX_JOBS:-$NUM_AVAIL}"   # default: 1 job per available GPU
MAX_RETRIES="${MAX_RETRIES:-5}"      # retries per job on failure
RETRY_DELAY="${RETRY_DELAY:-10}"     # seconds to wait between retries
GPU_IDX=0                            # round-robin counter

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
echo "EEF_DIR    = $EEF_DIR"
echo "PAMO_DIR   = $PAMO_DIR"
echo "AVAIL_GPUS = ${AVAIL_GPUS[*]}"
echo "MAX_JOBS   = $MAX_JOBS  (round-robin across available GPUs)"
echo "MAX_RETRIES= $MAX_RETRIES  (per job, ${RETRY_DELAY}s delay)"
echo "──────────────────────────────────────────"

# ── Conda ─────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pamo

# ── Define output directories ─────────────────────────────────
OUT_OBJ_DIR="${EEF_DIR}/objects"
OUT_META_DIR="${EEF_DIR}/objects_metadata"

mkdir -p "$OUT_OBJ_DIR"
mkdir -p "$OUT_META_DIR"

# ── Track failures (use a temp file so subshells can update) ──
FAIL_FILE=$(mktemp "${EEF_DIR}/.pamo_failures.XXXXXX")
echo 0 > "$FAIL_FILE"

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

        # ── Assign initial GPU (round-robin over AVAILABLE GPUs)
        INIT_GPU_IDX=$GPU_IDX
        GPU_IDX=$(( GPU_IDX + 1 ))

        # ── Launch pamo + retry logic in background ───────────
        (
            success=false
            for attempt in $(seq 0 "$MAX_RETRIES"); do
                # Cycle to a different GPU on each retry
                gpu_idx=$(( (INIT_GPU_IDX + attempt) % NUM_AVAIL ))
                GPU_ID="${AVAIL_GPUS[$gpu_idx]}"
                export CUDA_VISIBLE_DEVICES=$GPU_ID

                if [ "$attempt" -eq 0 ]; then
                    echo "[GPU $GPU_ID] [START] $obj_file"
                else
                    echo "[GPU $GPU_ID] [RETRY $attempt/$MAX_RETRIES] $obj_file"
                fi

                if python "$PAMO_DIR/example.py" \
                    --input "$obj_file" \
                    --output "$OUTPUT_OBJ" \
                    --ratio 0.1; then
                    success=true
                    break
                else
                    echo "[GPU $GPU_ID] [ATTEMPT $((attempt+1)) FAILED] $obj_file"
                    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                        sleep "$RETRY_DELAY"
                    fi
                fi
            done

            if $success; then
                # Copy the metadata file if it exists
                if [ -f "$INPUT_JSON" ]; then
                    cp "$INPUT_JSON" "$OUTPUT_JSON"
                else
                    echo "Warning: Metadata JSON not found for $obj_file"
                fi
                echo "[GPU $GPU_ID] [DONE]  $OUTPUT_OBJ"
            else
                echo "[FAIL after $((MAX_RETRIES+1)) attempts] $obj_file"
                # Atomically increment failure counter via flock
                flock "$FAIL_FILE" bash -c \
                    'n=$(cat "'"$FAIL_FILE"'"); echo $((n+1)) > "'"$FAIL_FILE"'"'
            fi
        ) &

    done
done

# ── Wait for all remaining background jobs ────────────────────
wait

FAIL_COUNT=$(cat "$FAIL_FILE")
rm -f "$FAIL_FILE"

echo "──────────────────────────────────────────"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "All pamo jobs finished.  ⚠  $FAIL_COUNT job(s) FAILED."
    exit 1
else
    echo "All pamo jobs finished.  ✓  No failures."
fi