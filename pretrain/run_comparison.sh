#!/bin/bash
# run_comparison.sh - Generate contact configs using both methods and heatmaps
#
# Usage:
#   ./run_comparison.sh <object.obj> <tool.obj> [output_name]
#
# Example:
#   ./run_comparison.sh /path/to/bottle.obj /path/to/scraper.obj bottle_scraper

set -e

OBJECT="$1"
TOOL="$2"
NAME="$3"

if [ -z "$OBJECT" ] || [ -z "$TOOL" ]; then
    echo "Usage: $0 <object.obj> <tool.obj> [output_name]"
    exit 1
fi

# Extract shortened name from object file if not provided
# Long names like "core-bottle-41a2005b595ae783be1868124d5ddbcb" → "bottle"
if [ -z "$NAME" ]; then
    FULLNAME=$(basename "$OBJECT" .obj)
    # Remove prefix like "core-", "acronym-", etc. and hash suffix
    # Pattern: <prefix>-<name>-<hash> → extract <name>
    NAME=$(echo "$FULLNAME" | sed 's/^core-//' | sed 's/^acronym-//' | cut -d'-' -f1)
fi

OUT_DIR="results/${NAME}"
mkdir -p "$OUT_DIR"

BATCH_SIZE=512
DEVICE="cuda:0"
SEED=42
RADIUS=0.03

echo "=============================================="
echo "Object: $OBJECT"
echo "Tool:   $TOOL"
echo "Output: $OUT_DIR"
echo "=============================================="

# Method 1: Original (optimize)
echo ""
echo "[1] Running ORIGINAL (optimize) method..."
python3 contact_gen.py \
    --object "$OBJECT" \
    --tool "$TOOL" \
    --output "${OUT_DIR}/${NAME}_orig.pt" \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --seed $SEED

# Method 2: Corn (gradient)
echo ""
echo "[2] Running CORN (gradient) method..."
python3 contact_gen_gradient.py \
    --object "$OBJECT" \
    --tool "$TOOL" \
    --output "${OUT_DIR}/${NAME}_corn.pt" \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --seed $SEED

# Heatmaps
echo ""
echo "[3] Generating heatmaps..."
python3 export_contact_heatmap.py \
    -i "${OUT_DIR}/${NAME}_orig.pt" \
    -o "${OUT_DIR}/heatmap_orig.obj" \
    -r $RADIUS

python3 export_contact_heatmap.py \
    -i "${OUT_DIR}/${NAME}_corn.pt" \
    -o "${OUT_DIR}/heatmap_corn.obj" \
    -r $RADIUS

echo ""
echo "=============================================="
echo "Done! Results in $OUT_DIR:"
ls -la "$OUT_DIR"
echo "=============================================="