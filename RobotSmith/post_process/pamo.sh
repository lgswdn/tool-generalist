#!/bin/bash

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pamo

# Define base directories
BASE_DIR="$HOME/project/RobotSmith/eef"
OUT_OBJ_DIR="${BASE_DIR}/objects"
OUT_META_DIR="${BASE_DIR}/objects_metadata"

mkdir -p "$OUT_OBJ_DIR"
mkdir -p "$OUT_META_DIR"

for trial_dir in "$BASE_DIR"/tmp_trial/*; do
    echo $trial_dir
    # Skip if it's not a directory
    [ -d "$trial_dir" ] || continue

    # Extract 'x' from the directory name (e.g., tmp_trial_5 -> 5)
    dir_name=$(basename "$trial_dir")
    x="${dir_name#tmp_trial/}"

    # Loop through all matching .obj files in this directory
    for obj_file in "$trial_dir"/*_var_*.obj; do
        # Skip if no matching files are found
        [ -f "$obj_file" ] || continue
        echo $obj_file

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

        # 1. Execute the Python processing script
        python "$HOME/project/pamo/example.py" \
            --input "$obj_file" \
            --output "$OUTPUT_OBJ" \
            --ratio 0.01
        
        # cp $obj_file $OUTPUT_OBJ

        # 2. Copy the metadata file if it exists
        if [ -f "$INPUT_JSON" ]; then
            cp "$INPUT_JSON" "$OUTPUT_JSON"
        else
            echo "Warning: Metadata JSON not found for $obj_file"
        fi
    done
done