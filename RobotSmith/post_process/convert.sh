#!/bin/bash

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate isaac

# Define base directories
BASE_DIR="$HOME/project/RobotSmith/eef"
DGN_DIR="$HOME/DexGraspNet"

cd $DGN_DIR/asset_process

python $DGN_DIR/asset_process/manifold.py --src $BASE_DIR/objects --dst $BASE_DIR/manifolds --manifold_path $DGN_DIR/thirdparty/ManifoldPlus/build/manifold
python $DGN_DIR/asset_process/poolrun.py -p 32

python $DGN_DIR/asset_process/normalize.py --src $BASE_DIR/manifolds --dst $BASE_DIR/normalized_models

python $DGN_DIR/asset_process/decompose_list.py --src $BASE_DIR/normalized_models --dst $BASE_DIR/meshdata --coacd_path $DGN_DIR/thirdparty/CoACD/build/main --t 0.04 --k 0.3
python $DGN_DIR/asset_process/poolrun.py -p 32

cd $BASE_DIR/../post_process

python convert_meta.py

python adjust_meshes.py