#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
DATA_DIR="/mnt/project/world_model/tool_generalist/artifacts/contact/fork_sdf/contact_gen_full_tool/281987b90b894c5a84c97b9b0c89bca2d8711036c52e2d2b3f7f0a65f7d94535"
OUT_ROOT="/mnt/project/world_model/tool_generalist/contact_viz_pre"
WORKERS="${CONTACT_OBJ_VIZ_WORKERS:-16}"
NUM_OUTPUTS="${CONTACT_OBJ_VIZ_NUM_OUTPUTS:-100}"

cd "${REPO_ROOT}"

source "${HOME}/.bashrc"
conda activate isaac

while IFS='|' read -r tool folder; do
  echo "[export_selected_contact_obj_viz] tool=${tool} output=${OUT_ROOT}/${folder}"
  python contact_generation/export_contact_obj_viz.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_ROOT}/${folder}" \
    --workers "${WORKERS}" \
    "${tool}" \
    "${NUM_OUTPUTS}"
done <<'EOF'
hand_fork|Hand_Fork
margin_trowel|Margin_Trowel
curved_claw_hammer|Curved_Claw_Hammer
nail_punch|Nail_Punch
chef_s_knife|Chef's_Knife
hooked_crowbar|Hooked_Crowbar
straight_icing_spatula|Straight_Icing_Spatula
offset_icing_spatula|Offset_Icing_Spatula
edger_trowel|Edger_Trowel
pasta_fork|Pasta_Fork
flour_scoop|Flour_Scoop
flared_flathead_screwdriver|Flared_Flathead_Screwdriver
oyster_knife|Oyster_Knife
EOF
