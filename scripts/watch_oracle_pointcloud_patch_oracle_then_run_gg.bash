#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
cd "${REPO_ROOT}"

exec python3 scripts/wait_for_full_yes_then_run_gg.py \
  --parent-experiment panda_general_oracle_pointcloud_patch_oracle_full_yes_5k \
  --child-experiment panda_general_oracle_pointcloud_patch_oracle_gg_from_full_yes_5k \
  --encoder-family oracle_pointcloud_patch_oracle \
  --encoder-backend oracle_pointcloud_patch_oracle \
  --poll-seconds "${POLL_SECONDS:-60}"
