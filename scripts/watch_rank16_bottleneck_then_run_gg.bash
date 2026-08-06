#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
cd "${REPO_ROOT}"

exec python3 scripts/wait_for_bottleneck_full_yes_then_run_gg.py \
  --rank 16 \
  --parent-experiment panda_general_unicorn_ours_encoder_bottleneck_rank16_full_yes_5k \
  --child-experiment panda_general_unicorn_ours_encoder_bottleneck_rank16_gg_from_full_yes_5k \
  --poll-seconds "${POLL_SECONDS:-60}"
