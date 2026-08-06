#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TOTAL_JOBS="${TOTAL_JOBS:-4}"
export JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
export SAVE_PATH="${SAVE_PATH:-/mnt/project/world_model/tool_generalist/grasp_result_dgn_test10}"

exec "$SCRIPT_DIR/eval_dgn.sh" "$@"
