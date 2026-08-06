#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}"
source "${HOME}/.bashrc"
conda activate isaac
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/source/IsaacLab_nonPrehensile${PYTHONPATH:+:${PYTHONPATH}}"

exec python scripts/run_ce_prl_oracle_rebuild_pipeline.py "$@"
