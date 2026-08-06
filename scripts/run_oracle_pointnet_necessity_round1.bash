#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  full_tce)
    config="ce_prl_oracle_ablation_d12_full_tce_dgn_5k"
    ;;
  bottleneck_budget)
    config="ce_prl_oracle_ablation_d12_bottleneck_resume_to_dgn_10k"
    ;;
  fitted_frozen)
    config="ce_prl_oracle_ablation_d12_fitted_pointnet_frozen_dgn_5k"
    ;;
  *)
    echo "Usage: $0 {full_tce|bottleneck_budget|fitted_frozen} [run arguments...]" >&2
    exit 2
    ;;
esac
shift

export RUN_NUM_GPUS=8
export RUN_TOTAL_ENVS=8192
exec ./run.bash "${config}" "$@"
