#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-name-or-path> [curr] [extra run_experiment.py args...]" >&2
  echo "Example: $0 multitools_full_tool_contact_shard_0" >&2
  echo "Curriculum from eval: $0 bimanual_unstable_yes curr" >&2
  echo "4-GPU with fixed total envs: RUN_NUM_GPUS=4 $0 bimanual_unstable_yes" >&2
  exit 2
fi

CONFIG_INPUT="$1"
shift
RUN_EXTRA_ARGS=()
RUN_CURR=0
if [[ "${1:-}" == "curr" ]]; then
  RUN_CURR=1
  RUN_EXTRA_ARGS+=(--curriculum-from-eval)
  if [[ "${RUN_CURR_RESUME:-1}" == "0" ]]; then
    RUN_EXTRA_ARGS+=(--no-curriculum-resume)
  fi
  shift
fi

if [[ -n "${RUN_NUM_GPUS:-}" ]]; then
  if ! [[ "${RUN_NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${RUN_NUM_GPUS}" -lt 1 ]]; then
    echo "RUN_NUM_GPUS must be a positive integer, got '${RUN_NUM_GPUS}'." >&2
    exit 2
  fi
  RUN_TOTAL_ENVS="${RUN_TOTAL_ENVS:-8192}"
  if ! [[ "${RUN_TOTAL_ENVS}" =~ ^[0-9]+$ ]] || [[ "${RUN_TOTAL_ENVS}" -lt 1 ]]; then
    echo "RUN_TOTAL_ENVS must be a positive integer, got '${RUN_TOTAL_ENVS}'." >&2
    exit 2
  fi
  if (( RUN_TOTAL_ENVS % RUN_NUM_GPUS != 0 )); then
    echo "RUN_TOTAL_ENVS (${RUN_TOTAL_ENVS}) must be divisible by RUN_NUM_GPUS (${RUN_NUM_GPUS})." >&2
    exit 2
  fi
  RUN_ENVS_PER_GPU=$((RUN_TOTAL_ENVS / RUN_NUM_GPUS))
  RUN_EXTRA_ARGS+=(
    --runtime-num-gpus "${RUN_NUM_GPUS}"
    --runtime-total-envs "${RUN_TOTAL_ENVS}"
  )
fi

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
cd "${REPO_ROOT}"

# The Isaac Conda activation hook reads $PYTHONPATH directly.  Define it
# before activation so `set -u` does not turn an unset variable into an error.
export PYTHONPATH="${PYTHONPATH:-}"
source "${HOME}/.bashrc"
conda activate isaac

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/source/IsaacLab_nonPrehensile${PYTHONPATH:+:${PYTHONPATH}}"

resolve_config() {
  local raw="$1"
  local stem="${raw%.py}"
  local normalized="${stem}"

  if [[ -f "${raw}" ]]; then
    echo "${raw}"
    return 0
  fi

  if [[ "${normalized}" =~ ^(.*_shard)_([0-9]+)$ ]]; then
    normalized="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
  fi

  local candidates=(
    "${stem}.py"
    "${normalized}.py"
    "configs/experiments/${stem}.py"
    "configs/experiments/${normalized}.py"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "Could not find config for '${raw}'." >&2
  echo "Tried:" >&2
  for candidate in "${candidates[@]}"; do
    echo "  ${candidate}" >&2
  done
  return 1
}

CONFIG_PATH="$(resolve_config "${CONFIG_INPUT}")"
echo "[run.bash] config=${CONFIG_PATH}"

CONFIG_STEM="$(basename "${CONFIG_PATH}" .py)"
if [[
  -z "${TOOL_GENERALIST_BYPASS_GG_PARENT_WAIT:-}"
  && "${CONFIG_STEM}" == *_gg_15k
]]; then
  echo "[run.bash] checking mapped DGN parent for GG child=${CONFIG_STEM}"
  if python scripts/wait_unicorn_full_yes_then_gg.py \
    --child-experiment "${CONFIG_STEM}"; then
    exit 0
  else
    WAIT_STATUS=$?
    if [[ "${WAIT_STATUS}" -ne 4 ]]; then
      exit "${WAIT_STATUS}"
    fi
    echo "[run.bash] no automatic parent watcher registered for ${CONFIG_STEM}"
  fi
fi

if [[ "${RUN_CURR}" == "1" ]]; then
  echo "[run.bash] curriculum_from_eval=1"
  echo "[run.bash] curriculum_resume_from_eval=${RUN_CURR_RESUME:-1}"
fi
if [[ -n "${RUN_NUM_GPUS:-}" ]]; then
  echo "[run.bash] runtime_num_gpus=${RUN_NUM_GPUS} runtime_total_envs=${RUN_TOTAL_ENVS} runtime_envs_per_gpu=${RUN_ENVS_PER_GPU}"
fi
echo "[run.bash] PYTHONPATH=${PYTHONPATH}"

exec python run_experiment.py --config "${CONFIG_PATH}" --mode run "${RUN_EXTRA_ARGS[@]}" "$@"
