#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${REPO_ROOT}"

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

CONFIG_INPUT="${PANDA_GRIPPER_VIS_CONFIG:-panda_gripper_diff_post}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_INPUT="$1"
  shift
fi

CONFIG_PATH="$(resolve_config "${CONFIG_INPUT}")"
echo "[panda_gripper_random.bash] config=${CONFIG_PATH}"
echo "[panda_gripper_random.bash] PYTHONPATH=${PYTHONPATH}"

EXTRA_ARGS=()
if [[ "${PANDA_GRIPPER_VIS_HEADLESS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--headless)
fi
if [[ "${PANDA_GRIPPER_VIS_VIDEO:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--video --video_dir "${PANDA_GRIPPER_VIS_VIDEO_DIR:-videos/panda_gripper_random}")
fi

exec python scripts/visualize_panda_gripper_random.py \
  --config "${CONFIG_PATH}" \
  --num_envs 1\
  --num_steps 100 \
  --seed "${PANDA_GRIPPER_VIS_SEED:-0}" \
  --action_mode "${PANDA_GRIPPER_VIS_ACTION_MODE:-random}" \
  --gripper_action_mode "${PANDA_GRIPPER_VIS_GRIPPER_ACTION_MODE:-sweep}" \
  --print_every "${PANDA_GRIPPER_VIS_PRINT_EVERY:-10}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
