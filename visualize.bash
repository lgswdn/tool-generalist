#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-name-or-path> [extra record_video.py args...]" >&2
  echo "Example: $0 bimanual_unstable_full_tool_diff_post_no_curriculum --video_length 200" >&2
  exit 2
fi

CONFIG_INPUT="$1"
shift

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
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

CONFIG_PATH="$(resolve_config "${CONFIG_INPUT}")"
echo "[visualize.bash] config=${CONFIG_PATH}"
echo "[visualize.bash] PYTHONPATH=${PYTHONPATH}"

exec python scripts/record_video.py \
  --config "${CONFIG_PATH}" \
  --num_envs "${VISUALIZE_NUM_ENVS:-1}" \
  --video_length "${VISUALIZE_VIDEO_LENGTH:-200}" \
  --video_dir "${VISUALIZE_VIDEO_DIR:-videos/bimanual_layout_check}" \
  --action_mode "${VISUALIZE_ACTION_MODE:-zero}" \
  --headless \
  "$@"
