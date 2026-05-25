#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-name-or-path> [extra record_failure_videos.py args...]" >&2
  echo "Example: $0 bimanual_stable_full_tool_diff_post --num_envs 128" >&2
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
echo "[record.bash] config=${CONFIG_PATH}"
echo "[record.bash] PYTHONPATH=${PYTHONPATH}"

EXTRA_RECORD_ARGS=()
if [[ "${RECORD_DISABLE_VISUAL_OVERRIDES:-0}" == "1" ]]; then
  EXTRA_RECORD_ARGS+=(--disable_recording_visual_overrides)
fi

DEFAULT_FAILURE_VIDEOS="${RECORD_NUM_FAILURE_VIDEOS:-4}"
DEFAULT_SUCCESS_VIDEOS="${RECORD_NUM_SUCCESS_VIDEOS:-4}"
DEFAULT_ACTIVE_EPISODES=2
if [[ "${DEFAULT_FAILURE_VIDEOS}" == "0" && "${DEFAULT_SUCCESS_VIDEOS}" != "0" ]]; then
  DEFAULT_ACTIVE_EPISODES=16
fi

exec python scripts/record_failure_videos.py \
  --config "${CONFIG_PATH}" \
  --num_envs "${RECORD_NUM_ENVS:-64}" \
  --num_failure_videos "${DEFAULT_FAILURE_VIDEOS}" \
  --num_success_videos "${DEFAULT_SUCCESS_VIDEOS}" \
  --video_max_active_episodes "${RECORD_VIDEO_MAX_ACTIVE_EPISODES:-${DEFAULT_ACTIVE_EPISODES}}" \
  --video_width "${RECORD_VIDEO_WIDTH:-512}" \
  --video_height "${RECORD_VIDEO_HEIGHT:-512}" \
  --video_fps "${RECORD_VIDEO_FPS:-10}" \
  --headless \
  "${EXTRA_RECORD_ARGS[@]}" \
  "$@"
