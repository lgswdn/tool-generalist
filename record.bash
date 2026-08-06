#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-name-or-path|rl-manifest.json|rl_runtime_spec.json> [extra record_failure_videos.py args...]" >&2
  echo "Example: $0 bimanual_stable_full_tool_diff_post --num_envs 128" >&2
  echo "Exact run: $0 /path/to/RL/.../manifest.json" >&2
  echo "Exact spec: $0 /path/to/RL/.../rl_runtime_spec.json" >&2
  echo "2-GPU: CUDA_VISIBLE_DEVICES=0,1 RECORD_NUM_GPUS=2 $0 bimanual_stable_full_tool_diff_post" >&2
  exit 2
fi

SOURCE_INPUT="$1"
shift

REPO_ROOT="/mnt/home/zhengyixin/tool-generalist"
cd "${REPO_ROOT}"

# The Isaac Conda activation hook reads $PYTHONPATH directly. Define it
# before activation so `set -u` does not reject an unset variable.
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

resolve_runtime_spec() {
  local raw="$1"
  local path="${raw}"

  if [[ -d "${path}" ]]; then
    path="${path}/manifest.json"
  fi
  if [[ ! -f "${path}" ]]; then
    return 1
  fi

  local base
  base="$(basename "${path}")"
  if [[ "${base}" == "rl_runtime_spec.json" ]]; then
    echo "${path}"
    return 0
  fi
  if [[ "${base}" != "manifest.json" ]]; then
    return 1
  fi

  local spec
  spec="$(dirname "${path}")/rl_runtime_spec.json"
  if [[ ! -f "${spec}" ]]; then
    echo "Manifest '${path}' does not have sibling rl_runtime_spec.json; pass an RL artifact manifest, not an experiment/pretrain/contact manifest." >&2
    return 1
  fi
  echo "${spec}"
}

SOURCE_ARGS=()
SOURCE_LABEL=""
if RUNTIME_SPEC_PATH="$(resolve_runtime_spec "${SOURCE_INPUT}")"; then
  SOURCE_ARGS+=(--runtime_spec "${RUNTIME_SPEC_PATH}")
  SOURCE_LABEL="runtime_spec=${RUNTIME_SPEC_PATH}"
else
  CONFIG_PATH="$(resolve_config "${SOURCE_INPUT}")"
  SOURCE_ARGS+=(--config "${CONFIG_PATH}")
  SOURCE_LABEL="config=${CONFIG_PATH}"
fi
echo "[record.bash] ${SOURCE_LABEL}"
echo "[record.bash] PYTHONPATH=${PYTHONPATH}"

if [[ "${SOURCE_ARGS[0]}" == "--config" ]]; then
  SOURCE_ROBOT_MODE="$(
    python -c \
      'import sys; from utils.config.loader import load_exp_cfg; print(load_exp_cfg(sys.argv[1]).rl.env.robot_mode)' \
      "${CONFIG_PATH}"
  )"
else
  SOURCE_ROBOT_MODE="$(
    python -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["env_params"]["robot_mode"])' \
      "${RUNTIME_SPEC_PATH}"
  )"
fi
if [[ "${SOURCE_ROBOT_MODE}" == "cross_embodiment_gripper" ]]; then
  # Recording only needs one rank per CE family.  Training uses all configured
  # GPUs, but spawning eight camera-enabled Isaac Sim processes makes
  # checkpoint loading/render startup unnecessarily expensive.
  NUM_GPUS="${RECORD_NUM_GPUS:-2}"
else
  NUM_GPUS="${RECORD_NUM_GPUS:-1}"
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "RECORD_NUM_GPUS must be a positive integer, got '${NUM_GPUS}'." >&2
  exit 2
fi

EXTRA_RECORD_ARGS=()
if [[ "${RECORD_DISABLE_VISUAL_OVERRIDES:-0}" == "1" ]]; then
  EXTRA_RECORD_ARGS+=(--disable_recording_visual_overrides)
fi
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  EXTRA_RECORD_ARGS+=(--distributed)
fi

DEFAULT_FAILURE_VIDEOS="${RECORD_NUM_FAILURE_VIDEOS:-8}"
DEFAULT_SUCCESS_VIDEOS="${RECORD_NUM_SUCCESS_VIDEOS:-32}"
DEFAULT_ACTIVE_EPISODES=2
if [[ "${DEFAULT_FAILURE_VIDEOS}" == "0" && "${DEFAULT_SUCCESS_VIDEOS}" != "0" ]]; then
  DEFAULT_ACTIVE_EPISODES=16
fi
DEFAULT_VIDEO_WIDTH=512
DEFAULT_VIDEO_HEIGHT=512
if [[ "${SOURCE_ARGS[0]}" == "--config" && "$(basename "${CONFIG_PATH}")" == panda_gripper* ]]; then
  DEFAULT_VIDEO_WIDTH=1280
  DEFAULT_VIDEO_HEIGHT=720
fi
VIDEO_ARGS=()
if [[ -n "${RECORD_VIDEO_WIDTH:-}" ]]; then
  VIDEO_ARGS+=(--video_width "${RECORD_VIDEO_WIDTH}")
elif [[ "${SOURCE_ARGS[0]}" == "--config" && "$(basename "${CONFIG_PATH}")" == panda_gripper* ]]; then
  VIDEO_ARGS+=(--video_width "${DEFAULT_VIDEO_WIDTH}")
fi
if [[ -n "${RECORD_VIDEO_HEIGHT:-}" ]]; then
  VIDEO_ARGS+=(--video_height "${RECORD_VIDEO_HEIGHT}")
elif [[ "${SOURCE_ARGS[0]}" == "--config" && "$(basename "${CONFIG_PATH}")" == panda_gripper* ]]; then
  VIDEO_ARGS+=(--video_height "${DEFAULT_VIDEO_HEIGHT}")
fi

RECORD_CMD=(
  scripts/record_failure_videos.py
  "${SOURCE_ARGS[@]}" \
  --num_envs "${RECORD_NUM_ENVS:-64}" \
  --num_failure_videos "${DEFAULT_FAILURE_VIDEOS}" \
  --num_success_videos "${DEFAULT_SUCCESS_VIDEOS}" \
  --video_max_active_episodes "${RECORD_VIDEO_MAX_ACTIVE_EPISODES:-${DEFAULT_ACTIVE_EPISODES}}" \
  "${VIDEO_ARGS[@]}" \
  --video_fps "${RECORD_VIDEO_FPS:-10}" \
  --headless \
  "${EXTRA_RECORD_ARGS[@]}" \
  "$@"
)

echo "[record.bash] robot_mode=${SOURCE_ROBOT_MODE} num_gpus=${NUM_GPUS} num_envs_per_gpu=${RECORD_NUM_ENVS:-64}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  exec python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" "${RECORD_CMD[@]}"
fi

exec python "${RECORD_CMD[@]}"
