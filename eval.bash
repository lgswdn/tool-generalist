#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-name-or-path|rl-manifest.json|rl_runtime_spec.json> [extra eval_objects.py args...]" >&2
  echo "Example: $0 bimanual_unstable_yes --num_envs 512 --num_episodes 5" >&2
  echo "Exact run: $0 /path/to/RL/.../manifest.json" >&2
  echo "Exact spec: $0 /path/to/RL/.../rl_runtime_spec.json" >&2
  echo "4-GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 EVAL_NUM_GPUS=4 $0 bimanual_unstable_yes" >&2
  echo "8-GPU full set: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EVAL_NUM_GPUS=8 EVAL_NUM_ENVS=1450 $0 /path/to/rl_runtime_spec.json --replicate_objects_across_ranks --require_one_env_per_object --randomize_grippers" >&2
  exit 2
fi

SOURCE_INPUT="$1"
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
echo "[eval.bash] ${SOURCE_LABEL}"
echo "[eval.bash] PYTHONPATH=${PYTHONPATH}"

NUM_GPUS="${EVAL_NUM_GPUS:-1}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "EVAL_NUM_GPUS must be a positive integer, got '${NUM_GPUS}'." >&2
  exit 2
fi
NUM_ENVS="${EVAL_NUM_ENVS:-2048}"
NUM_EPISODES="${EVAL_NUM_EPISODES:-5}"

EXTRA_EVAL_ARGS=()
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  EXTRA_EVAL_ARGS+=(--distributed)
fi

EVAL_CMD=(
  scripts/eval_objects.py
  "${SOURCE_ARGS[@]}" \
  --num_envs "${NUM_ENVS}" \
  --num_episodes "${NUM_EPISODES}" \
  --headless \
  "${EXTRA_EVAL_ARGS[@]}" \
  "$@"
)

echo "[eval.bash] num_gpus=${NUM_GPUS} num_envs_per_gpu=${NUM_ENVS} episodes_per_object=${NUM_EPISODES}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  exec python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" "${EVAL_CMD[@]}"
fi

exec python "${EVAL_CMD[@]}"
