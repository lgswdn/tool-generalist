#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGN_CANDIDATES_JSON="${DGN_CANDIDATES_JSON:-/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json}"
DGN_ASSET_ROOT="${DGN_ASSET_ROOT:-/mnt/project/world_model/tool_generalist/assets/DGN}"
SAVE_PATH="${SAVE_PATH:-/mnt/project/world_model/tool_generalist/grasp_result_dgn_full_yes}"
BASE_SAVE_PATH="$SAVE_PATH"
DGN_SHARD_COUNT="${DGN_SHARD_COUNT:-${SHARD_COUNT:-1}}"
DGN_SHARD_INDEX="${DGN_SHARD_INDEX:-${SHARD_INDEX:-0}}"
DGN_SHARD_SPLIT_SAVE_PATH="${DGN_SHARD_SPLIT_SAVE_PATH:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-8}"
if [[ -n "${DGN_OBJECT_SCALE:-}" ]]; then
  DGN_OBJECT_SCALE="${DGN_OBJECT_SCALE}"
  DGN_OBJECT_SCALE_MIN="${DGN_OBJECT_SCALE_MIN:-}"
  DGN_OBJECT_SCALE_MAX="${DGN_OBJECT_SCALE_MAX:-}"
else
  DGN_OBJECT_SCALE=""
  if [[ -n "${DGN_OBJECT_SCALE_MIN:-}" || -n "${DGN_OBJECT_SCALE_MAX:-}" ]]; then
    DGN_OBJECT_SCALE_MIN="${DGN_OBJECT_SCALE_MIN:-}"
    DGN_OBJECT_SCALE_MAX="${DGN_OBJECT_SCALE_MAX:-}"
  else
    DGN_OBJECT_SCALE_MIN="0.10"
    DGN_OBJECT_SCALE_MAX="0.30"
  fi
fi
DGN_OBJECT_SCALE_SEED="${DGN_OBJECT_SCALE_SEED:-0}"
DGN_STABLE_POSE="${DGN_STABLE_POSE:-1}"
DGN_STABLE_POSE_SAMPLE_NUM="${DGN_STABLE_POSE_SAMPLE_NUM:-64}"
DGN_STABLE_POSE_SEED="${DGN_STABLE_POSE_SEED:-0}"
FAILURE_RETRY_COUNT="${FAILURE_RETRY_COUNT:-2}"
FAILURE_RETRY_XY_RADIUS="${FAILURE_RETRY_XY_RADIUS:-0.04}"
FAILURE_RETRY_SEED="${FAILURE_RETRY_SEED:-0}"
OBJECT_NUM="${OBJECT_NUM:-1}"
START_SERVERS="${START_SERVERS:-0}"

if [[ ! -f "$DGN_CANDIDATES_JSON" ]]; then
  echo "[ERROR] Missing DGN candidates JSON: $DGN_CANDIDATES_JSON" >&2
  exit 1
fi
if [[ ! -d "$DGN_ASSET_ROOT" ]]; then
  echo "[ERROR] Missing DGN asset root: $DGN_ASSET_ROOT" >&2
  exit 1
fi

manifest_jobs="$(
  python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DGN_CANDIDATES_JSON"
)"
base_start="${START_FROM:-0}"
if ! [[ "$base_start" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] START_FROM must be a non-negative integer, got: $base_start" >&2
  exit 1
fi
if ! [[ "$DGN_SHARD_COUNT" =~ ^[0-9]+$ ]] || (( DGN_SHARD_COUNT <= 0 )); then
  echo "[ERROR] DGN_SHARD_COUNT must be a positive integer, got: $DGN_SHARD_COUNT" >&2
  exit 1
fi
if ! [[ "$DGN_SHARD_INDEX" =~ ^[0-9]+$ ]] || (( DGN_SHARD_INDEX >= DGN_SHARD_COUNT )); then
  echo "[ERROR] DGN_SHARD_INDEX must be in [0, DGN_SHARD_COUNT), got: $DGN_SHARD_INDEX of $DGN_SHARD_COUNT" >&2
  exit 1
fi
if [[ -z "${TOTAL_JOBS:-}" ]]; then
  base_total=$((manifest_jobs - base_start))
else
  base_total="$TOTAL_JOBS"
fi
if ! [[ "$base_total" =~ ^[0-9]+$ ]] || (( base_total <= 0 )); then
  echo "[ERROR] TOTAL_JOBS must resolve to a positive integer, got: $base_total" >&2
  exit 1
fi
if (( base_start + base_total > manifest_jobs )); then
  echo "[ERROR] Requested DGN range exceeds manifest: start=$base_start total=$base_total manifest=$manifest_jobs" >&2
  exit 1
fi

if (( DGN_SHARD_COUNT > 1 )); then
  shard_start_offset=$((base_total * DGN_SHARD_INDEX / DGN_SHARD_COUNT))
  shard_end_offset=$((base_total * (DGN_SHARD_INDEX + 1) / DGN_SHARD_COUNT))
  START_FROM=$((base_start + shard_start_offset))
  TOTAL_JOBS=$((shard_end_offset - shard_start_offset))
  if (( TOTAL_JOBS <= 0 )); then
    echo "[ERROR] Empty shard: index=$DGN_SHARD_INDEX count=$DGN_SHARD_COUNT base_total=$base_total" >&2
    exit 1
  fi
  case "$DGN_SHARD_SPLIT_SAVE_PATH" in
    1|true|TRUE|yes|YES)
      SAVE_PATH="$BASE_SAVE_PATH/shard_${DGN_SHARD_INDEX}_of_${DGN_SHARD_COUNT}"
      CONCLUSION_PATH="${CONCLUSION_PATH:-$BASE_SAVE_PATH/conclusions}"
      ;;
    0|false|FALSE|no|NO)
      CONCLUSION_PATH="${CONCLUSION_PATH:-$SAVE_PATH/conclusions}"
      ;;
    *)
      echo "[ERROR] DGN_SHARD_SPLIT_SAVE_PATH must be 0/1 or true/false, got: $DGN_SHARD_SPLIT_SAVE_PATH" >&2
      exit 1
      ;;
  esac
else
  START_FROM="$base_start"
  TOTAL_JOBS="$base_total"
fi

echo "[INFO] Running DGN StereoVLA eval"
echo "[INFO] dgn_candidates_json=$DGN_CANDIDATES_JSON"
echo "[INFO] dgn_asset_root=$DGN_ASSET_ROOT"
echo "[INFO] manifest_jobs=$manifest_jobs"
echo "[INFO] base_start=$base_start"
echo "[INFO] base_total=$base_total"
echo "[INFO] dgn_shard_index=$DGN_SHARD_INDEX"
echo "[INFO] dgn_shard_count=$DGN_SHARD_COUNT"
if [[ -n "$DGN_OBJECT_SCALE_MIN" || -n "$DGN_OBJECT_SCALE_MAX" ]]; then
  echo "[INFO] dgn_object_scale_range=[$DGN_OBJECT_SCALE_MIN, $DGN_OBJECT_SCALE_MAX]"
  echo "[INFO] dgn_object_scale_seed=$DGN_OBJECT_SCALE_SEED"
  echo "[INFO] dgn_scale_mode=rl_style_final_xform_scale_random"
elif [[ "$DGN_OBJECT_SCALE" == "manifest" || -z "$DGN_OBJECT_SCALE" ]]; then
  echo "[INFO] dgn_object_scale=manifest"
  echo "[INFO] dgn_scale_mode=manifest_suffix"
else
  echo "[INFO] dgn_object_scale=$DGN_OBJECT_SCALE"
  echo "[INFO] dgn_scale_mode=rl_style_final_xform_scale"
fi
echo "[INFO] dgn_stable_pose=$DGN_STABLE_POSE"
echo "[INFO] dgn_stable_pose_sample_num=$DGN_STABLE_POSE_SAMPLE_NUM"
echo "[INFO] dgn_stable_pose_seed=$DGN_STABLE_POSE_SEED"
echo "[INFO] failure_retry_count=$FAILURE_RETRY_COUNT"
echo "[INFO] failure_retry_xy_radius=$FAILURE_RETRY_XY_RADIUS"
echo "[INFO] failure_retry_seed=$FAILURE_RETRY_SEED"
echo "[INFO] object_num=$OBJECT_NUM"
echo "[INFO] start_from=$START_FROM"
echo "[INFO] total_jobs=$TOTAL_JOBS"
echo "[INFO] jobs_per_gpu=$JOBS_PER_GPU"
echo "[INFO] start_servers=$START_SERVERS"
echo "[INFO] save_path=$SAVE_PATH"
echo "[INFO] conclusion_path=${CONCLUSION_PATH:-$SAVE_PATH/conclusions}"

export DGN_CANDIDATES_JSON
export DGN_ASSET_ROOT
export TOTAL_JOBS
export START_FROM
export JOBS_PER_GPU
export SAVE_PATH
export CONCLUSION_PATH
export DGN_OBJECT_SCALE
export DGN_OBJECT_SCALE_MIN
export DGN_OBJECT_SCALE_MAX
export DGN_OBJECT_SCALE_SEED
export DGN_STABLE_POSE
export DGN_STABLE_POSE_SAMPLE_NUM
export DGN_STABLE_POSE_SEED
export FAILURE_RETRY_COUNT
export FAILURE_RETRY_XY_RADIUS
export FAILURE_RETRY_SEED
export OBJECT_NUM
export START_SERVERS
exec "$SCRIPT_DIR/eval_2servers.sh" "$@"
