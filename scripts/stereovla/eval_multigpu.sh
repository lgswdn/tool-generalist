#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run StereoVLA GraspSimulator eval across multiple GPUs.

Examples:
  GPU_IDS=0,1,2,3,4,5,6 TOTAL_JOBS=56 ./scripts/stereovla/eval_multigpu.sh
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NUM_JOBS=56 ./scripts/stereovla/eval_multigpu.sh

Environment:
  GPU_IDS        Comma or space separated GPU ids. Defaults to CUDA_VISIBLE_DEVICES, then 0.
  TOTAL_JOBS     Total scenes/jobs to evaluate. Defaults to MAX_SCENES, then NUM_JOBS, then 20.
  JOBS_PER_GPU   Jobs per worker. Defaults to ceil(TOTAL_JOBS / number_of_gpus).
  START_FROM     Global dataset offset. Defaults to 0.
  SAVE_PATH      Base output directory. Each worker writes under this directory.
  STATUS_INTERVAL Seconds between compact worker status prints. Defaults to 30.
  LOG_TAIL_LINES Number of log lines shown per worker in status prints. Defaults to 1.

All other environment variables are passed through to scripts/stereovla/eval.sh.
EOF
  exit 0
fi

gpu_list="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
gpu_list="${gpu_list//,/ }"
read -r -a gpu_ids <<< "$gpu_list"
if (( ${#gpu_ids[@]} == 0 )); then
  echo "[ERROR] No GPUs selected. Set GPU_IDS=0,1,... or CUDA_VISIBLE_DEVICES=0,1,..." >&2
  exit 1
fi

if [[ -n "${TOTAL_JOBS:-}" ]]; then
  total_jobs="$TOTAL_JOBS"
elif [[ -n "${MAX_SCENES:-}" ]]; then
  total_jobs="$MAX_SCENES"
elif [[ -n "${NUM_JOBS:-}" ]]; then
  total_jobs="$NUM_JOBS"
else
  total_jobs=20
fi

if ! [[ "$total_jobs" =~ ^[0-9]+$ ]] || (( total_jobs <= 0 )); then
  echo "[ERROR] TOTAL_JOBS/MAX_SCENES/NUM_JOBS must resolve to a positive integer, got: $total_jobs" >&2
  exit 1
fi

gpu_count=${#gpu_ids[@]}
if [[ -n "${JOBS_PER_GPU:-}" ]]; then
  jobs_per_gpu="$JOBS_PER_GPU"
  if ! [[ "$jobs_per_gpu" =~ ^[0-9]+$ ]] || (( jobs_per_gpu <= 0 )); then
    echo "[ERROR] JOBS_PER_GPU must be a positive integer, got: $jobs_per_gpu" >&2
    exit 1
  fi
  capacity=$((jobs_per_gpu * gpu_count))
  if (( capacity < total_jobs )); then
    echo "[ERROR] JOBS_PER_GPU=$jobs_per_gpu across $gpu_count GPUs only covers $capacity jobs, but TOTAL_JOBS=$total_jobs" >&2
    exit 1
  fi
else
  jobs_per_gpu=$(((total_jobs + gpu_count - 1) / gpu_count))
fi

base_start="${START_FROM:-0}"
if ! [[ "$base_start" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] START_FROM must be a non-negative integer, got: $base_start" >&2
  exit 1
fi

base_save_path="${SAVE_PATH:-/mnt/project/world_model/tool_generalist/grasp_result_2}"
base_conclusion_path="${CONCLUSION_PATH:-$base_save_path/conclusions}"
mkdir -p "$base_save_path/logs"
mkdir -p "$base_conclusion_path"

echo "[INFO] Running multi-GPU StereoVLA eval"
echo "[INFO] gpus=${gpu_ids[*]}"
echo "[INFO] total_jobs=$total_jobs"
echo "[INFO] jobs_per_gpu=$jobs_per_gpu"
echo "[INFO] start_from=$base_start"
echo "[INFO] base_save_path=$base_save_path"
echo "[INFO] conclusion_path=$base_conclusion_path"

pids=()
labels=()
logs=()
remaining=$total_jobs
next_start=$base_start

for idx in "${!gpu_ids[@]}"; do
  if (( remaining <= 0 )); then
    break
  fi

  gpu_id="${gpu_ids[$idx]}"
  worker_jobs=$jobs_per_gpu
  if (( worker_jobs > remaining )); then
    worker_jobs=$remaining
  fi

  worker_start=$next_start
  worker_save_path="$base_save_path/gpu_${idx}_start_${worker_start}"
  log_file="$base_save_path/logs/gpu_${idx}_start_${worker_start}.log"

  echo "[INFO] launching worker=$idx gpu=$gpu_id start_from=$worker_start num_jobs=$worker_jobs save_path=$worker_save_path"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export START_FROM="$worker_start"
    export NUM_JOBS="$worker_jobs"
    export MAX_SCENES="$worker_jobs"
    export SAVE_PATH="$worker_save_path"
    export CONCLUSION_PATH="$base_conclusion_path"
    exec "$SCRIPT_DIR/eval.sh" "$@"
  ) >"$log_file" 2>&1 &

  pids+=("$!")
  labels+=("worker=$idx gpu=$gpu_id start_from=$worker_start num_jobs=$worker_jobs")
  logs+=("$log_file")

  remaining=$((remaining - worker_jobs))
  next_start=$((next_start + worker_jobs))
done

echo "[INFO] worker logs:"
for log_file in "${logs[@]}"; do
  echo "[INFO]   $log_file"
done
echo "[INFO] to follow logs manually: tail -f $base_save_path/logs/*.log"

status_interval="${STATUS_INTERVAL:-30}"
log_tail_lines="${LOG_TAIL_LINES:-1}"
if ! [[ "$status_interval" =~ ^[0-9]+$ ]] || (( status_interval <= 0 )); then
  echo "[ERROR] STATUS_INTERVAL must be a positive integer, got: $status_interval" >&2
  exit 1
fi
if ! [[ "$log_tail_lines" =~ ^[0-9]+$ ]] || (( log_tail_lines <= 0 )); then
  echo "[ERROR] LOG_TAIL_LINES must be a positive integer, got: $log_tail_lines" >&2
  exit 1
fi

monitor_workers() {
  while true; do
    sleep "$status_interval"
    echo "[INFO] worker status:"
    for idx in "${!pids[@]}"; do
      state="exited"
      if kill -0 "${pids[$idx]}" 2>/dev/null; then
        state="running"
      fi
      echo "[INFO]   ${labels[$idx]} status=$state log=${logs[$idx]}"
      tail -n "$log_tail_lines" "${logs[$idx]}" 2>/dev/null | sed 's/^/[INFO]     /' || true
    done
  done
}

monitor_workers &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

failed=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[INFO] completed ${labels[$idx]} log=${logs[$idx]}"
  else
    status=$?
    failed=1
    echo "[ERROR] failed ${labels[$idx]} status=$status log=${logs[$idx]}" >&2
    tail -n 80 "${logs[$idx]}" >&2 || true
  fi
done
cleanup_monitor
trap - EXIT

if (( failed != 0 )); then
  exit 1
fi

echo "[INFO] all workers completed"
echo "[INFO] outputs are under $base_save_path"
