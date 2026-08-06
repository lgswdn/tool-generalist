#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run StereoVLA eval with two local model servers and multiple eval GPUs.

Default topology:
  server GPU 0 -> port 6666 -> eval GPUs 2,3,4
  server GPU 1 -> port 6667 -> eval GPUs 5,6,7

Example:
  TOTAL_JOBS=56 ./scripts/stereovla/eval_2servers.sh
  DGN_CANDIDATES_JSON=/mnt/project/world_model/tool_generalist/assets/DGN/full_yes.json TOTAL_JOBS=5676 JOBS_PER_GPU=8 ./scripts/stereovla/eval_2servers.sh
  START_SERVERS=0 TOTAL_JOBS=56 ./scripts/stereovla/eval_2servers.sh

Environment:
  START_SERVERS        Start/stop model servers inside this script. Default: 1. Set 0 to reuse existing ports.
  SERVER_GPU_IDS       Server GPUs. Default: 0,1
  SERVER_PORTS         Server ports. Default: 6666,6667
  EVAL_GPU_IDS         Eval GPUs. Default: 2,3,4,5,6,7
  WORKERS_PER_SERVER   Eval workers assigned to each server. Default: 3
  TOTAL_JOBS           Total scenes/jobs to evaluate. Defaults to MAX_SCENES, then NUM_JOBS, then 48.
  JOBS_PER_GPU         Jobs per eval worker per wave. Defaults to ceil(TOTAL_JOBS / number_of_eval_gpus).
  START_FROM           Global dataset offset. Default: 0
  SAVE_PATH            Base output directory. Default: /mnt/project/world_model/tool_generalist/grasp_result_2
  CONCLUSION_PATH      Shared live conclusion directory. Default: SAVE_PATH/conclusions
  RESUME                Set to 1 to run only unfinished DGN manifest indices from episodes.jsonl.
  DGN_CANDIDATES_JSON  Optional DGN manifest for replacing the template target object.
  SERVER_BATCH_SIZE    StereoVLA server batch size. Defaults to JOBS_PER_GPU.
  SERVER_START_TIMEOUT Seconds to wait for server ports. Default: 900
  STATUS_INTERVAL      Seconds between compact status prints. Default: 30
  LOG_TAIL_LINES       Lines shown per process in status prints. Default: 1

All other eval/server environment variables are passed through.
EOF
  exit 0
fi
passthrough_args=("$@")

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a parsed_list <<< "$raw"
}

parse_list "${SERVER_GPU_IDS:-0,1}"
server_gpu_ids=("${parsed_list[@]}")
parse_list "${SERVER_PORTS:-6666,6667}"
server_ports=("${parsed_list[@]}")
parse_list "${EVAL_GPU_IDS:-2,3,4,5,6,7}"
eval_gpu_ids=("${parsed_list[@]}")

start_servers="${START_SERVERS:-1}"
case "$start_servers" in
  1|true|TRUE|yes|YES) start_servers=1 ;;
  0|false|FALSE|no|NO) start_servers=0 ;;
  *)
    echo "[ERROR] START_SERVERS must be 0/1 or true/false, got: $start_servers" >&2
    exit 1
    ;;
esac

if (( ${#server_gpu_ids[@]} == 0 )); then
  echo "[ERROR] SERVER_GPU_IDS is empty" >&2
  exit 1
fi
if (( ${#server_ports[@]} != ${#server_gpu_ids[@]} )); then
  echo "[ERROR] SERVER_PORTS count (${#server_ports[@]}) must match SERVER_GPU_IDS count (${#server_gpu_ids[@]})" >&2
  exit 1
fi
if (( ${#eval_gpu_ids[@]} == 0 )); then
  echo "[ERROR] EVAL_GPU_IDS is empty" >&2
  exit 1
fi

workers_per_server="${WORKERS_PER_SERVER:-3}"
if ! [[ "$workers_per_server" =~ ^[0-9]+$ ]] || (( workers_per_server <= 0 )); then
  echo "[ERROR] WORKERS_PER_SERVER must be a positive integer, got: $workers_per_server" >&2
  exit 1
fi

server_count=${#server_gpu_ids[@]}
eval_gpu_count=${#eval_gpu_ids[@]}
server_capacity=$((server_count * workers_per_server))
if (( eval_gpu_count > server_capacity )); then
  echo "[ERROR] $eval_gpu_count eval GPUs need more than $server_count servers with WORKERS_PER_SERVER=$workers_per_server" >&2
  exit 1
fi

if [[ -n "${TOTAL_JOBS:-}" ]]; then
  total_jobs="$TOTAL_JOBS"
elif [[ -n "${MAX_SCENES:-}" ]]; then
  total_jobs="$MAX_SCENES"
elif [[ -n "${NUM_JOBS:-}" ]]; then
  total_jobs="$NUM_JOBS"
else
  total_jobs=48
fi
if ! [[ "$total_jobs" =~ ^[0-9]+$ ]] || (( total_jobs <= 0 )); then
  echo "[ERROR] TOTAL_JOBS/MAX_SCENES/NUM_JOBS must resolve to a positive integer, got: $total_jobs" >&2
  exit 1
fi

if [[ -n "${JOBS_PER_GPU:-}" ]]; then
  jobs_per_gpu="$JOBS_PER_GPU"
  if ! [[ "$jobs_per_gpu" =~ ^[0-9]+$ ]] || (( jobs_per_gpu <= 0 )); then
    echo "[ERROR] JOBS_PER_GPU must be a positive integer, got: $jobs_per_gpu" >&2
    exit 1
  fi
else
  jobs_per_gpu=$(((total_jobs + eval_gpu_count - 1) / eval_gpu_count))
fi

base_start="${START_FROM:-0}"
if ! [[ "$base_start" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] START_FROM must be a non-negative integer, got: $base_start" >&2
  exit 1
fi

base_save_path="${SAVE_PATH:-/mnt/project/world_model/tool_generalist/grasp_result_2}"
base_conclusion_path="${CONCLUSION_PATH:-$base_save_path/conclusions}"
logs_dir="$base_save_path/logs"
mkdir -p "$logs_dir"
mkdir -p "$base_conclusion_path"

resume="${RESUME:-0}"
case "$resume" in
  1|true|TRUE|yes|YES) resume=1 ;;
  0|false|FALSE|no|NO) resume=0 ;;
  *)
    echo "[ERROR] RESUME must be 0/1 or true/false, got: $resume" >&2
    exit 1
    ;;
esac

job_indices=()
if (( resume == 1 )); then
  if [[ -z "${DGN_CANDIDATES_JSON:-}" ]]; then
    echo "[ERROR] RESUME=1 currently requires DGN_CANDIDATES_JSON" >&2
    exit 1
  fi
  mapfile -t job_indices < <(
    python "$REPO_ROOT/scripts/pending_stereovla_dgn_indices.py" \
      --episodes "$base_conclusion_path/episodes.jsonl" \
      --start "$base_start" \
      --count "$total_jobs" \
      --retry-count "${FAILURE_RETRY_COUNT:-0}"
  )
  requested_jobs="$total_jobs"
  total_jobs=${#job_indices[@]}
  echo "[INFO] resume scan: completed=$((requested_jobs - total_jobs)) pending=$total_jobs range=[$base_start,$((base_start + requested_jobs)))"
  if (( total_jobs == 0 )); then
    echo "[INFO] shard is already complete"
    exit 0
  fi
fi

server_batch_size="${SERVER_BATCH_SIZE:-$jobs_per_gpu}"
server_start_timeout="${SERVER_START_TIMEOUT:-900}"
status_interval="${STATUS_INTERVAL:-30}"
log_tail_lines="${LOG_TAIL_LINES:-1}"
for name_value in \
  "SERVER_BATCH_SIZE=$server_batch_size" \
  "SERVER_START_TIMEOUT=$server_start_timeout" \
  "STATUS_INTERVAL=$status_interval" \
  "LOG_TAIL_LINES=$log_tail_lines"; do
  name="${name_value%%=*}"
  value="${name_value#*=}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "[ERROR] $name must be a positive integer, got: $value" >&2
    exit 1
  fi
done

echo "[INFO] Running StereoVLA eval with local servers"
echo "[INFO] start_servers=$start_servers"
echo "[INFO] server_gpus=${server_gpu_ids[*]}"
echo "[INFO] server_ports=${server_ports[*]}"
echo "[INFO] eval_gpus=${eval_gpu_ids[*]}"
echo "[INFO] workers_per_server=$workers_per_server"
echo "[INFO] total_jobs=$total_jobs"
echo "[INFO] resume=$resume"
echo "[INFO] jobs_per_gpu=$jobs_per_gpu"
echo "[INFO] waves=$(((total_jobs + jobs_per_gpu * eval_gpu_count - 1) / (jobs_per_gpu * eval_gpu_count)))"
echo "[INFO] server_batch_size=$server_batch_size"
echo "[INFO] start_from=$base_start"
echo "[INFO] base_save_path=$base_save_path"
echo "[INFO] conclusion_path=$base_conclusion_path"

server_pids=()
server_labels=()
server_logs=()
eval_pids=()
eval_labels=()
eval_logs=()
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
  for pid in "${eval_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${server_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local pid="$4"
  local log_file="$5"
  local start_s
  start_s="$(date +%s)"
  while true; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] server port $port process exited before becoming ready. log=$log_file" >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
      return 0
    fi
    now_s="$(date +%s)"
    if (( now_s - start_s >= timeout_s )); then
      echo "[ERROR] timed out waiting for server port $port after ${timeout_s}s. log=$log_file" >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done
}

wait_for_existing_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local start_s
  start_s="$(date +%s)"
  while true; do
    if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
      return 0
    fi
    now_s="$(date +%s)"
    if (( now_s - start_s >= timeout_s )); then
      echo "[ERROR] timed out waiting for existing server at $host:$port after ${timeout_s}s" >&2
      return 1
    fi
    sleep 2
  done
}

if (( start_servers == 1 )); then
  for idx in "${!server_gpu_ids[@]}"; do
    gpu_id="${server_gpu_ids[$idx]}"
    port="${server_ports[$idx]}"
    log_file="$logs_dir/server_${idx}_gpu_${gpu_id}_port_${port}.log"
    if (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1; then
      echo "[ERROR] server port $port is already open. Stop the existing server, choose another SERVER_PORTS value, or set START_SERVERS=0." >&2
      exit 1
    fi
    echo "[INFO] launching server=$idx gpu=$gpu_id port=$port log=$log_file"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export PORT="$port"
      export BATCH_SIZE="$server_batch_size"
      exec "$SCRIPT_DIR/server.sh"
    ) >"$log_file" 2>&1 &
    server_pids+=("$!")
    server_labels+=("server=$idx gpu=$gpu_id port=$port")
    server_logs+=("$log_file")
  done

  for idx in "${!server_pids[@]}"; do
    wait_for_port "127.0.0.1" "${server_ports[$idx]}" "$server_start_timeout" "${server_pids[$idx]}" "${server_logs[$idx]}"
    echo "[INFO] ready ${server_labels[$idx]}"
  done
else
  for idx in "${!server_ports[@]}"; do
    port="${server_ports[$idx]}"
    server_labels+=("existing_server=$idx port=$port")
    server_logs+=("")
    echo "[INFO] waiting for existing server=$idx port=$port"
    wait_for_existing_port "127.0.0.1" "$port" "$server_start_timeout"
    echo "[INFO] ready ${server_labels[$idx]}"
  done
fi

remaining=$total_jobs
next_start=$base_start
next_job_index=0
if (( start_servers == 1 )); then
  echo "[INFO] server logs:"
  for log_file in "${server_logs[@]}"; do
    echo "[INFO]   $log_file"
  done
else
  echo "[INFO] using existing servers; this script will not stop them"
fi
echo "[INFO] to follow logs manually: tail -f $logs_dir/*.log"

launch_eval_wave() {
  local wave="$1"
  wave_pids=()
  wave_labels=()
  wave_logs=()
  for idx in "${!eval_gpu_ids[@]}"; do
    if (( remaining <= 0 )); then
      break
    fi

    gpu_id="${eval_gpu_ids[$idx]}"
    server_idx=$((idx / workers_per_server))
    if (( server_idx >= server_count )); then
      echo "[ERROR] eval worker $idx has no server assignment" >&2
      exit 1
    fi
    server_port="${server_ports[$server_idx]}"

    worker_jobs=$jobs_per_gpu
    if (( worker_jobs > remaining )); then
      worker_jobs=$remaining
    fi

    worker_start=$next_start
    worker_dgn_indices=""
    if (( resume == 1 )); then
      worker_start="${job_indices[$next_job_index]}"
      worker_slice=("${job_indices[@]:$next_job_index:$worker_jobs}")
      worker_dgn_indices="$(IFS=,; echo "${worker_slice[*]}")"
    fi
    worker_save_path="$base_save_path/wave_${wave}_eval_gpu_${gpu_id}_server_${server_idx}_start_${worker_start}"
    log_file="$logs_dir/wave_${wave}_eval_gpu_${gpu_id}_server_${server_idx}_start_${worker_start}.log"

    echo "[INFO] launching wave=$wave eval=$idx gpu=$gpu_id server_port=$server_port start_from=$worker_start num_jobs=$worker_jobs save_path=$worker_save_path"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export SERVER_IP="127.0.0.1"
      export SERVER_PORT="$server_port"
      export START_FROM="$worker_start"
      export DGN_INDICES="$worker_dgn_indices"
      export NUM_JOBS="$worker_jobs"
      export MAX_SCENES="$worker_jobs"
      export SAVE_PATH="$worker_save_path"
      export CONCLUSION_PATH="$base_conclusion_path"
      exec "$SCRIPT_DIR/eval.sh" "${passthrough_args[@]}"
    ) >"$log_file" 2>&1 &

    pid="$!"
    label="wave=$wave eval=$idx gpu=$gpu_id server_port=$server_port start_from=$worker_start num_jobs=$worker_jobs"
    eval_pids+=("$pid")
    eval_labels+=("$label")
    eval_logs+=("$log_file")
    wave_pids+=("$pid")
    wave_labels+=("$label")
    wave_logs+=("$log_file")

    remaining=$((remaining - worker_jobs))
    next_start=$((next_start + worker_jobs))
    next_job_index=$((next_job_index + worker_jobs))
  done
}

monitor_processes() {
  while true; do
    sleep "$status_interval"
    echo "[INFO] process status:"
    for idx in "${!server_pids[@]}"; do
      state="exited"
      if kill -0 "${server_pids[$idx]}" 2>/dev/null; then
        state="running"
      fi
      echo "[INFO]   ${server_labels[$idx]} status=$state log=${server_logs[$idx]}"
      tail -n "$log_tail_lines" "${server_logs[$idx]}" 2>/dev/null | sed 's/^/[INFO]     /' || true
    done
    if (( start_servers == 0 )); then
      for idx in "${!server_labels[@]}"; do
        echo "[INFO]   ${server_labels[$idx]} status=external"
      done
    fi
    for idx in "${!eval_pids[@]}"; do
      state="exited"
      if kill -0 "${eval_pids[$idx]}" 2>/dev/null; then
        state="running"
      fi
      echo "[INFO]   ${eval_labels[$idx]} status=$state log=${eval_logs[$idx]}"
      tail -n "$log_tail_lines" "${eval_logs[$idx]}" 2>/dev/null | sed 's/^/[INFO]     /' || true
    done
  done
}

monitor_processes &
monitor_pid=$!

failed=0
wave=0
while (( remaining > 0 )); do
  launch_eval_wave "$wave"
  for idx in "${!wave_pids[@]}"; do
    if wait "${wave_pids[$idx]}"; then
      echo "[INFO] completed ${wave_labels[$idx]} log=${wave_logs[$idx]}"
    else
      status=$?
      failed=1
      echo "[ERROR] failed ${wave_labels[$idx]} status=$status log=${wave_logs[$idx]}" >&2
      tail -n 120 "${wave_logs[$idx]}" >&2 || true
    fi
  done
  if (( failed != 0 )); then
    break
  fi
  echo "[INFO] wave=$wave completed remaining=$remaining next_start=$next_start"
  wave=$((wave + 1))
done

if (( failed != 0 )); then
  exit 1
fi

echo "[INFO] all eval workers completed"
if (( start_servers == 1 )); then
  echo "[INFO] stopping servers"
  for pid in "${server_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
else
  echo "[INFO] leaving existing servers running"
fi
trap - EXIT INT TERM
if [[ -n "$monitor_pid" ]]; then
  kill "$monitor_pid" 2>/dev/null || true
fi
echo "[INFO] outputs are under $base_save_path"
