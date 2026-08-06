#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Launch persistent StereoVLA servers.

Default topology:
  GPU 0 -> port 6666
  GPU 1 -> port 6667

Example:
  ./scripts/stereovla/servers_2.sh
  SERVER_GPU_IDS=4,5 SERVER_PORTS=6766,6767 SERVER_BATCH_SIZE=8 ./scripts/stereovla/servers_2.sh

Environment:
  SERVER_GPU_IDS        Server GPUs. Default: 0,1
  SERVER_PORTS          Server ports. Default: 6666,6667
  SERVER_BATCH_SIZE     Batch size passed to each server. Default: 8
  SERVER_START_TIMEOUT  Seconds to wait for ports. Default: 900
  STATUS_INTERVAL       Seconds between health checks. Default: 30
  SERVER_LOG_DIR        Log directory. Default: /mnt/project/world_model/tool_generalist/stereovla_server_logs

Stop the servers with Ctrl-C.
EOF
  exit 0
fi

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a parsed_list <<< "$raw"
}

parse_list "${SERVER_GPU_IDS:-0,1}"
server_gpu_ids=("${parsed_list[@]}")
parse_list "${SERVER_PORTS:-6666,6667}"
server_ports=("${parsed_list[@]}")

if (( ${#server_gpu_ids[@]} == 0 )); then
  echo "[ERROR] SERVER_GPU_IDS is empty" >&2
  exit 1
fi
if (( ${#server_ports[@]} != ${#server_gpu_ids[@]} )); then
  echo "[ERROR] SERVER_PORTS count (${#server_ports[@]}) must match SERVER_GPU_IDS count (${#server_gpu_ids[@]})" >&2
  exit 1
fi

server_batch_size="${SERVER_BATCH_SIZE:-8}"
server_start_timeout="${SERVER_START_TIMEOUT:-900}"
status_interval="${STATUS_INTERVAL:-30}"
logs_dir="${SERVER_LOG_DIR:-/mnt/project/world_model/tool_generalist/stereovla_server_logs}"
for name_value in \
  "SERVER_BATCH_SIZE=$server_batch_size" \
  "SERVER_START_TIMEOUT=$server_start_timeout" \
  "STATUS_INTERVAL=$status_interval"; do
  name="${name_value%%=*}"
  value="${name_value#*=}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "[ERROR] $name must be a positive integer, got: $value" >&2
    exit 1
  fi
done

mkdir -p "$logs_dir"

server_pids=()
server_labels=()
server_logs=()

cleanup() {
  echo "[INFO] stopping StereoVLA servers"
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

echo "[INFO] Launching persistent StereoVLA servers"
echo "[INFO] server_gpus=${server_gpu_ids[*]}"
echo "[INFO] server_ports=${server_ports[*]}"
echo "[INFO] server_batch_size=$server_batch_size"
echo "[INFO] logs_dir=$logs_dir"

for idx in "${!server_gpu_ids[@]}"; do
  gpu_id="${server_gpu_ids[$idx]}"
  port="${server_ports[$idx]}"
  log_file="$logs_dir/server_${idx}_gpu_${gpu_id}_port_${port}.log"
  if (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1; then
    echo "[ERROR] server port $port is already open. Stop the existing server or choose another SERVER_PORTS value." >&2
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

echo "[INFO] servers are ready"
echo "[INFO] eval command can reuse them with: START_SERVERS=0 ./scripts/stereovla/eval_2servers.sh"
echo "[INFO] logs:"
for log_file in "${server_logs[@]}"; do
  echo "[INFO]   $log_file"
done

while true; do
  sleep "$status_interval"
  for idx in "${!server_pids[@]}"; do
    if ! kill -0 "${server_pids[$idx]}" 2>/dev/null; then
      echo "[ERROR] ${server_labels[$idx]} exited. log=${server_logs[$idx]}" >&2
      tail -n 120 "${server_logs[$idx]}" >&2 || true
      exit 1
    fi
  done
  echo "[INFO] servers still running: ${server_labels[*]}"
done
