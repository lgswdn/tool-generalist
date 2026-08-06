#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_ids="${GPU_IDS:-0,1,2,3}"
epochs="${EPOCHS:-20}"
batch_size="${BATCH_SIZE:-4096}"
output_root="${OUTPUT_ROOT:-${repo_root}/artifacts/probes/pointnet_normalization_bottleneck_ablation}"
IFS=',' read -r -a gpus <<< "${gpu_ids}"
variants=(
  normalized_direct128
  normalized_rank10
  unnormalized_direct128
  unnormalized_rank10
)

if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi
if [[ -e "${output_root}" ]]; then
  echo "Refusing to overwrite existing output root: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}/logs"

pids=()
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  gpu="${gpus[$index]}"
  echo "[offline-ablation] gpu=${gpu} variant=${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" python \
    "${repo_root}/scripts/train_pointnet_normalization_bottleneck_ablation.py" \
    --variant "${variant}" \
    --device cuda \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --output-root "${output_root}" \
    2>&1 | tee "${output_root}/logs/${variant}.log" &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[offline-ablation] failed: ${variants[$index]}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

python "${repo_root}/scripts/summarize_pointnet_normalization_bottleneck_ablation.py" \
  --output-root "${output_root}"
