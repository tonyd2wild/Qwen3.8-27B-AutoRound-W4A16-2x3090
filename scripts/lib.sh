#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

load_env() {
  local root
  root="$(repo_root)"
  if [[ ! -f "$root/.env" ]]; then
    echo "Missing $root/.env. Copy .env.example to .env and edit it." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  source "$root/.env"
  set +a

  : "${CONTAINER_NAME:=qwen38-autoround-int4}"
  : "${VLLM_IMAGE:=vllm/vllm-openai:qwen38-x86_64-cu129}"
  : "${PORT:=8011}"
  : "${GPU_IDS:=0,1}"
  : "${TOPOLOGY:=auto}"
  : "${MODEL_REPO:=Vishva007/Qwen3.8-27B-W4A16-AutoRound}"
  : "${MODEL_DIR:=/opt/models/qwen3.8-27b-w4a16-autoround}"
  : "${SERVED_MODEL_NAME:=qwen3.8-27b}"
  : "${MAX_MODEL_LEN:=262144}"
  : "${GPU_MEMORY_UTILIZATION:=0.90}"
  : "${MAX_NUM_SEQS:=16}"
  : "${MAX_NUM_BATCHED_TOKENS:=8192}"
  : "${NUM_SPECULATIVE_TOKENS:=3}"
  : "${MAX_IMAGE_PIXELS:=1048576}"
  : "${MIN_IMAGE_PIXELS:=65536}"
  : "${VLLM_CACHE_DIR:=/opt/qwen38-autoround-cache/vllm}"
  : "${TRITON_CACHE_DIR:=/opt/qwen38-autoround-cache/triton}"
  : "${REPLACE_CONTAINER:=0}"
}

split_gpu_ids() {
  IFS=',' read -r GPU_A GPU_B GPU_EXTRA <<<"$GPU_IDS"
  if [[ -z "${GPU_A:-}" || -z "${GPU_B:-}" || -n "${GPU_EXTRA:-}" ]]; then
    echo "GPU_IDS must contain exactly two comma-separated GPU indices." >&2
    exit 1
  fi
}

detect_topology() {
  split_gpu_ids
  local cell
  cell="$(nvidia-smi topo -m | awk -v row="GPU${GPU_A}" -v col=$((GPU_B + 2)) '$1 == row {print $col}')"
  if [[ "$cell" =~ ^NV[0-9]+$ ]]; then
    printf 'nvlink\n'
  else
    printf 'pcie\n'
  fi
}

resolved_topology() {
  case "$TOPOLOGY" in
    auto) detect_topology ;;
    nvlink|pcie) printf '%s\n' "$TOPOLOGY" ;;
    *) echo "TOPOLOGY must be auto, nvlink, or pcie." >&2; exit 1 ;;
  esac
}
