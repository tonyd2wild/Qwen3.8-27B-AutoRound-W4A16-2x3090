#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
replace_override="${REPLACE_CONTAINER:-}"
load_env
if [[ -n "$replace_override" ]]; then
  REPLACE_CONTAINER="$replace_override"
fi

profile="$(resolved_topology)"
mkdir -p "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [[ "$REPLACE_CONTAINER" != 1 ]]; then
    echo "Container $CONTAINER_NAME already exists." >&2
    echo "Set REPLACE_CONTAINER=1 only when you intend to replace it." >&2
    exit 1
  fi
  echo "Replacing existing container $CONTAINER_NAME"
  docker update --restart=no "$CONTAINER_NAME" >/dev/null || true
  docker stop "$CONTAINER_NAME" >/dev/null || true
  docker rm "$CONTAINER_NAME" >/dev/null
fi

topology_args=()
if [[ "$profile" == nvlink ]]; then
  # This selects NCCL/PyNCCL collectives; NCCL still uses the NVLink hardware.
  topology_args+=(--disable-custom-all-reduce)
fi

echo "Launching $SERVED_MODEL_NAME on GPUs $GPU_IDS using $profile"
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus "\"device=$GPU_IDS\"" \
  --ipc host \
  --shm-size=16g \
  --network host \
  --restart unless-stopped \
  -e VLLM_USE_STANDALONE_COMPILE=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=0 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$VLLM_CACHE_DIR:/root/.cache/vllm" \
  -v "$TRITON_CACHE_DIR:/root/.triton/cache" \
  "$VLLM_IMAGE" \
  /model \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --quantization inc \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype fp8 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mm-processor-kwargs "{\"max_pixels\":$MAX_IMAGE_PIXELS,\"min_pixels\":$MIN_IMAGE_PIXELS}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}" \
  "${topology_args[@]}"

echo "Container started. Run ./scripts/wait-ready.sh"
