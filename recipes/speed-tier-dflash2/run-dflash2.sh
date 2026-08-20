#!/usr/bin/env bash
# Speed tier: Qwen3.8-27B AutoRound W4A16 + DFlash2 drafter on 2x RTX 3090.
# Applies the vllm-dflash2 overlay (ports vllm-project/vllm#52816) at boot and hard-fails
# on anchor drift, so a re-pinned image can never serve a half-wired drafter.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$HERE/speed-tier-dflash2-2x3090.env}"
[ -f "$ENV_FILE" ] || { echo "env file not found: $ENV_FILE" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a

: "${CONTAINER_NAME:?}" "${VLLM_IMAGE:?}" "${PORT:?}" "${GPU_IDS:?}" "${MODEL_DIR:?}" "${DRAFTER_DIR:?}"
MODELS_PARENT="$(dirname "$MODEL_DIR")"
TARGET_NAME="$(basename "$MODEL_DIR")"
DRAFTER_NAME="$(basename "$DRAFTER_DIR")"
[ -f "$MODEL_DIR/config.json" ]   || { echo "target weights missing at $MODEL_DIR (download $MODEL_REPO)" >&2; exit 1; }
[ -f "$DRAFTER_DIR/config.json" ] || { echo "drafter weights missing at $DRAFTER_DIR (download $DRAFTER_REPO)" >&2; exit 1; }
[ "$(dirname "$DRAFTER_DIR")" = "$MODELS_PARENT" ] || { echo "target and drafter must share a parent dir" >&2; exit 1; }

if [ "${REPLACE_CONTAINER:-0}" = "1" ]; then docker rm -f "$CONTAINER_NAME" 2>/dev/null || true; fi
docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && {
  echo "container $CONTAINER_NAME exists; set REPLACE_CONTAINER=1 to recreate" >&2; exit 1; }

docker run -d --name "$CONTAINER_NAME" --restart unless-stopped \
  --gpus "\"device=${GPU_IDS}\"" --ipc host -p "${PORT}:8000" \
  -e CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  -e W4A8="${W4A8:-0}" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_NO_USAGE_STATS=1 -e VLLM_SKIP_P2P_CHECK=1 \
  -v "${MODELS_PARENT}:/root/.cache/huggingface" \
  -v "${HERE}/patches/vllm-dflash2:/etc/club3090/dflash2:ro" \
  -v "${HERE}/patches/w4a8-int8-act:/etc/club3090/w4a8:ro" \
  -v "${HERE}/patches/pr48375-mamba-drop-eagle-block:/etc/club3090/pr48375:ro" \
  --entrypoint bash "$VLLM_IMAGE" -c '
    set -e
    # W4A8=0 -> W4A16 path (int8-activation marlin is AutoRound-INT4 only).
    if [ -z "${VLLM_MARLIN_INPUT_DTYPE:-}" ] && [ "${W4A8:-0}" != "0" ]; then export VLLM_MARLIN_INPUT_DTYPE=int8; fi
    bash /etc/club3090/w4a8/install.sh
    bash /etc/club3090/pr48375/install.sh
    bash /etc/club3090/dflash2/install.sh   # hard-fails on anchor drift
    export PYTHONPATH="/etc/club3090/dflash2${PYTHONPATH:+:$PYTHONPATH}"
    exec vllm serve "$@"
  ' -- \
    --model "/root/.cache/huggingface/${TARGET_NAME}" \
    --served-model-name "${SERVED_MODEL_NAME:-qwen3.8-27b}" \
    --quantization "${QUANTIZATION:-auto_round}" --dtype bfloat16 \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
    --max-num-seqs "${MAX_NUM_SEQS:-16}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
    --trust-remote-code --enable-prefix-caching --enable-chunked-prefill \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs "{\"enable_thinking\": false}" --enable-flashinfer-autotune \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/root/.cache/huggingface/${DRAFTER_NAME}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS:-7}}"

echo "started ${CONTAINER_NAME} on :${PORT} (served model: ${SERVED_MODEL_NAME:-qwen3.8-27b})"
echo "watch load:  docker logs -f ${CONTAINER_NAME}"
echo "ready check: curl -s http://127.0.0.1:${PORT}/v1/models"
