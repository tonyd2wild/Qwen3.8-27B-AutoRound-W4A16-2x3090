#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_env

docker inspect -f 'name={{.Name}} status={{.State.Status}} restarts={{.RestartCount}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}' "$CONTAINER_NAME"
curl -sS --max-time 4 -o /dev/null -w 'health=%{http_code}\n' "http://127.0.0.1:$PORT/health" || true
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
echo "Topology:"
nvidia-smi topo -m
echo "Quantization/kernel and KV allocation:"
docker logs "$CONTAINER_NAME" 2>&1 | grep -E 'quantization=inc|MarlinLinearKernel|Available KV cache memory|GPU KV cache size|Maximum concurrency|Current kv cache memory' | tail -12 || true
