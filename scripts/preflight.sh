#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_env
split_gpu_ids

for command in docker nvidia-smi curl python3; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done

docker info >/dev/null
mapfile -t gpu_names < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)
gpu_count="${#gpu_names[@]}"
for id in "$GPU_A" "$GPU_B"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $id" >&2; exit 1; }
  (( id < gpu_count )) || { echo "GPU $id does not exist; found $gpu_count GPUs." >&2; exit 1; }
  echo "Selected: ${gpu_names[$id]}"
done

actual="$(detect_topology)"
selected="$(resolved_topology)"
echo "Detected topology: $actual"
echo "Selected profile:  $selected"
if [[ "$selected" == nvlink && "$actual" != nvlink ]]; then
  echo "NVLink requested but no NV# link exists between GPU $GPU_A and GPU $GPU_B." >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model is missing. Run ./scripts/download-model.sh" >&2
  exit 1
fi

python3 - "$MODEL_DIR/config.json" <<'PY'
import json
import sys
q = json.load(open(sys.argv[1])).get("quantization_config", {})
method = str(q.get("quant_method", "")).lower().replace("_", "-")
packing = str(q.get("packing_format", "")).lower()
if method not in {"auto-round", "autoround"} and "auto_round" not in packing:
    raise SystemExit(f"Model is not an AutoRound checkpoint: {q}")
print("AutoRound quantization metadata: OK")
PY

mkdir -p "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR"
docker run --rm --gpus "device=$GPU_A" "$VLLM_IMAGE" --version >/dev/null
echo "Preflight passed."
