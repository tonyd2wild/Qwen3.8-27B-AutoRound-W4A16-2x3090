#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_env

command -v hf >/dev/null || {
  echo "Install the Hugging Face CLI: python3 -m pip install -U huggingface_hub" >&2
  exit 1
}

mkdir -p "$MODEL_DIR"
echo "Downloading $MODEL_REPO to $MODEL_DIR"
hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

if find "$MODEL_DIR" -type f -name '*.incomplete' | grep -q .; then
  echo "Incomplete download files remain in $MODEL_DIR" >&2
  exit 1
fi

python3 - "$MODEL_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
quant = config.get("quantization_config", {})
method = str(quant.get("quant_method", "")).lower().replace("_", "-")
packing = str(quant.get("packing_format", "")).lower()
if method not in {"auto-round", "autoround"} and "auto_round" not in packing:
    raise SystemExit(f"Unexpected quantization metadata: {quant}")
shards = list(root.glob("*.safetensors"))
if not shards:
    raise SystemExit("No safetensors files found")
print(f"Validated AutoRound metadata and {len(shards)} safetensors files")
PY

du -sh "$MODEL_DIR"
