# Troubleshooting

## Server is not ready immediately

The initial compiled launch includes weight loading, Torch/Inductor compilation, MTP compilation, KV profiling, kernel warmup, CUDA graph capture, and multimodal warmup. Allow several minutes:

```bash
./scripts/wait-ready.sh
docker logs -f qwen38-autoround-int4
```

Check for a real crash:

```bash
docker inspect -f 'status={{.State.Status}} restarts={{.RestartCount}} oom={{.State.OOMKilled}}' qwen38-autoround-int4
```

## Wrong or slow quantization kernel

Run `./scripts/status.sh`. Logs should show both:

```text
quantization=inc
Using MarlinLinearKernel for AutoGPTQLinearMethod
```

If AutoRound metadata validation fails, verify that `MODEL_REPO` and `MODEL_DIR` point to the W4A16 AutoRound checkpoint, not FP8, GPTQ, or an incomplete download.

## Vision causes CUDA OOM

Keep the image preprocessing cap. The unbounded processor can admit images too large for safe concurrent work on 24 GB cards:

```text
MAX_IMAGE_PIXELS=1048576
MIN_IMAGE_PIXELS=65536
```

If your workload still OOMs, reduce the maximum to 786432 or 524288 and retest. A CUDA worker OOM can trigger a container restart even when Docker reports `OOMKilled=false`.

## Tools return HTTP 400 or malformed calls

The verified Qwen3.8 template uses:

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_xml
```

Do not substitute `hermes` unless the model template/image has changed and the parser is revalidated.

## Internal reasoning leaks into content

The required settings are:

```text
--reasoning-parser qwen3
--default-chat-template-kwargs {"enable_thinking":false}
```

Also ensure the client is not explicitly overriding `enable_thinking`.

## Disk is full

Model weights, Hugging Face data, Torch compilation artifacts, and Triton caches can consume tens of gigabytes. Inspect exact directories before removing anything:

```bash
df -h
du -sh "$MODEL_DIR" "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR"
```

Never delete the active model directory or broad cache paths blindly. A cache cleanup is recoverable but makes the next boot recompile.
