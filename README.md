# Qwen3.8-27B AutoRound W4A16 on 2x RTX 3090

Production-oriented Docker deployment for [`Vishva007/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound) on two 24 GB RTX 3090 GPUs.

The repository supports paired 3090s with NVLink and ordinary PCIe-only pairs. It exposes an OpenAI-compatible vLLM API with compiled CUDA graphs, speculative decoding, vision, structured tool calling, and separated reasoning.

## Two tiers, pick your drafter

The same W4A16 weights run under two speculative-decoding drafters. Both were validated on the same 2x RTX 3090 (NV4) host; both preserve the model output exactly, because speculative decoding only proposes tokens the target model then verifies.

- **Speed tier — DFlash2 (default / recommended).** A block-diffusion "keep drafting parallel" drafter (`incoai/Qwen3.8-27B-DFlash2`, `n=7`), the sibling successor to DFlash, ported from [vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816). It took **#1 on our internal 69-scenario agent eval**. Best responsiveness; the trade is a heavier drafter, so a smaller KV pool and a lower safe context ceiling (see below). Recipe: [`recipes/speed-tier-dflash2/`](recipes/speed-tier-dflash2/).
- **Context tier — AutoRound + MTP (`n=4`).** The in-checkpoint MTP head. Lighter, so it keeps the full 262,144-token context and the larger KV pool. Reach for it when you need maximum context or maximum concurrent long-context streams. This is the profile the rest of this README documents.

### Head to head (measured, same box, same day)

| Metric | Speed tier · DFlash2 | Context tier · MTP n=4 |
|---|---:|---:|
| Drafter | DFlash2 `n=7` (block-diffusion, separate 3.6 GB checkpoint) | MTP `n=4` (in-checkpoint head) |
| vLLM image | `v0.27.1` + `vllm-dflash2` overlay | `qwen38-x86_64-cu129` |
| GPU memory utilization | 0.80 (heavy drafter) | 0.90 |
| **Max context (verified safe)** | **131,072** | **262,144** |
| **Logical KV pool** | **258,735 tokens** | **532,026 tokens** |
| Concurrency at max context | 1.97x @ 131K | 2.03x @ 262K |
| Single-stream decode | 97.6 tok/s | 99.9 tok/s |
| Aggregate decode (saturated, ~C2) | **~181 tok/s** | ~180 tok/s |
| Our 69-scenario eval | **★★★★★ #1 — 66/2/1, Quality 97.1, Deployability 97.9** | ★★★★★ baseline — 66/2/1, Quality 97.1, Deployability 97.5 |

**Read this:** raw decode is a wash (both ~180 tok/s aggregate, single-stream within a point). The DFlash2 win is **responsiveness** — median turn latency 528 ms drove it to a 99.9 responsiveness score and the top Deployability on our board (97.9 vs 97.5), at **identical model quality** (both 66 pass / 2 partial / 1 fail out of 69). It is the default because on real agent traffic it feels snappier and scores highest, for free.

### Why the KV pool and max context differ

The DFlash2 drafter is **heavy**: its two candidate-selector codebooks (~0.5 GiB per GPU) plus a five-layer backbone sit in VRAM alongside the 27B weights, and they eat the headroom the first prefill needs for activation buffers. To keep the first prefill from OOM-ing, the speed tier runs at `GPU_MEMORY_UTILIZATION=0.80` instead of 0.90. Lower utilization means a smaller KV cache — **258,735 logical tokens vs 532,026** — so we ship the speed tier at a **131,072** context ceiling rather than the model's native 262,144. The weights can address 262K, but the pool at 0.80 will not hold a full 256K sequence next to the drafter without risking a prefill OOM. If you need the full 262K context or the deepest concurrent-stream pool, use the context tier; if you want the top-ranked, snappiest agent server, use DFlash2. (Logical token counts are for the tensor-parallel engine — do not double them for two cards.)

## Verified reference result

The checked-in profile was validated on GPU 2/3 of a four-3090 host with an NV4 link. GPU 0/1 ran a separate model and were not touched.

| Setting | Verified value |
|---|---:|
| Model | Qwen3.8-27B AutoRound W4A16 |
| vLLM quantization | `inc` |
| Linear kernel | AutoGPTQ Marlin |
| Tensor parallelism | 2 |
| Context | 262,144 tokens |
| KV dtype | FP8 |
| KV allocation | 9.6 GiB per GPU |
| Logical KV pool | 532,026 tokens |
| 262K concurrency | 2.03x |
| GPU memory after startup | 22,154 MiB per GPU |
| Free GPU memory | 1,971 MiB per GPU |
| MTP draft length | **4 tokens** (was 3; see A/B benchmark below) |
| Restart/OOM count after verification | 0 / 0 |

Four forced 384-token single-stream checks measured 94.42, 97.23, 103.77, and 98.61 tokens/s. The three-run repeat average was **99.87 tokens/s**. Performance varies with prompt length, MTP acceptance, clocks, thermals, drivers, topology, and client overhead. These numbers measure speed, not model-quality equivalence to another quant.

## MTP3 vs MTP4: measured A/B on the same box, same day

We originally shipped this profile with a 3-token MTP draft. A community sweep
suggested `n=4` is the knee of the curve, so we ran both side by side on one
4x3090 host: **GPU 0/1 serving MTP n=3 and GPU 2/3 serving MTP n=4**, identical
model, image, and flags otherwise. Each number is the mean decode rate of
repeated single-stream runs (streamed, timed first token to last, prefill
excluded, thinking disabled, temperature 0.2).

**Easy prompts** (highly predictable output — this is where speculative
decoding shines, and where you see the top speeds):

| Prompt | MTP n=3 (tok/s) | MTP n=4 (tok/s) | Gain |
|---|---:|---:|---:|
| alphabet-rows | 145.9 | 154.4 | +6% |
| count-1-200 | 142.7 | 154.0 | +8% |
| json-fill | 141.6 | 159.5 | +13% |
| repeat-hello | 103.9 | 139.0 | +34% |

![Easy prompts](docs/bench-mtp4-easy.svg)

**Normal prompts** (real prose and code):

| Prompt | MTP n=3 (tok/s) | MTP n=4 (tok/s) | Gain |
|---|---:|---:|---:|
| code-api | 126.1 | 133.5 | +6% |
| code-quicksort | 126.3 | 126.7 | +0% |
| email-draft | 99.6 | 104.4 | +5% |
| prose-fridge | 92.8 | 88.3 | -5% |

![Normal prompts](docs/bench-mtp4-normal.svg)

**Summary:** easy/structured output mean **133.5 → 151.7 tok/s (+14%)** with a
best single run of **159.9 tok/s**; normal-work mean **111.2 → 113.2 (+2%)**
with code up ~6% and free-form prose down ~5%. MTP n=4 wins or ties everywhere
that matters for agent workloads (structured output, JSON, code), so **n=4 is
now the repository default** (`NUM_SPECULATIVE_TOKENS=4`). If your workload is
overwhelmingly free-form prose, `n=3` remains a fine choice — it is one number
in `.env`.

Reproduce with `scripts/mtp-ab-bench.py` against any two endpoints.

## What is quantized

This checkpoint uses AutoRound W4A16: the main quantized weights are 4-bit and activations remain 16-bit. Its metadata uses `auto_round:auto_gptq` packing, and the tested vLLM image resolves it through `--quantization inc` to the Marlin AutoGPTQ kernel.

Selected sensitive layers, the vision components, and MTP tensors remain at higher precision by checkpoint design. The server uses BF16 compute and FP8 KV cache.

## Requirements

- Linux x86-64
- Two RTX 3090 24 GB GPUs
- Recent NVIDIA driver
- Docker Engine and NVIDIA Container Toolkit
- Python 3 and `curl`
- Hugging Face `hf` CLI for downloading
- At least 35 GB free before downloading weights and building caches
- A Qwen3.8-capable vLLM image; the verified image tag is the default in `.env.example`

## Quick start

Create your local configuration:

```bash
cp .env.example .env
chmod +x scripts/*.sh
```

Edit `.env`, especially `GPU_IDS`, `MODEL_DIR`, and `TOPOLOGY`. On the exact reference host, start from:

```bash
cp deployments/reference-4x3090-gpu2-3.env .env
```

Download and deploy:

```bash
./scripts/download-model.sh
./scripts/preflight.sh
./scripts/run.sh
./scripts/wait-ready.sh
python3 scripts/verify.py --benchmark-tokens 384
python3 scripts/benchmark.py --runs 3 --tokens 384
```

The default API is:

```text
http://SERVER_IP:8011/v1
model: qwen3.8-27b
context: 262144
```

## Exact serving behavior

The launch script intentionally enables:

- `--quantization inc` and BF16 compute;
- compiled execution and CUDA graphs; it does not use eager mode;
- four-token MTP speculative decoding (see the MTP3 vs MTP4 benchmark below);
- FP8 KV cache;
- `qwen3_xml` automatic tool parsing;
- `qwen3` reasoning parsing;
- thinking disabled by default;
- a 1,048,576-pixel image preprocessing ceiling;
- persistent vLLM and Triton caches.

The model's generation configuration currently supplies temperature `1.0`, top-k `20`, and top-p `0.95` when a client omits sampling values. Clients can explicitly send their own sampling configuration.

## NVLink versus PCIe

Use `TOPOLOGY=auto`, `nvlink`, or `pcie`.

With NVLink, `nvidia-smi topo -m` must show an `NV#` connection between the selected pair. The launch adds `--disable-custom-all-reduce`, selecting NCCL/PyNCCL tensor-parallel collectives. Despite its name, this does not disable NVLink; NCCL uses the available NVLink transport.

For PCIe-only pairs, the flag is omitted so vLLM can choose its supported custom collective. PCIe performance depends on slot/root-complex topology and peer-access support, so benchmark that host rather than assuming the NVLink result.

## API example

```bash
curl http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-27b",
    "messages":[{"role":"user","content":"Reply with exactly OK"}],
    "temperature":0
  }'
```

## Operations

```bash
./scripts/status.sh
docker logs -f qwen38-autoround-int4
docker stop qwen38-autoround-int4
docker start qwen38-autoround-int4
```

To intentionally recreate the repository's own container after editing `.env`:

```bash
REPLACE_CONTAINER=1 ./scripts/run.sh
```

The script refuses replacement by default. For a cutover from another quant, follow [docs/CUTOVER.md](docs/CUTOVER.md) so the old container remains recoverable.

## Systemd

Create the container once with `scripts/run.sh`, then install the provided service. It assumes the repository is located at `/opt/qwen38-autoround-2x3090`:

```bash
sudo cp systemd/qwen38-autoround-int4.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qwen38-autoround-int4.service
sudo systemctl start qwen38-autoround-int4.service
```

## Documentation

- [**Speed tier — DFlash2 drafter (default, #1 on our eval)**](recipes/speed-tier-dflash2/)
- [Tuning](docs/TUNING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Safe cutover and rollback](docs/CUTOVER.md)

## Repository layout

```text
.
|-- .env.example
|-- README.md
|-- deployments/
|   `-- reference-4x3090-gpu2-3.env
|-- docs/
|   |-- CUTOVER.md
|   |-- TROUBLESHOOTING.md
|   `-- TUNING.md
|-- scripts/
|   |-- benchmark.py
|   |-- download-model.sh
|   |-- lib.sh
|   |-- preflight.sh
|   |-- run.sh
|   |-- status.sh
|   |-- verify.py
|   `-- wait-ready.sh
|-- recipes/
|   `-- speed-tier-dflash2/   # SPEED TIER (default): DFlash2 drafter + vendored vllm#52816 overlay
`-- systemd/
    `-- qwen38-autoround-int4.service
```

## License

MIT
