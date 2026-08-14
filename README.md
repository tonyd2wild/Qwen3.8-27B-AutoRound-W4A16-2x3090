# Qwen3.8-27B AutoRound W4A16 on 2x RTX 3090

Production-oriented Docker deployment for [`Vishva007/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/Vishva007/Qwen3.8-27B-W4A16-AutoRound) on two 24 GB RTX 3090 GPUs.

The repository supports paired 3090s with NVLink and ordinary PCIe-only pairs. It exposes an OpenAI-compatible vLLM API with compiled CUDA graphs, MTP speculative decoding, vision, structured tool calling, and separated reasoning.

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
| MTP draft length | 3 tokens |
| Restart/OOM count after verification | 0 / 0 |

Four forced 384-token single-stream checks measured 94.42, 97.23, 103.77, and 98.61 tokens/s. The three-run repeat average was **99.87 tokens/s**. Performance varies with prompt length, MTP acceptance, clocks, thermals, drivers, topology, and client overhead. These numbers measure speed, not model-quality equivalence to another quant.

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
- three-token MTP speculative decoding;
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
`-- systemd/
    `-- qwen38-autoround-int4.service
```

## License

MIT
