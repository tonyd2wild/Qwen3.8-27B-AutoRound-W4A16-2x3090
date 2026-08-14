# Tuning

## Verified baseline

Start with the shipped profile before changing one variable at a time:

```text
MAX_MODEL_LEN=220000
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192
NUM_SPECULATIVE_TOKENS=3
MAX_IMAGE_PIXELS=1048576
```

The reference NVLink deployment reported:

```text
Model memory:                 9.59 GiB per GPU
Available KV cache memory:   9.6 GiB per GPU
GPU KV cache size:           532,026 logical tokens
220K maximum concurrency:    2.42x
CUDA graph memory:           0.59 GiB per GPU
```

The logical token count is for the tensor-parallel engine. Do not double it because there are two cards.

## AutoRound and Marlin

The checkpoint is W4A16 AutoRound with `auto_round:auto_gptq` packing. The launch command deliberately specifies:

```text
--quantization inc
--dtype bfloat16
```

Successful logs must include `MarlinLinearKernel for AutoGPTQLinearMethod`. If they do not, stop and investigate rather than assuming the intended fast kernel is active.

The checkpoint retains selected sensitive layers, the vision tower, and the MTP head at higher precision. This is expected and is not evidence that the main language model failed to quantize.

## Compilation and CUDA graphs

Do not add `--enforce-eager` when optimizing decode speed. Normal `torch.compile`, Triton compilation, and CUDA graph capture are enabled. The first launch can take several minutes.

These settings disable only problematic serialized standalone artifact reuse:

```text
VLLM_USE_STANDALONE_COMPILE=0
VLLM_USE_MEGA_AOT_ARTIFACT=0
```

With FlashInfer plus speculative decoding, this build may select piecewise rather than full CUDA graphs. That is normal on the tested image.

## MTP speculative decoding

Three speculative tokens performed well on the reference workload. Acceptance varies with prompt and output distribution, so compare 1, 2, and 3 using `scripts/benchmark.py`. Higher is not automatically faster.

## NVLink and PCIe

`TOPOLOGY=auto` detects an `NV#` edge in `nvidia-smi topo -m`. The NVLink profile adds `--disable-custom-all-reduce`, selecting NCCL/PyNCCL, which uses NVLink. It does not disable NVLink.

The PCIe profile omits that flag so vLLM can select its custom collective where supported. PCIe-only results depend heavily on motherboard topology, CPU sockets, ACS/IOMMU, and peer access. Benchmark the actual machine.

Do not set `NCCL_P2P_DISABLE=1` in normal operation.

## Benchmarking

Warm the server first, then use forced-length output so early EOS cannot inflate or distort results:

```bash
python3 scripts/benchmark.py --runs 3 --tokens 384
```

This measures end-to-end single-stream completion rate. It is not aggregate throughput under concurrency.
