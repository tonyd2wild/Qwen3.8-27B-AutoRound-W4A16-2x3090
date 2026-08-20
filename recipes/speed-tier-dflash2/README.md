# Speed tier — DFlash2 drafter (default / recommended)

Qwen3.8-27B AutoRound W4A16 served with the **DFlash2** block-diffusion drafter instead of the
in-checkpoint MTP head. Same weights, same output, ranked **#1 on our internal 69-scenario agent
eval** (66 pass / 2 partial / 1 fail, Quality 97.1, Deployability 97.9, ★★★★★).

Speculative decoding never changes what the model says — the target model verifies every token the
drafter proposes — so this is a pure serving-speed / responsiveness change, not a quality change.

## What DFlash2 is

`incoai/Qwen3.8-27B-DFlash2` (3.6 GB) is the block-diffusion "keep drafting parallel" successor to
DFlash: a five-layer backbone plus a grouped depthwise conv and a **candidate selector** (two
codebook matrices). It proposes `n=7` tokens per step.

It is **not** a drop-in checkpoint. The candidate selector is only exercised by the V2 speculator
(`DFlash2Speculator`); on the V1 runner the same checkpoint silently degrades to base DFlash and
weight-load fails with `no module named candidate_selector`. The working deployment therefore needs
the full six-piece port of **[vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816)**,
vendored here under `patches/vllm-dflash2/` (model file + base refactor + registry entry + the
DFlash2Speculator + the `init_speculator` dispatch + the `use_v2_model_runner` force). The overlay's
`install.sh` **hard-fails boot on anchor drift**, so a re-pinned image can never serve a half-wired
drafter. The `patches/` overlays are pinned to the `vllm/vllm-openai:v0.27.1` image.

Credit: the DFlash2 port and the `w4a8` / `pr48375` overlays originate from
[noonghunna/club-3090 #1060](https://github.com/noonghunna/club-3090), which ports upstream
vllm-project/vllm#52816.

## Run it

```bash
# 1. weights: the same AutoRound target as the context tier, plus the drafter
hf download Vishva007/Qwen3.8-27B-W4A16-AutoRound --local-dir /home/user/models/qwen3.8-27b-w4a16-autoround
hf download incoai/Qwen3.8-27B-DFlash2          --local-dir /home/user/models/qwen3.8-27b-dflash2
# (the target and drafter must live under the same parent directory)

# 2. edit paths/GPUs in the env, then launch
$EDITOR recipes/speed-tier-dflash2/speed-tier-dflash2-2x3090.env
bash recipes/speed-tier-dflash2/run-dflash2.sh

# 3. wait + check
docker logs -f qwen38-dflash2-speed          # first boot compiles CUDA graphs; a few minutes
curl -s http://127.0.0.1:8011/v1/models
```

The overlay prints `[dflash2] ...`, `[pr48375] applied`, and `[w4a8] ... W4A16` during boot; the
engine log shows `method='dflash'` in the speculative config once it is wired.

## Verified result (2x RTX 3090, NV4 link, GPU 2/3)

| | Value |
|---|---:|
| Max context (verified safe) | 131,072 |
| GPU memory utilization | 0.80 |
| Logical KV pool | 258,735 tokens |
| Concurrency @ 131K | 1.97x |
| Single-stream decode | 97.6 tok/s |
| Aggregate decode (C2+, saturated) | ~181 tok/s |
| 69-scenario eval | ★★★★★ #1 — 66/2/1, Quality 97.1, Responsiveness 99.9, Deployability 97.9 |

Concurrency sweep (aggregate / per-stream tok/s), `temperature 0`, 400-token completions:

| Concurrency | Aggregate tok/s | Per-stream tok/s |
|---:|---:|---:|
| 1 | 97.6 | 97.6 |
| 2 | 181.1 | 90.5 |
| 4 | 180.5 | 45.1 |
| 6 | 179.9 | 30.0 |
| 8 | 177.9 | 22.2 |

Aggregate throughput saturates the two cards at ~180 tok/s by C2 (memory-bandwidth bound), the same
ceiling as the MTP tier. DFlash2's advantage is responsiveness (median turn latency 528 ms), which is
what carried it to the top Deployability score.

## Two gotchas (both handled by the recipe)

- **Custom all-reduce crashes these 3090s** even with NVLink present: `custom_all_reduce.cuh:164
  'invalid argument'`, worker dies, engine restarts. `run-dflash2.sh` always passes
  `--disable-custom-all-reduce`; NCCL still uses the NVLink transport.
- **The drafter is heavy.** Its codebooks (~0.5 GiB/GPU) + backbone eat prefill-activation headroom.
  That is why utilization is 0.80 and context is capped at 131K (see the root README's KV-pool
  section). Do not raise `MAX_MODEL_LEN` toward 256K without re-measuring the prefill peak first, or
  the first large prefill can OOM the engine.

## When to use the context tier instead

Use [the root recipe](../../README.md) (AutoRound + MTP `n=4`) when you need the full **262,144**
context, the larger **532,026-token** KV pool, or the deepest pool of concurrent long-context
streams. Model quality is identical (both score 66/2/1); the trade is context ceiling vs. the top
responsiveness score.
