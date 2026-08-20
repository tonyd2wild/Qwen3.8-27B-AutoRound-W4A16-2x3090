#!/usr/bin/env bash
# DFlash2 drafter installer — runs in the container entrypoint before `vllm serve`.
#
# Installs the DFlash2 speculative-decoding drafter (the "Keep Drafting Parallel"
# successor to DFlash) into the stock vLLM image. This is a full port of
# vllm-project/vllm#52816 (the 11-file upstream PR), reduced to the pieces that
# make the drafter actually work:
#   1. vllm/model_executor/models/qwen3_dflash2.py   (new) — the DFlash2 model
#      (grouped dynamic depthwise conv + candidate selector + unquantized-LM-head
#      guard fix from vllm#52883).
#   2. vllm/model_executor/models/qwen3_dflash.py    (edit) — base refactor so the
#      DFlash2 subclass can override decoder_layer_cls / model_cls.
#   3. vllm/model_executor/models/registry.py         (edit) — register
#      "DFlash2DraftModel" -> qwen3_dflash2.DFlash2Qwen3ForCausalLM.
#   4. vllm/v1/worker/gpu/spec_decode/dflash2/       (new) — the DFlash2Speculator.
#   5. vllm/v1/worker/gpu/spec_decode/__init__.py    (edit) — init_speculator
#      dispatches to DFlash2Speculator for DFlash2 drafts.
#   6. vllm/config/vllm.py                           (edit) — use_v2_model_runner
#      forces the V2 runner for DFlash2 drafts (the candidate selector only runs
#      in the V2 speculator; on V1 the checkpoint silently degrades to base DFlash
#      and weight-load fails with "no module named 'candidate_selector'").
#
# WHY the speculator + dispatch + V2-runner force are non-optional: the DFlash2
# candidate selector is only exercised by the V2 speculator. The oceanplexian/vllm
# fork's PR #1 shipped ONLY the model file, so on the V1 runner the same
# checkpoint degrades to the base DFlash drafter and fails to load. All six
# pieces are required for a working drafter.
#
# Idempotent + anchor-checked. Hard-fails boot (exit 1) on anchor drift so a
# re-pinned image can't serve a half-wired drafter. No-op if already applied.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/patch_dflash2.py" || {
  echo "[dflash2] FATAL: DFlash2 patch failed to apply (anchor drift / custom image?)" >&2
  echo "[dflash2] Refusing to boot a stack that advertises a drafter it cannot load." >&2
  exit 1
}
