#!/usr/bin/env python3
"""club-3090 DFlash2 installer — full port of vllm-project/vllm#52816 onto the
stock vllm/vllm-openai image.

DFlash2 is the "Keep Drafting Parallel" drafter: a block-diffusion drafter with a
grouped dynamic depthwise conv + a candidate selector. The drafter checkpoint
(incoai_Qwen3.8-27B-DFlash2) declares architecture "DFlash2DraftModel".

Why a full port (not just the model file):
  The DFlash2 candidate selector is only exercised by the V2 speculator
  (DFlash2Speculator). On the V1 runner the same checkpoint silently degrades to
  the base DFlash drafter (DFlashProposer), which never builds the DFlash2 model —
  so weight-load fails with "no module named 'candidate_selector'". The upstream
  PR therefore ships 11 files; we port the 5 that matter here:

    1. vllm/model_executor/models/qwen3_dflash2.py      (new)  — the DFlash2 model
    2. vllm/model_executor/models/qwen3_dflash.py        (edit) — base refactor so
       the DFlash2 subclass can override decoder_layer_cls / model_cls
    3. vllm/model_executor/models/registry.py            (edit) — register
       "DFlash2DraftModel" -> qwen3_dflash2.DFlash2Qwen3ForCausalLM
    4. vllm/v1/worker/gpu/spec_decode/dflash2/           (new)  — DFlash2Speculator
    5. vllm/v1/worker/gpu/spec_decode/__init__.py        (edit) — init_speculator
       dispatches to DFlash2Speculator for DFlash2 drafts
    6. vllm/config/vllm.py                               (edit) — use_v2_model_runner
       forces the V2 runner for DFlash2 drafts (so the selector is actually used)

All edits are anchor-checked + idempotent; any anchor drift hard-fails boot so a
re-pinned image can't serve a half-wired drafter.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    print("[dflash2] REFUSE: cannot locate vllm package")
    sys.exit(1)
VLLM = os.path.dirname(spec.origin)
MODELS = os.path.join(VLLM, "model_executor", "models")
SPECDEC = os.path.join(VLLM, "v1", "worker", "gpu", "spec_decode")
CONFIG = os.path.join(VLLM, "config", "vllm.py")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def patch_once(name, path, old, new, count=1):
    """Idempotent, anchor-checked single replacement.

    A patch is already-applied when the *new* text is present in the file. This is
    the correct idempotency test for "insert a new block" patches, where the *old*
    anchor text legitimately remains in the file after the patch (e.g. the
    `DFlashDraftModel` registry line stays when we add `DFlash2DraftModel` after
    it). Re-running then no-ops instead of double-inserting.

    On a fresh (unpatched) file the old anchor must appear exactly `count` times or
    we refuse (version skew). After applying we verify the new text landed exactly
    once, so a corrupted double-insert is caught, not shipped.
    """
    s = read(path)
    if new in s:
        if s.count(new) != 1:
            print(f"[dflash2] REFUSE {name}: new text present {s.count(new)}x (corrupt double-insert?)")
            return False
        print(f"[dflash2] {name}: already applied (idempotent no-op)")
        return True
    n = s.count(old)
    if n != count:
        print(f"[dflash2] REFUSE {name}: anchor found {n}x, need {count} (version skew?)")
        return False
    write(path + ".bak", s)
    result = s.replace(old, new)
    if result.count(new) != 1:
        print(f"[dflash2] REFUSE {name}: post-apply verification failed (new text {result.count(new)}x)")
        return False
    write(path, result)
    print(f"[dflash2] {name}: applied")
    return True


def main() -> int:
    ok = True

    # ---- 1. Install the DFlash2 model file (always overwrite = source of truth) ----
    model_src = os.path.join(HERE, "qwen3_dflash2.py")
    model_content = read(model_src)
    if not model_content.strip():
        print("[dflash2] REFUSE: vendored qwen3_dflash2.py is empty")
        return 1
    write(os.path.join(MODELS, "qwen3_dflash2.py"), model_content)
    print(f"[dflash2] installed qwen3_dflash2.py ({len(model_content)} bytes)")

    # ---- 2. Base refactor in qwen3_dflash.py ----
    qd = os.path.join(MODELS, "qwen3_dflash.py")
    # 2a. add decoder_layer_cls to DFlashQwen3Model
    ok &= patch_once(
        "base.decoder_layer_cls", qd,
        "@support_torch_compile\nclass DFlashQwen3Model(nn.Module):\n    hf_to_vllm_mapper = WeightsMapper(",
        "@support_torch_compile\nclass DFlashQwen3Model(nn.Module):\n    decoder_layer_cls = DFlashQwen3DecoderLayer\n\n    hf_to_vllm_mapper = WeightsMapper(",
    )
    # 2b. use the class attr when building layers
    ok &= patch_once(
        "base.layers_use_cls", qd,
        "                DFlashQwen3DecoderLayer(\n                    current_vllm_config,",
        "                self.decoder_layer_cls(\n                    current_vllm_config,",
    )
    # 2c. add model_cls to DFlashQwen3ForCausalLM
    ok &= patch_once(
        "base.model_cls", qd,
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):",
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n    model_cls = DFlashQwen3Model\n\n    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):",
    )
    # 2d. use the class attr when building the model
    ok &= patch_once(
        "base.self_model_use_cls", qd,
        "        self.model = DFlashQwen3Model(\n            vllm_config=vllm_config,",
        "        self.model = self.model_cls(\n            vllm_config=vllm_config,",
    )

    # ---- 3. Register the architecture in registry.py ----
    reg = os.path.join(MODELS, "registry.py")
    ok &= patch_once(
        "registry.DFlash2", reg,
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
        '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
    )

    # ---- 4. Install the DFlash2 speculator package ----
    pkgdir = os.path.join(SPECDEC, "dflash2")
    os.makedirs(pkgdir, exist_ok=True)
    write(os.path.join(pkgdir, "__init__.py"), read(os.path.join(HERE, "pkg_init.py")))
    write(os.path.join(pkgdir, "speculator.py"), read(os.path.join(HERE, "speculator.py")))
    print("[dflash2] installed v1/worker/gpu/spec_decode/dflash2/ (speculator)")

    # ---- 5. init_speculator dispatch in spec_decode/__init__.py ----
    sd = os.path.join(SPECDEC, "__init__.py")
    ok &= patch_once(
        "spec_decode.dispatch", sd,
        '    if speculative_config.method == "dflash":\n        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n            DFlashSpeculator,\n        )\n',
        '    if speculative_config.method == "dflash":\n'
        '        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:\n'
        '            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n'
        '                DFlash2Speculator,\n'
        '            )\n'
        '\n'
        '            return DFlash2Speculator(vllm_config, device)\n'
        '        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n'
        '            DFlashSpeculator,\n'
        '        )\n',
    )

    # ---- 6. Force the V2 runner for DFlash2 drafts in config/vllm.py ----
    ok &= patch_once(
        "config.use_v2_dflash2", CONFIG,
        "        if self._dflash_needs_multi_kv_group():\n            return True\n",
        "        if self._dflash_needs_multi_kv_group():\n            return True\n\n"
        "        # The DFlash2 candidate selector exists only in the V2 speculator. On V1\n"
        "        # the same checkpoint drafts through DFlashProposer, which never calls\n"
        "        # it, so the draft degrades to DFlash1 silently. Force V2 as for dspark.\n"
        "        if self._is_dflash2_draft():\n"
        "            return True\n",
    )
    # add the helper method right before _dflash_needs_multi_kv_group
    ok &= patch_once(
        "config._is_dflash2_draft", CONFIG,
        "    def _dflash_needs_multi_kv_group(self) -> bool:\n",
        "    def _is_dflash2_draft(self) -> bool:\n"
        '        """Whether the DFlash draft is a DFlash2 one, by the architecture the\n'
        "        speculator selects on (v1/worker/gpu/spec_decode/__init__.py).\"\"\"\n"
        "        spec = self.speculative_config\n"
        "        if spec is None or spec.method != \"dflash\":\n"
        "            return False\n"
        "        draft_config = getattr(spec, \"draft_model_config\", None)\n"
        "        if draft_config is None:\n"
        "            return False\n"
        "        return \"DFlash2DraftModel\" in (draft_config.architectures or [])\n\n"
        "    def _dflash_needs_multi_kv_group(self) -> bool:\n",
    )

    # ---- 7. Clear the model-info cache (fresh introspection for the new module) ----
    cache_dir = os.path.join(os.environ.get("VLLM_CACHE_ROOT", "/root/.cache/vllm"), "modelinfos")
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            p = os.path.join(cache_dir, name)
            if os.path.isfile(p):
                os.remove(p)
        print(f"[dflash2] cleared model-info cache ({cache_dir})")
    else:
        print(f"[dflash2] no model-info cache dir at {cache_dir} (cold start, ok)")

    # ---- 8. Verify: model imports + speculator imports + registry resolves ----
    try:
        import importlib
        mod = importlib.import_module("vllm.model_executor.models.qwen3_dflash2")
        cls = getattr(mod, "DFlash2Qwen3ForCausalLM")
        print(f"[dflash2] import OK: {cls.__module__}.{cls.__name__}")
        spec_mod = importlib.import_module("vllm.v1.worker.gpu.spec_decode.dflash2.speculator")
        print(f"[dflash2] import OK: {spec_mod.DFlash2Speculator.__name__}")
        from vllm.model_executor.models import ModelRegistry
        entry = ModelRegistry.models.get("DFlash2DraftModel")
        if entry is None:
            print("[dflash2] REFUSE: DFlash2DraftModel not in registry after patch")
            return 1
        print(f"[dflash2] registry OK: {entry.module_name}.{entry.class_name}")
    except Exception as e:
        import traceback
        print(f"[dflash2] REFUSE: post-patch import check failed: {e!r}")
        traceback.print_exc()
        return 1

    if not ok:
        print("[dflash2] REFUSE: one or more patches failed to apply (anchor drift?)")
        return 1
    print("[dflash2] all patches applied + verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
