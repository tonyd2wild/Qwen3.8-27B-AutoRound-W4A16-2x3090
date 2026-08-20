"""club-3090 activation-headroom probe.

Measures the REAL peak transient activation memory of the running model, so we
can size `--gpu-memory-utilization` to the actual headroom the workload needs
instead of vLLM's boot-time estimate.

How it works:
  We wrap the model's forward() so that after every forward pass we read
  torch.cuda.max_memory_allocated() (the high-water mark since process start)
  and torch.cuda.memory_allocated() (what's live right now). The difference
  between the two, tracked across steps, is the transient activation working set
  that the static budget does NOT reserve. We log:

    [memprobe] step=<n> peak=<X>GiB live=<Y>GiB transient_peak=<X-Y_at_peak>GiB

  We reset the high-water mark every step (torch.cuda.reset_peak_memory_stats)
  so each logged line is the peak for THAT step only — that's the number that
  matters for headroom (the worst single-step transient), not a cumulative max.

  It runs in the worker process (the one that actually holds the GPU tensors),
  so it sees real per-GPU numbers. One line per N steps to keep the log quiet.

Enable via the entrypoint:  MEMPROBE=1  (default off = zero overhead).
Tune log cadence with MEMPROBE_EVERY (default 20 steps).
"""
import os

import torch

_enabled = os.environ.get("MEMPROBE", "0") == "1"
_every = int(os.environ.get("MEMPROBE_EVERY", "20"))
_step = 0
_orig_forward = None


def _gib(nbytes: float) -> float:
    return nbytes / (1024 ** 3)


def _probe_once(model, *args, **kwargs):
    """Run the real forward, then report this step's transient activation peak."""
    global _step
    # Reset the high-water mark so we measure THIS step's transient peak only.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = model._memprobe_real_forward(*args, **kwargs)
    _step += 1
    if _step % _every == 0 and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()      # high-water since reset
        live = torch.cuda.memory_allocated()           # live right now
        # transient peak = peak minus what was live at step start. We don't have
        # the exact start value, but live-at-end is a close lower bound; the
        # meaningful headroom number is `peak` (the max the allocator had to reach).
        print(
            f"[memprobe] step={_step} peak={_gib(peak):.3f}GiB "
            f"live={_gib(live):.3f}GiB "
            f"transient≈{_gib(peak - live):.3f}GiB "
            f"(device={torch.cuda.current_device()})",
            flush=True,
        )
    return out


def install(model):
    """Wrap `model.forward` with the probe. Idempotent."""
    global _orig_forward
    if not _enabled:
        return
    if getattr(model, "_memprobe_installed", False):
        return
    model._memprobe_real_forward = model.forward
    model.forward = lambda *a, **k: _probe_once(model, *a, **k)
    model._memprobe_installed = True
    print(f"[memprobe] installed (every {_every} steps)", flush=True)
