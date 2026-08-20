"""club-3090 memprobe sitecustomize — auto-loads on every Python startup in the
container (sitecustomize is imported by site.py at interpreter init).

When MEMPROBE=1, it wraps vllm's model-loader `get_model` so that, right after
the (target) model is built, we install a forward() hook that logs the per-step
peak transient activation memory. That's the real headroom number we need to size
--gpu-memory-utilization.

CRITICAL: this file is imported by the `site` module during interpreter startup,
in a context where raising SystemExit is FATAL (it aborts the whole process with
"init_import_site: Failed to import the site module"). So the off-path must simply
RETURN — never raise, never sys.exit. No-op unless MEMPROBE=1.
"""
import os

if os.environ.get("MEMPROBE", "0") == "1":
    def _install():
        import importlib
        try:
            import vllm.model_executor.model_loader as ML
        except Exception:
            return  # vllm not importable in this process (e.g. the API server); skip

        try:
            memprobe = importlib.import_module("memprobe")  # on PYTHONPATH via /etc/club3090/dflash2
        except Exception as e:
            print(f"[memprobe] import failed (non-fatal): {e!r}", flush=True)
            return

        orig_get_model = getattr(ML, "get_model", None)
        if orig_get_model is None or getattr(ML, "_memprobe_patched", False):
            return

        def get_model(*args, **kwargs):
            model = orig_get_model(*args, **kwargs)
            try:
                memprobe.install(model)
            except Exception as e:
                print(f"[memprobe] install failed (non-fatal): {e!r}", flush=True)
            return model

        ML.get_model = get_model
        ML._memprobe_patched = True
        print("[memprobe] sitecustomize: hooked vllm model_loader.get_model", flush=True)

    try:
        _install()
    except Exception as e:
        # Never let a probe break model load — but never raise out of sitecustomize.
        print(f"[memprobe] sitecustomize hook error (non-fatal): {e!r}", flush=True)

# else: MEMPROBE != 1 -> do nothing. Returning from sitecustomize is the only safe
# "off" — a raise/SystemExit here would abort the interpreter at startup.
