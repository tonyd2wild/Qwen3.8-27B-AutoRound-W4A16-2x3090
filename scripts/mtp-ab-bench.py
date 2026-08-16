#!/usr/bin/env python3
"""MTP3 vs MTP4 A/B benchmark — Qwen3.8-27B AutoRound W4A16 on 2x3090 pairs.
Point ENDPOINT_A / ENDPOINT_B (and LABEL_A / LABEL_B, MODEL) env vars at any two
OpenAI-compatible endpoints, e.g. the same model served with different MTP drafts.
Streams each completion and measures DECODE tok/s (first token -> last token),
so TTFT/prefill is excluded. Thinking off per fleet rule."""
import json, time, urllib.request, statistics, sys

import os
ENDPOINTS = {
    os.environ.get("LABEL_A", "endpoint-A"): os.environ.get("ENDPOINT_A", "http://127.0.0.1:8010/v1"),
    os.environ.get("LABEL_B", "endpoint-B"): os.environ.get("ENDPOINT_B", "http://127.0.0.1:8011/v1"),
}
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
RUNS = 3

EASY = [  # highly predictable output -> max MTP acceptance -> top speeds
    ("count-1-200", "Count from 1 to 200, one number per line. Output only the numbers.", 600),
    ("repeat-hello", "Write the word hello exactly 250 times, separated by single spaces. Output only that.", 400),
    ("alphabet-rows", "Print the lowercase alphabet a-z on one line, repeated on 40 lines. Output only the lines.", 600),
    ("json-fill", "Output a JSON array of 60 objects, each exactly {\"id\": N, \"ok\": true} with N from 1 to 60. Output only JSON.", 700),
]
NORMAL = [  # realistic prose + code
    ("prose-fridge", "Explain how a refrigerator works in three clear paragraphs for a curious teenager.", 450),
    ("code-quicksort", "Write a Python quicksort implementation with type hints and a docstring, then show one usage example.", 450),
    ("email-draft", "Write a professional email to a supplier asking about lead times, minimum order quantities, and bulk pricing for sneaker storage crates.", 400),
    ("code-api", "Write a small Flask API with two endpoints: GET /health and POST /items that appends JSON items to an in-memory list. Include error handling.", 600),
]

def bench_one(base, prompt, max_tok):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok, "temperature": 0.2, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_first = None; n_chunks = 0; t_last = None
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            chs = d.get("choices") or []
            delta = (chs[0].get("delta", {}) if chs else {})
            if delta.get("content"):
                now = time.time()
                if t_first is None: t_first = now
                t_last = now; n_chunks += 1
            u = d.get("usage")
            if u and u.get("completion_tokens"): n_tok = u["completion_tokens"]
    # chunks != tokens with spec decode (a chunk can carry several tokens).
    # Re-request usage via non-stream? Instead: count tokens = completion_tokens from usage
    # when present; fall back to chunks (underestimate, noted).
    try:
        toks = n_tok
    except NameError:
        toks = n_chunks
    dur = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    if not dur or toks < 20: return None
    return toks / dur

def main():
    results = {}
    for label, base in ENDPOINTS.items():
        # warmup
        try: bench_one(base, "Say hello.", 20)
        except Exception as e: print(f"{label}: warmup failed {e}"); continue
        results[label] = {}
        for cat, prompts in (("easy", EASY), ("normal", NORMAL)):
            for key, prompt, mt in prompts:
                speeds = []
                for i in range(RUNS):
                    try:
                        s = bench_one(base, prompt, mt)
                        if s: speeds.append(s)
                    except Exception as e:
                        print(f"  {label} {key} run{i}: {type(e).__name__}", flush=True)
                if speeds:
                    results[label][f"{cat}:{key}"] = {
                        "runs": [round(x,1) for x in speeds],
                        "mean": round(statistics.mean(speeds),1),
                        "best": round(max(speeds),1)}
                    print(f"{label} {cat}:{key} -> mean {round(statistics.mean(speeds),1)} best {round(max(speeds),1)}", flush=True)
    out = "mtp_bench_results.json"
    json.dump(results, open(out, "w"), indent=1)
    print("WROTE", out)

if __name__ == "__main__":
    main()
