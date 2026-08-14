#!/usr/bin/env python3
"""Run repeatable forced-length single-stream decode benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def run(url: str, model: str, tokens: int) -> tuple[float, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a detailed technical explanation of tensor parallel inference."}],
        "temperature": 0.7,
        "max_tokens": tokens,
        "min_tokens": tokens,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    actual = int(result.get("usage", {}).get("completion_tokens", 0))
    return actual / elapsed, actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    rates = []
    for index in range(1, args.runs + 1):
        rate, actual = run(url, args.model, args.tokens)
        if actual != args.tokens:
            raise SystemExit(f"Run {index} returned {actual} tokens, expected {args.tokens}")
        rates.append(rate)
        print(f"run={index} tokens={actual} rate={rate:.2f} tok/s")
    print(f"mean={statistics.mean(rates):.2f} median={statistics.median(rates):.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
