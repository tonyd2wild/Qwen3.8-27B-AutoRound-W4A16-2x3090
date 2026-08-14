#!/usr/bin/env python3
"""Verify model identity, chat, reasoning separation, tools, vision, and speed."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import time
import urllib.error
import urllib.request
import zlib


def get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, payload: dict, timeout: int = 180) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), time.perf_counter() - started


def solid_png(width: int = 64, height: int = 64) -> bytes:
    row = b"\x00" + bytes((240, 220, 40)) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = binascii.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--benchmark-tokens", type=int, default=384)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    chat_url = base_url + "/chat/completions"

    models = get(base_url + "/models")
    entry = next((m for m in models.get("data", []) if m.get("id") == args.model), None)
    assert entry, models
    print(f"PASS model id: {args.model}, context={entry.get('max_model_len')}")

    chat, _ = post(
        chat_url,
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly HARNESS_OK and nothing else."}],
            "temperature": 0,
            "max_tokens": 32,
        },
    )
    message = chat["choices"][0]["message"]
    assert message.get("content") == "HARNESS_OK", message
    assert not message.get("reasoning_content"), message
    print("PASS chat and reasoning separation")

    tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
    result, _ = post(
        chat_url,
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Use the calculator tool to add 17 and 25."}],
            "tools": [tool],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 128,
        },
    )
    message = result["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "calculator", message
    json.loads(calls[0]["function"]["arguments"])
    assert not message.get("reasoning_content"), message
    print("PASS structured tool calling")

    image_url = "data:image/png;base64," + base64.b64encode(solid_png()).decode()
    result, _ = post(
        chat_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "What is the dominant color? Answer briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
        },
    )
    content = result["choices"][0]["message"].get("content", "").lower()
    assert "yellow" in content, content
    print("PASS vision")

    result, elapsed = post(
        chat_url,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain speculative decoding, acceptance rates, and latency tradeoffs in technical detail.",
                }
            ],
            "temperature": 0.7,
            "max_tokens": args.benchmark_tokens,
            "min_tokens": args.benchmark_tokens,
        },
    )
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    assert tokens == args.benchmark_tokens, result.get("usage")
    rate = tokens / elapsed if elapsed else 0
    print(f"PASS benchmark: {tokens} tokens in {elapsed:.2f}s = {rate:.2f} tok/s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
