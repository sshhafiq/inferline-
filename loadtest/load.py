"""Asyncio load generator for inferline.

Hammers ``POST /v1/chat/completions`` with N concurrent clients and reports
throughput / latency percentiles. No extra dependencies beyond httpx.

Usage:
    python loadtest/load.py --url http://localhost:8000 --concurrency 16 \\
        --requests 200 --max-tokens 64
    python loadtest/load.py --stream   # measure time-to-first-token too
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    max_tokens: int,
    stream: bool,
    api_key: str | None,
) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": "inferline-mock-1b",
        "messages": [{"role": "user", "content": "Write a short poem about GPUs."}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    tokens = 0
    async with client.stream(
        "POST", f"{url}/v1/chat/completions", json=payload, headers=headers,
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                tokens += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
    ended = time.perf_counter()
    return {
        "latency": ended - started,
        "ttft": (first_token_at - started) if first_token_at else None,
        "tokens": tokens,
    }


async def worker(
    url: str, max_tokens: int, stream: bool, api_key: str | None,
    queue: asyncio.Queue, results: list,
):
    async with httpx.AsyncClient() as client:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(await one_request(client, url, max_tokens, stream, api_key))
            except Exception as exc:  # noqa: BLE001 - counted as failure
                results.append({"error": str(exc)})
            finally:
                queue.task_done()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test inferline.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(args.requests):
        queue.put_nowait(None)
    results: list = []
    started = time.perf_counter()
    await asyncio.gather(
        *[
            worker(args.url, args.max_tokens, True, args.api_key, queue, results)
            for _ in range(args.concurrency)
        ]
    )
    # NOTE: streaming is always used on the wire so TTFT is measurable;
    # --stream only controls whether TTFT is reported.
    elapsed = time.perf_counter() - started

    ok = [r for r in results if "error" not in r]
    failed = len(results) - len(ok)
    lat = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    total_tokens = sum(r["tokens"] for r in ok)

    print(f"requests:      {args.requests} ({failed} failed)")
    print(f"concurrency:   {args.concurrency}")
    print(f"elapsed:       {elapsed:.2f}s")
    print(f"throughput:    {len(ok) / elapsed:.1f} req/s")
    print(f"token rate:    {total_tokens / elapsed:.1f} tok/s")
    print(f"latency p50:   {percentile(lat, 50) * 1000:.0f} ms")
    print(f"latency p99:   {percentile(lat, 99) * 1000:.0f} ms")
    if args.stream:
        print(f"TTFT p50:      {percentile(ttfts, 50) * 1000:.0f} ms")
        print(f"TTFT p99:      {percentile(ttfts, 99) * 1000:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
