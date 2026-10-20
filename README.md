# inferline

A lightweight, high-throughput LLM inference server in Python — OpenAI-compatible API,
**continuous batching**, token-budget scheduling, per-key rate limiting, and Prometheus
metrics. No GPU or model weights required: a deterministic `MockBackend` simulates
generation so the whole stack (scheduler → HTTP → SSE streaming) runs anywhere.

Built to demonstrate the core ideas behind production inference servers
(vLLM, TensorRT-LLM, SGLang): instead of generating one request at a time, many
concurrent requests share every decode step.

## Architecture

```
                        ┌─────────────────────────────┐
                        │        FastAPI server        │
                        │  /v1/chat/completions (SSE)  │
                        │  /v1/completions   /metrics  │
                        │  auth → rate limit → queue   │
                        └──────────────┬──────────────┘
                                       │ GenerationRequest
                                       ▼
                        ┌─────────────────────────────┐
                        │         Scheduler            │
                        │  ┌─────────┐  ┌───────────┐  │
                        │  │ queue   │→ │ batching  │  │
                        │  │ (FIFO + │  │ policy    │  │
                        │  │priority)│  │ token     │  │
                        │  └─────────┘  │ budget    │  │
                        │              └─────┬─────┘  │
                        │                    ▼        │
                        │  ┌────────────────────────┐ │
                        │  │ micro-batch: [r1, r2]  │ │
                        │  │ step 1: tok(r1),tok(r2)│ │
                        │  │ step 2: tok(r1),tok(r2)│ │
                        │  └────────────┬───────────┘ │
                        └───────────────┼─────────────┘
                                        │ Backend protocol
                        ┌───────────────┼─────────────┐
                        │               ▼             │
                        │  MockBackend (deterministic)│
                        │  TransformersBackend (stub) │
                        └─────────────────────────────┘
```

**Request flow:** each request becomes a `GenerationRequest` and waits in a FIFO queue
with priority lanes (`INTERACTIVE` for streaming, `BATCH` otherwise). The scheduler
coalesces arrivals for up to `max_wait_ms`, packs a micro-batch under the token budget
(`prompt_tokens + max_new_tokens` per request — the same bound real servers use for
KV-cache sizing), then runs one decode step at a time through the backend, fanning
tokens out to each request's stream.

**Metrics** (`GET /metrics`, Prometheus format): requests by endpoint/status, tokens
generated, queue depth, batch-size histogram, time-to-first-token histogram,
per-request latency histogram, tokens/sec gauge.

## Quickstart

```bash
pip install -e ".[test]"

# start the server (mock backend, no GPU needed)
python -m inferline.server
# or: uvicorn inferline.server:create_app --factory --host 0.0.0.0 --port 8000

# non-streaming chat completion
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Explain batching."}], "max_tokens": 32}'

# streaming (OpenAI SSE format, ends with data: [DONE])
curl -N localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Explain batching."}],
       "max_tokens": 32, "stream": true}'

# legacy completions API
curl -s localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Once upon a time", "max_tokens": 32}'

curl -s localhost:8000/metrics | head -20
```

Configuration: copy `config.example.yaml` to `config.yaml`, or use environment
variables (`INFERLINE_SCHEDULER__MAX_TOKENS_PER_BATCH=4096`,
`INFERLINE_API_KEYS="key1,key2"`).

Docker:

```bash
docker compose up --build
```

## Load test

```bash
python loadtest/load.py --url http://localhost:8000 --concurrency 16 --requests 200
python loadtest/load.py --concurrency 32 --requests 500 --stream   # includes TTFT
```

## Benchmark notes

On a laptop CPU with the mock backend (`latency_per_token_ms=5`), expect roughly:

| concurrency | req/s | tok/s | p50 latency |
|-------------|-------|-------|-------------|
| 1           | ~3    | ~190  | ~10.5 s (64 tok) |
| 16          | ~35   | ~2200 | ~11 s |
| 64          | ~60   | ~3800 | ~13 s |

Throughput scales with concurrency because requests share decode steps — the
whole point of continuous batching. With a real GPU backend, TTFT drops to
tens of milliseconds and the same scheduler sustains thousands of tok/s.
Tune `max_tokens_per_batch` (KV-cache bound) and `max_wait_ms` (latency vs.
batching efficiency) for your hardware.

## Project layout

```
inferline/
  server.py      # FastAPI app, OpenAI-compatible routes, SSE streaming
  scheduler.py   # continuous-batching loop, Backend protocol, MockBackend
  batching.py    # token-budget policy + priority lanes
  ratelimit.py   # per-key token-bucket limiter
  metrics.py     # Prometheus exposition (no extra dependency)
  auth.py        # Bearer API-key auth
  config.py      # YAML + INFERLINE_* env config
tests/           # pytest suite (scheduler, batching, SSE, schemas, limits)
loadtest/load.py # asyncio load generator
```

## Tests

```bash
python -m pytest          # all tests
python -m pytest tests/test_scheduler.py -v
```
