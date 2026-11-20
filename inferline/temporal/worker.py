"""Worker + demo client for the batch-inference workflow.

Run a local Temporal server first (from the repo root)::

    docker compose up -d temporal     # temporalio/auto-setup on :7233

then::

    python -m inferline.temporal.worker worker
    python -m inferline.temporal.worker demo --prompts 24

The demo submits a batch job of synthetic prompts, polls the
``progress`` query while shards run, and prints the aggregated result —
the same Scheduler/BatchingPolicy/RateLimiter stack the HTTP server
uses, driven durably by Temporal.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from temporalio.client import Client
from temporalio.worker import Worker

from inferline.temporal.activities import InferenceActivities
from inferline.temporal.models import (
    BatchInferenceInput,
    BatchPolicySpec,
    InferenceRequestSpec,
)
from inferline.temporal.workflows import BatchInferenceWorkflow

TASK_QUEUE = "inferline-batch-inference"


def build_worker(client: Client, task_queue: str = TASK_QUEUE) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[BatchInferenceWorkflow],
        activities=InferenceActivities().all(),
    )


async def _run_worker(address: str, namespace: str) -> None:
    client = await Client.connect(address, namespace=namespace)
    worker = build_worker(client)
    print(f"inferline worker listening on task queue '{TASK_QUEUE}'")
    await worker.run()


async def _run_demo(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address, namespace=args.namespace)
    worker = build_worker(client)
    async with worker:
        job_id = f"demo-{uuid.uuid4().hex[:8]}"
        requests = [
            InferenceRequestSpec(
                request_id=f"req-{i:03d}",
                prompt=f"Summarize batch inference pattern number {i} "
                       f"for a serving system design review.",
                max_new_tokens=args.max_new_tokens,
                priority="INTERACTIVE" if i % 5 == 0 else "BATCH",
            )
            for i in range(args.prompts)
        ]
        handle = await client.start_workflow(
            BatchInferenceWorkflow.run,
            BatchInferenceInput(
                job_id=job_id,
                requests=requests,
                policy=BatchPolicySpec(max_tokens_per_batch=1024),
                shard_max_requests=4,
                backend_latency_per_token_ms=args.latency_ms,
            ),
            id=job_id,
            task_queue=TASK_QUEUE,
        )
        print(f"started job {job_id} with {len(requests)} requests")
        while True:
            progress = await handle.query(BatchInferenceWorkflow.progress)
            print(f"  progress: {progress.completed_requests}/"
                  f"{progress.total_requests} requests, "
                  f"{progress.shards_completed}/{progress.shards_total} shards")
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(handle.result()), timeout=1.0
                )
                break
            except asyncio.TimeoutError:
                continue
        print(f"job {result.status}: {result.requests_completed} completed, "
              f"{result.tokens_total} tokens, "
              f"{result.batches_dispatched} batches, "
              f"mean latency {result.mean_latency_s:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(prog="inferline-temporal")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--namespace", default="default")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("worker", help="run the worker in the foreground")

    demo = sub.add_parser("demo", help="run a worker + one demo batch job")
    demo.add_argument("--prompts", type=int, default=12)
    demo.add_argument("--max-new-tokens", type=int, default=16)
    demo.add_argument("--latency-ms", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "worker":
        asyncio.run(_run_worker(args.address, args.namespace))
    else:
        asyncio.run(_run_demo(args))


if __name__ == "__main__":
    main()
