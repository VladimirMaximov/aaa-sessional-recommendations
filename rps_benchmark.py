"""
RPS benchmark for SessionRec API.

Tests /api/v1/feed with unique session_id per virtual user to avoid
session-state interference. Reports RPS, latency percentiles, and
compares against the 1 RPS/core threshold.
"""

import asyncio
import time
import statistics
import os
import sys

import aiohttp

BASE_URL = os.getenv("API_URL", "http://5.129.201.113:8000")
ENDPOINT = "/api/v1/feed"
LIMIT = 5

SCENARIOS = [
    {"concurrency": 1,  "duration": 15},
    {"concurrency": 5,  "duration": 15},
    {"concurrency": 10, "duration": 15},
    {"concurrency": 20, "duration": 15},
]


async def worker(
    session: aiohttp.ClientSession,
    worker_id: int,
    duration: float,
    results: list,
    errors: list,
):
    session_id = f"bench_w{worker_id:04d}"
    user_id = worker_id + 1
    url = f"{BASE_URL}{ENDPOINT}"
    params = {"session_id": session_id, "user_id": user_id, "limit": LIMIT}

    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                await resp.read()
                elapsed = (time.monotonic() - t0) * 1000
                if resp.status == 200:
                    results.append(elapsed)
                else:
                    errors.append(resp.status)
        except Exception as e:
            errors.append(str(e))


async def run_scenario(concurrency: int, duration: float) -> dict:
    results: list[float] = []
    errors: list = []

    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        # warm-up: 1 request
        async with session.get(
            f"{BASE_URL}{ENDPOINT}",
            params={"session_id": "warmup", "user_id": 0, "limit": 1},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            await r.read()

        workers = [
            worker(session, i, duration, results, errors)
            for i in range(concurrency)
        ]
        await asyncio.gather(*workers)

    total = len(results) + len(errors)
    rps = len(results) / duration
    p50 = statistics.median(results) if results else 0
    p95 = sorted(results)[int(len(results) * 0.95)] if results else 0
    p99 = sorted(results)[int(len(results) * 0.99)] if results else 0
    mean = statistics.mean(results) if results else 0

    return {
        "concurrency": concurrency,
        "duration_s": duration,
        "total_requests": total,
        "successful": len(results),
        "errors": len(errors),
        "rps": rps,
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def get_cpu_count() -> int:
    try:
        import subprocess
        out = subprocess.check_output(["nproc"], text=True).strip()
        return int(out)
    except Exception:
        return os.cpu_count() or 1


async def main():
    print(f"Target: {BASE_URL}{ENDPOINT}")
    print(f"Endpoint: GET /api/v1/feed?limit={LIMIT}")
    print("=" * 70)

    all_results = []
    for sc in SCENARIOS:
        print(f"\nRunning: concurrency={sc['concurrency']}, duration={sc['duration']}s ...")
        r = await run_scenario(sc["concurrency"], sc["duration"])
        all_results.append(r)
        print(
            f"  RPS={r['rps']:.1f}  mean={r['mean_ms']:.0f}ms  "
            f"p50={r['p50_ms']:.0f}ms  p95={r['p95_ms']:.0f}ms  p99={r['p99_ms']:.0f}ms  "
            f"ok={r['successful']}  err={r['errors']}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Concurrency':>12}  {'RPS':>8}  {'mean ms':>8}  {'p50 ms':>7}  {'p95 ms':>7}  {'p99 ms':>7}  {'Errors':>6}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['concurrency']:>12}  {r['rps']:>8.1f}  {r['mean_ms']:>8.0f}  "
            f"{r['p50_ms']:>7.0f}  {r['p95_ms']:>7.0f}  {r['p99_ms']:>7.0f}  {r['errors']:>6}"
        )

    best_rps = max(r["rps"] for r in all_results)
    print(f"\nPeak RPS: {best_rps:.1f}")

    # Check against 1 RPS/core threshold on the server
    # We don't know server's core count, but the requirement is >= 1 RPS/core.
    # With single-core baseline (concurrency=1) we measure per-core capacity.
    c1 = next(r for r in all_results if r["concurrency"] == 1)
    print(f"\nSingle-worker RPS (proxy for 1-core throughput): {c1['rps']:.1f}")
    threshold = 1.0
    status = "PASS" if c1["rps"] >= threshold else "FAIL"
    print(f"Threshold: >= {threshold} RPS/core  →  {status}")


if __name__ == "__main__":
    asyncio.run(main())
