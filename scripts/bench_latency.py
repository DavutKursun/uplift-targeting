"""Latency of /score: p50 / p95 at batch 1 for both deployed models, in-process (TestClient) and over HTTP (uvicorn subprocess),
and the throughput of one /score/batch call with 1,000 customers. Writes eval/results/latency_local.json. Target: p50 < 30 ms.

Usage: python scripts/bench_latency.py [--calls 500] [--out eval/results/latency_local.json] [--no-http]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import FEATURES
from uplift.serving import DEPLOYED, SERVING

TARGET_P50_MS = 30


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "mean_ms": round(float(a.mean()), 2)}


def customers(n: int) -> list[dict]:
    sample = pd.read_csv(SERVING / "customers_dev_sample.csv")
    rows = [{f: float(sample.iloc[i][f]) for f in FEATURES} for i in range(min(len(sample), 50))]
    return [rows[i % len(rows)] for i in range(n)]


def bench_client(post, calls: int, label: str) -> dict:
    out = {}
    body = customers(1)[0]
    for model in DEPLOYED:
        for _ in range(20):  # warm-up
            post(f"/score?model={model}", body)
        ms = []
        for _ in range(calls):
            t0 = time.perf_counter()
            r = post(f"/score?model={model}", body)
            ms.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200, r.text
        out[model] = _stats(ms)
        print(f"{label} {model}: p50 {out[model]['p50_ms']} ms, p95 {out[model]['p95_ms']} ms over {calls} calls", flush=True)
    batch = {"customers": customers(1000)}
    for model in DEPLOYED:
        t0 = time.perf_counter()
        r = post(f"/score/batch?model={model}", batch)
        ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, r.text
        out[f"batch1000_{model}"] = {"total_ms": round(ms, 1), "rows_per_s": round(1000 / (ms / 1000), 1), "ms_per_row": round(ms / 1000, 3)}
        print(f"{label} batch 1,000 {model}: {ms:.0f} ms total, {1000 / (ms / 1000):.0f} rows/s", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "latency_local.json")
    ap.add_argument("--no-http", action="store_true")
    a = ap.parse_args(argv)
    import logging

    from fastapi.testclient import TestClient

    import app as app_module

    for name in ("uplift.app", "httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
    client = TestClient(app_module.app)
    result = {"machine": {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(), "cpus": os.cpu_count(), "python": platform.python_version()},
              "budget": app_module.scorer.budget, "version": app_module.scorer.version, "calls_per_model": a.calls,
              "in_process": bench_client(lambda p, b: client.post(p, json=b), a.calls, "in-process")}
    if not a.no_http:
        port = _free_port()
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(120):
                try:
                    if httpx.get(base + "/health", timeout=1.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("uvicorn did not come up")
            with httpx.Client(base_url=base, timeout=120.0) as http:
                result["http"] = bench_client(lambda p, b: http.post(p, json=b), a.calls, "http")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    result["target_p50_ms"] = TARGET_P50_MS
    result["meets_target_in_process"] = all(result["in_process"][m]["p50_ms"] < TARGET_P50_MS for m in DEPLOYED)
    result["meets_target_http"] = all(result["http"][m]["p50_ms"] < TARGET_P50_MS for m in DEPLOYED) if "http" in result else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "machine"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
