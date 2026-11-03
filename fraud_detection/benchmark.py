"""Scoring latency benchmark:  python -m fraud_detection.benchmark

Measures (1) the in-process scoring function and (2) the full HTTP round trip
to the API on localhost, using a fresh simulated transaction stream. These are
*local single-process* numbers; a deployed service adds network latency.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import tempfile
import threading
from pathlib import Path

import numpy as np

from .api import build_server
from .data import generate_transactions
from .preprocess import clean_transactions
from .scoring import Scorer, limit_threads_for_serving


def _payloads(n: int, seed: int) -> list:
    raw = generate_transactions(n_transactions=max(n + 200, 1500), seed=seed, inject_quality_issues=False)
    clean, _ = clean_transactions(raw)
    clean = clean.sort_values("timestamp").head(n)
    out = []
    for r in clean.itertuples():
        out.append({
            "customer_id": r.customer_id, "amount": float(r.amount), "timestamp": r.timestamp.isoformat(),
            "merchant_category": r.merchant_category, "channel": r.channel,
            "device_id": r.device_id, "city": r.city,
        })
    return out


def _summary(ms: list) -> dict:
    a = np.asarray(ms)
    return {
        "requests": int(len(a)), "mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2), "p99_ms": round(float(np.percentile(a, 99)), 2),
        "max_ms": round(float(a.max()), 2), "throughput_per_s": round(float(1000.0 / a.mean()), 1),
    }


def run_benchmark(model_dir=None, n_requests: int = 1000, seed: int = 999, reports_dir=None) -> dict:
    payloads = _payloads(n_requests, seed)

    limit_threads_for_serving()  # same setting the API process uses
    scorer = Scorer.load(model_dir)
    scorer.score(payloads[0])  # warm-up
    in_proc = [scorer.score(p)["latency_ms"] for p in payloads]

    import time
    with tempfile.TemporaryDirectory() as tmp:
        server = build_server("127.0.0.1", 0, Scorer.load(model_dir), db_path=f"{tmp}/c.db", outbox_path=f"{tmp}/o.jsonl")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        http_ms = []
        for p in payloads:
            body = json.dumps(p)
            t0 = time.perf_counter()
            conn.request("POST", "/score", body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            resp.read()
            http_ms.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status == 200, resp.status
        conn.close()
        server.shutdown()

    result = {
        "in_process_scoring": _summary(in_proc),
        "http_round_trip_localhost": _summary(http_ms),
        "environment": {"cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()},
        "note": "Local, single-process, sequential requests. A cloud deployment adds network latency.",
    }
    if reports_dir:
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        (Path(reports_dir) / "latency.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()
    print(json.dumps(run_benchmark(args.model_dir, args.requests, reports_dir=args.reports_dir), indent=2))


if __name__ == "__main__":
    main()
