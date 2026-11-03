"""Real-time scoring: raw transaction in, fraud probability + decision out."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS, CustomerProfileStore, build_features
from .rules import RuleConfig, decide


def limit_threads_for_serving() -> None:
    """Use one BLAS/OpenMP thread per process.

    Scoring is one row at a time, so a thread pool only adds overhead (measured: ~3x
    slower HTTP responses on a 2-CPU machine). Scale serving by running more
    processes/containers instead. Call this in the serving process only, not in training.
    """
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


class Scorer:
    """Holds the trained model, the customer profile store and the rules."""

    def __init__(self, bundle: dict, rules: Optional[RuleConfig] = None) -> None:
        self.bundle = bundle
        self.pre = bundle["preprocessor"]
        self.model = bundle["model"]
        self.t_review = float(bundle["t_review"])
        self.t_block = float(bundle["t_block"])
        self.version = bundle["version"]
        self.rules = rules or RuleConfig()
        self.store = CustomerProfileStore()
        self._lock = threading.Lock()  # profile updates must be serial per process

    @classmethod
    def load(cls, model_dir: Optional[str] = None, models_root: str = "models", rules: Optional[RuleConfig] = None) -> "Scorer":
        if model_dir is None:
            registry = json.loads((Path(models_root) / "registry.json").read_text())
            model_dir = str(Path(models_root) / registry["latest"])
        return cls(joblib.load(Path(model_dir) / "model.joblib"), rules)

    def warm_start(self, cleaned: pd.DataFrame) -> int:
        """Replay historical transactions so customers start with a behavioural profile."""
        with self._lock:
            build_features(cleaned, self.store)
        return len(cleaned)

    @staticmethod
    def _parse(txn: dict) -> dict:
        if not isinstance(txn, dict):
            raise ValueError("body must be a JSON object")
        cid = str(txn.get("customer_id", "")).strip()
        if not cid:
            raise ValueError("customer_id is required")
        try:
            amount = float(txn["amount"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("amount is required and must be a number") from None
        if not amount > 0 or amount != amount or amount == float("inf"):
            raise ValueError("amount must be a positive, finite number")
        ts_raw = txn.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw)).replace(tzinfo=None) if ts_raw else datetime.now()
        except ValueError:
            raise ValueError("timestamp must be ISO-8601, e.g. 2026-11-03T14:30:00") from None
        return {
            "customer_id": cid, "amount": amount,
            "ts": (ts - datetime(1970, 1, 1)).total_seconds(),
            "city": txn.get("city"), "device_id": txn.get("device_id"),
            "merchant_category": txn.get("merchant_category"), "channel": txn.get("channel"),
            "transaction_id": str(txn.get("transaction_id") or f"TXN-{uuid.uuid4().hex[:12]}"),
        }

    def score(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        t = self._parse(txn)
        with self._lock:
            feats = self.store.compute_and_update(
                t["customer_id"], t["ts"], t["amount"], t["city"], t["device_id"],
                t["merchant_category"], t["channel"],
            )
        row = pd.DataFrame([feats])[FEATURE_COLUMNS]
        proba = float(self.model.predict_proba(self.pre.transform(row))[:, 1][0])
        verdict = decide(proba, self.t_review, self.t_block, feats, t["customer_id"], t["amount"], self.rules)
        return {
            "transaction_id": t["transaction_id"],
            "customer_id": t["customer_id"],
            "fraud_probability": round(proba, 4),
            **verdict,
            "model_version": self.version,
            "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats.items()},
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
