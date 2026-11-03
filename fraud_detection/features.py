"""Behavioural feature engineering shared by training and real-time scoring.

The same ``CustomerProfileStore`` is used to build the training features (by
replaying the history in time order) and to score live transactions, so the
model sees identical features in both places (no train/serve skew).

Every feature for a transaction is computed from *earlier* transactions of the
same customer only, so no label or future information leaks into training.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set

import pandas as pd

DAY_SECONDS = 86_400.0
MAX_GAP_MIN = 7 * 24 * 60.0  # cap for "time since last transaction" (also used for a first transaction)

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "travel", "electronics",
    "fashion", "utilities", "entertainment", "jewellery", "unknown",
]
CHANNELS = ["pos", "online", "upi", "atm", "unknown"]

NUMERIC_FEATURES = [
    "log_amount",
    "hour_sin",
    "hour_cos",
    "is_night",
    "txn_count_24h",
    "amount_ratio",
    "time_since_last_min",
    "location_change",
    "new_device",
    "txn_index",
]
CATEGORICAL_FEATURES = ["merchant_category", "channel"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _norm_cat(value, default: str = "unknown") -> str:
    if _missing(value):
        return default
    text = str(value).strip().lower()
    return text or default


@dataclass
class _Profile:
    n: int = 0
    total: float = 0.0
    last_ts: Optional[float] = None
    last_city: Optional[str] = None
    devices: Set[str] = field(default_factory=set)
    recent: Deque[float] = field(default_factory=deque)


class CustomerProfileStore:
    """In-memory behavioural profile per customer (frequency, spend, device, location)."""

    def __init__(self) -> None:
        self._profiles: Dict[str, _Profile] = {}

    def __len__(self) -> int:
        return len(self._profiles)

    def compute_and_update(
        self,
        customer_id: str,
        ts: float,
        amount: float,
        city,
        device_id,
        merchant_category,
        channel,
    ) -> dict:
        """Return the feature dict for this transaction, then add it to the profile."""
        p = self._profiles.setdefault(str(customer_id), _Profile())

        # Drop transactions older than 24h from the sliding window.
        while p.recent and ts - p.recent[0] > DAY_SECONDS:
            p.recent.popleft()

        avg_prior = (p.total / p.n) if p.n else float(amount)
        gap_min = MAX_GAP_MIN if p.last_ts is None else min(max(ts - p.last_ts, 0.0) / 60.0, MAX_GAP_MIN)

        has_city = not _missing(city)
        has_device = not _missing(device_id)
        location_change = int(p.n > 0 and has_city and p.last_city is not None and str(city) != p.last_city)
        new_device = int(p.n > 0 and has_device and str(device_id) not in p.devices)

        hour = (ts % DAY_SECONDS) / 3600.0
        feats = {
            "log_amount": math.log1p(float(amount)),
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "is_night": int(hour < 5.0 or hour >= 23.0),
            "txn_count_24h": len(p.recent),
            "amount_ratio": float(amount) / max(avg_prior, 1.0),
            "time_since_last_min": gap_min,
            "location_change": location_change,
            "new_device": new_device,
            "txn_index": p.n,
            "merchant_category": _norm_cat(merchant_category),
            "channel": _norm_cat(channel),
        }

        # Update profile with the current transaction.
        p.n += 1
        p.total += float(amount)
        p.last_ts = ts if p.last_ts is None else max(p.last_ts, ts)
        if has_city:
            p.last_city = str(city)
        if has_device:
            p.devices.add(str(device_id))
        p.recent.append(ts)
        return feats


def to_seconds(timestamps: pd.Series) -> pd.Series:
    """Naive datetimes -> float seconds since 1970-01-01."""
    return (pd.to_datetime(timestamps) - pd.Timestamp("1970-01-01")).dt.total_seconds()


def build_features(df: pd.DataFrame, store: Optional[CustomerProfileStore] = None) -> pd.DataFrame:
    """Add behavioural features to a cleaned transaction frame (replayed in time order)."""
    store = store if store is not None else CustomerProfileStore()
    d = df.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    secs = to_seconds(d["timestamp"]).tolist()
    rows = [
        store.compute_and_update(cid, ts, amt, city, dev, cat, ch)
        for cid, ts, amt, city, dev, cat, ch in zip(
            d["customer_id"], secs, d["amount"], d["city"], d["device_id"],
            d["merchant_category"], d["channel"],
        )
    ]
    feats = pd.DataFrame(rows)
    # Feature columns replace any same-named raw columns (categoricals were normalised).
    d = d.drop(columns=[c for c in feats.columns if c in d.columns])
    return pd.concat([d, feats], axis=1)
