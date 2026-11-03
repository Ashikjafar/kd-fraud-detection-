"""Synthetic card/UPI transaction generator.

Real fraud datasets cannot be redistributed, so the project ships a generator
that produces a *labelled, imbalanced* transaction stream with realistic
structure: each customer has a home city, usual devices, a typical spend level
and active hours; fraud arrives as events (account takeover, stolen card, card
testing bursts and subtle look-alike fraud). Legitimate traffic also contains
"suspicious-looking" behaviour (travel, new phones, big purchases), so the
classes overlap and the task is not trivially separable.

The output also contains deliberately dirty rows (missing values, duplicates)
so that the preprocessing module has real work to do.

Everything is driven by a seed, so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = [
    "Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Kolkata",
    "Pune", "Coimbatore", "Salem", "Madurai", "Ahmedabad", "Jaipur",
]
CATEGORIES = [
    "grocery", "fuel", "restaurant", "travel", "electronics",
    "fashion", "utilities", "entertainment", "jewellery",
]
LEGIT_CATEGORY_P = [0.26, 0.14, 0.16, 0.07, 0.06, 0.10, 0.12, 0.07, 0.02]
FRAUD_CATEGORY_P = [0.04, 0.04, 0.06, 0.18, 0.28, 0.12, 0.02, 0.10, 0.16]
DAY = 86_400


def generate_transactions(
    n_transactions: int = 80_000,
    n_customers: int = 2_500,
    fraud_rate: float = 0.02,
    days: int = 90,
    seed: int = 42,
    start: str = "2026-01-01",
    inject_quality_issues: bool = True,
) -> pd.DataFrame:
    """Return raw transactions (unsorted, with a few dirty rows)."""
    rng = np.random.default_rng(seed)

    # ---- customer behaviour profiles -------------------------------------------------
    home_city = rng.choice(CITIES, n_customers)
    mean_spend = rng.lognormal(np.log(1800), 0.55, n_customers)
    activity = rng.lognormal(0.0, 0.8, n_customers)
    activity /= activity.sum()
    night_owl = rng.random(n_customers) < 0.12
    online_pref = rng.beta(2, 3, n_customers)
    has_second_device = rng.random(n_customers) < 0.3

    def usual_device(c: int, second: bool) -> str:
        return f"D{c:05d}-{1 if (second and has_second_device[c]) else 0}"

    n_fraud_target = int(round(n_transactions * fraud_rate))
    n_legit = n_transactions - n_fraud_target

    # ---- legitimate traffic ----------------------------------------------------------
    cust = rng.choice(n_customers, size=n_legit, p=activity)
    day = rng.integers(0, days, n_legit)
    hour_day = np.clip(rng.normal(14, 4.5, n_legit), 6, 23.9)
    hour_night = rng.uniform(0, 6, n_legit)
    use_night = (night_owl[cust] & (rng.random(n_legit) < 0.35)) | (rng.random(n_legit) < 0.02)
    hour = np.where(use_night, hour_night, hour_day)
    seconds = day * DAY + hour * 3600.0

    amount = mean_spend[cust] * rng.lognormal(0.0, 0.6, n_legit)
    big = rng.random(n_legit) < 0.03  # occasional legitimate big purchase
    amount = np.where(big, amount * rng.lognormal(1.2, 0.4, n_legit), amount)
    amount = np.round(np.maximum(amount, 10.0), 2)

    category = rng.choice(CATEGORIES, size=n_legit, p=LEGIT_CATEGORY_P)
    r = rng.random(n_legit)
    channel = np.where(
        rng.random(n_legit) < online_pref[cust], "online",
        np.where(r < 0.45, "pos", np.where(r < 0.85, "upi", "atm")),
    )
    second = rng.random(n_legit) < 0.35
    device = np.array([usual_device(int(c), bool(s)) for c, s in zip(cust, second)], dtype=object)
    new_dev = rng.random(n_legit) < 0.03  # new phone / shared device
    device[new_dev] = [f"N{x:06d}" for x in rng.integers(0, 999_999, int(new_dev.sum()))]
    travel = rng.random(n_legit) < 0.06  # travelling customer
    city = np.where(travel, rng.choice(CITIES, n_legit), home_city[cust])

    legit = pd.DataFrame({
        "customer_id": [f"C{c:05d}" for c in cust],
        "seconds": seconds,
        "amount": amount,
        "merchant_category": category,
        "channel": channel,
        "device_id": device,
        "city": city,
        "is_fraud": 0,
    })

    # ---- fraud events ----------------------------------------------------------------
    fraud_rows = []
    kinds = ["takeover", "stolen_card", "card_testing", "subtle"]
    kind_p = [0.30, 0.30, 0.20, 0.20]
    while len(fraud_rows) < n_fraud_target:
        kind = rng.choice(kinds, p=kind_p)
        c = int(rng.integers(0, n_customers))
        d0 = int(rng.integers(0, days - 1))
        night = rng.random() < (0.5 if kind != "stolen_card" else 0.4)
        h0 = rng.uniform(0, 5.5) if night else rng.uniform(8, 22)
        t = d0 * DAY + h0 * 3600.0
        base = mean_spend[c]
        other_city = rng.choice([x for x in CITIES if x != home_city[c]])
        fraud_device = f"F{int(rng.integers(0, 999_999)):06d}"

        if kind == "takeover":
            k = int(rng.integers(1, 4))
            for _ in range(k):
                fraud_rows.append((
                    f"C{c:05d}", t, round(float(base * rng.lognormal(1.3, 0.5)), 2),
                    rng.choice(CATEGORIES, p=FRAUD_CATEGORY_P),
                    "online" if rng.random() < 0.85 else "upi",
                    fraud_device if rng.random() < 0.85 else usual_device(c, False),
                    other_city if rng.random() < 0.7 else home_city[c],
                ))
                t += rng.uniform(60, 900)
        elif kind == "stolen_card":
            k = int(rng.integers(1, 5))
            for _ in range(k):
                fraud_rows.append((
                    f"C{c:05d}", t, round(float(base * rng.lognormal(0.9, 0.5)), 2),
                    rng.choice(CATEGORIES, p=FRAUD_CATEGORY_P),
                    rng.choice(["pos", "atm", "online"], p=[0.45, 0.25, 0.30]),
                    fraud_device if rng.random() < 0.5 else usual_device(c, False),
                    other_city if rng.random() < 0.75 else home_city[c],
                ))
                t += rng.uniform(300, 3600)
        elif kind == "card_testing":
            k = int(rng.integers(3, 7))
            for j in range(k):
                small = j < k - 1
                amt = rng.uniform(20, 300) if small else base * rng.lognormal(1.0, 0.4)
                fraud_rows.append((
                    f"C{c:05d}", t, round(float(amt), 2),
                    rng.choice(["entertainment", "electronics", "fashion"]),
                    "online", fraud_device,
                    home_city[c] if rng.random() < 0.6 else other_city,
                ))
                t += rng.uniform(20, 180)
        else:  # subtle: looks almost normal except for the time of day
            fraud_rows.append((
                f"C{c:05d}", t, round(float(base * rng.lognormal(0.4, 0.5)), 2),
                rng.choice(CATEGORIES), "online" if rng.random() < 0.6 else "pos",
                usual_device(c, True), home_city[c],
            ))
    fraud_rows = fraud_rows[:n_fraud_target]
    fraud = pd.DataFrame(
        fraud_rows,
        columns=["customer_id", "seconds", "amount", "merchant_category", "channel", "device_id", "city"],
    )
    fraud["seconds"] = fraud["seconds"].clip(upper=days * DAY - 1)
    fraud["is_fraud"] = 1

    # ---- assemble --------------------------------------------------------------------
    df = pd.concat([legit, fraud], ignore_index=True).sort_values("seconds", kind="mergesort").reset_index(drop=True)
    df["timestamp"] = pd.Timestamp(start) + pd.to_timedelta(df.pop("seconds"), unit="s").dt.floor("s")
    df.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(len(df))])
    df = df[["transaction_id", "customer_id", "timestamp", "amount", "merchant_category",
             "channel", "device_id", "city", "is_fraud"]]

    if inject_quality_issues:
        n = len(df)
        df.loc[rng.random(n) < 0.010, "merchant_category"] = None
        df.loc[rng.random(n) < 0.005, "city"] = None
        df.loc[rng.random(n) < 0.002, "amount"] = np.nan
        dupes = df.sample(frac=0.003, random_state=seed)
        df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # raw feed arrives unsorted
