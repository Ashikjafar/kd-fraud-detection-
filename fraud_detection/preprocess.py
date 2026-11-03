"""Data preprocessing: cleaning, encoding/scaling and SMOTE for class imbalance."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CATEGORICAL_FEATURES, CHANNELS, MERCHANT_CATEGORIES, NUMERIC_FEATURES,
)


def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Remove duplicates/invalid rows and fill missing categorical values.

    Returns the cleaned frame and a small report of what was changed.
    """
    report = {"rows_in": int(len(df))}
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")

    before = len(d)
    d = d.drop_duplicates(subset="transaction_id", keep="first")
    report["duplicates_removed"] = int(before - len(d))

    before = len(d)
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d = d.dropna(subset=["customer_id", "timestamp", "amount"])
    d = d[d["amount"] > 0]
    report["invalid_rows_removed"] = int(before - len(d))

    report["missing_category_filled"] = int(d["merchant_category"].isna().sum())
    report["missing_city_left_as_unknown"] = int(d["city"].isna().sum())
    report["missing_channel_filled"] = int(d["channel"].isna().sum())
    d["merchant_category"] = d["merchant_category"].fillna("unknown").str.strip().str.lower()
    d["channel"] = d["channel"].fillna("unknown").str.strip().str.lower()

    d = d.reset_index(drop=True)
    report["rows_out"] = int(len(d))
    return d, report


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categoricals (fitted on training data only)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[MERCHANT_CATEGORIES, CHANNELS],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    sampling_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE oversampling (Chawla et al., 2002).

    New minority samples are created on the line segment between a minority
    sample and one of its k nearest minority neighbours. ``sampling_ratio`` is
    the desired minority/majority ratio after resampling. Apply to the
    *training* split only.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    minority = X[y == 1]
    n_min, n_maj = len(minority), int((y == 0).sum())
    n_new = int(sampling_ratio * n_maj) - n_min
    if n_new <= 0 or n_min < 2:
        return X.copy(), y.copy()

    rng = np.random.default_rng(random_state)
    k = min(k_neighbors, n_min - 1)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(minority).kneighbors(minority)
    base = rng.integers(0, n_min, n_new)
    neighbour = idx[base, rng.integers(1, k + 1, n_new)]
    gap = rng.random((n_new, 1))
    synthetic = minority[base] + gap * (minority[neighbour] - minority[base])
    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
    return X_res, y_res
