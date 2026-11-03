"""Model training, comparison, threshold selection and the model registry.

Pipeline
    raw -> clean -> behavioural features -> time-ordered split (train/val/test)
    -> scale/encode (fit on train) -> SMOTE (train only) -> fit models
    -> choose best supervised model on the *validation* split
    -> choose decision thresholds on validation -> report on the untouched *test* split.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)

from . import __version__
from .data import generate_transactions
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_features
from .preprocess import build_preprocessor, clean_transactions, smote


# ----------------------------------------------------------------------------- helpers
def time_split(df: pd.DataFrame, train_frac: float = 0.60, val_frac: float = 0.15):
    """Chronological split: older data trains, newer data validates/tests (no look-ahead)."""
    d = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(d)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]


def classification_metrics(y, proba, threshold: float) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def best_f1_threshold(y, proba) -> float:
    p, r, thr = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def block_threshold(y, proba, t_review: float, min_precision: float = 0.95) -> float:
    """Lowest threshold whose precision on validation data is >= min_precision."""
    p, _, thr = precision_recall_curve(y, proba)
    ok = np.where(p[:-1] >= min_precision)[0]
    ok = [i for i in ok if thr[i] >= t_review]
    if ok:
        return float(thr[ok[0]])
    return float(min(1.0, t_review + (1.0 - t_review) / 2.0))


def _curve_points(y, proba, kind: str, max_points: int = 200):
    if kind == "roc":
        a, b, _ = roc_curve(y, proba)
    else:
        b, a, _ = precision_recall_curve(y, proba)  # a = recall (x), b = precision (y)
    idx = np.unique(np.linspace(0, len(a) - 1, min(max_points, len(a))).astype(int))
    return {"x": [round(float(v), 5) for v in a[idx]], "y": [round(float(v), 5) for v in b[idx]]}


def build_models(seed: int, fast: bool = False) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=60 if fast else 150, max_depth=16, min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        ),
    }
    try:  # use XGBoost when it is installed
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=80 if fast else 250, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=seed,
        )
    except ImportError:  # scikit-learn gradient boosting is the drop-in alternative
        models["Gradient Boosting (HistGB)"] = HistGradientBoostingClassifier(
            max_iter=80 if fast else 250, learning_rate=0.1, max_leaf_nodes=31, random_state=seed,
        )
    return models


def rule_baseline_flags(df: pd.DataFrame) -> np.ndarray:
    """Traditional fixed-rule check used as a comparison baseline."""
    return ((df["amount_ratio"] >= 4.0) | ((df["new_device"] == 1) & (df["location_change"] == 1))).astype(int).to_numpy()


# ----------------------------------------------------------------------------- main entry
def run_training(
    n_transactions: int = 80_000,
    seed: int = 42,
    out_dir: Optional[str] = "models",
    reports_dir: Optional[str] = "reports",
    data_path: Optional[str] = "data/raw_transactions.csv",
    fast: bool = False,
    verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t_all = time.time()

    raw = generate_transactions(n_transactions=n_transactions, seed=seed)
    if data_path:
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(data_path, index=False)
    clean, clean_report = clean_transactions(raw)
    feat = build_features(clean)
    train, val, test = time_split(feat)
    log(f"rows: train={len(train)} val={len(val)} test={len(test)} | fraud rate {feat.is_fraud.mean():.3%}")

    pre = build_preprocessor()
    X_tr = pre.fit_transform(train[FEATURE_COLUMNS])
    X_va, X_te = pre.transform(val[FEATURE_COLUMNS]), pre.transform(test[FEATURE_COLUMNS])
    y_tr, y_va, y_te = train["is_fraud"].to_numpy(), val["is_fraud"].to_numpy(), test["is_fraud"].to_numpy()
    X_sm, y_sm = smote(X_tr, y_tr, sampling_ratio=0.5, random_state=seed)
    log(f"SMOTE: {int(y_tr.sum())} fraud / {int((y_tr == 0).sum())} genuine -> {int(y_sm.sum())} / {int((y_sm == 0).sum())}")

    results: Dict[str, dict] = {}
    fitted: Dict[str, object] = {}
    val_proba: Dict[str, np.ndarray] = {}
    test_proba: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}

    for name, model in build_models(seed, fast).items():
        t0 = time.time()
        model.fit(X_sm, y_sm)
        fit_s = time.time() - t0
        pv, pt = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
        thr = best_f1_threshold(y_va, pv)
        m = classification_metrics(y_te, pt, thr)
        m.update(kind="supervised", fit_seconds=round(fit_s, 2), val_pr_auc=float(average_precision_score(y_va, pv)))
        results[name], fitted[name] = m, model
        val_proba[name], test_proba[name], thresholds[name] = pv, pt, thr
        log(f"{name:28s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} AUC={m['roc_auc']:.3f} ({fit_s:.1f}s)")

    # Isolation Forest: unsupervised anomaly detector (never sees labels or SMOTE data).
    iso = IsolationForest(n_estimators=100 if fast else 200, random_state=seed, n_jobs=-1).fit(X_tr)
    s_va, s_te = -iso.score_samples(X_va), -iso.score_samples(X_te)
    iso_thr = best_f1_threshold(y_va, s_va)
    m = classification_metrics(y_te, s_te, iso_thr)
    m.update(kind="unsupervised", fit_seconds=0.0, val_pr_auc=float(average_precision_score(y_va, s_va)))
    results["Isolation Forest"] = m
    test_proba["Isolation Forest"] = s_te
    log(f"{'Isolation Forest':28s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} AUC={m['roc_auc']:.3f}")

    # Fixed-rule baseline (what the report calls "traditional fixed rules").
    rule_pred = rule_baseline_flags(test)
    m = classification_metrics(y_te, rule_pred.astype(float), 0.5)
    m.update(kind="baseline", fit_seconds=0.0, val_pr_auc=None)
    results["Fixed rules (baseline)"] = m
    log(f"{'Fixed rules (baseline)':28s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")

    # Choose the best supervised model on validation PR-AUC (never on test).
    sup = [n for n, r in results.items() if r["kind"] == "supervised"]
    best = max(sup, key=lambda n: results[n]["val_pr_auc"])
    t_review = thresholds[best]
    t_block = block_threshold(y_va, val_proba[best], t_review)
    log(f"best model (validation PR-AUC): {best} | review>={t_review:.3f} block>={t_block:.3f}")

    # Ablation: effect of SMOTE on the best model (trained without resampling).
    ablation = {}
    plain = build_models(seed, fast)[best].fit(X_tr, y_tr)
    pv0, pt0 = plain.predict_proba(X_va)[:, 1], plain.predict_proba(X_te)[:, 1]
    ablation["without_smote"] = classification_metrics(y_te, pt0, best_f1_threshold(y_va, pv0))
    ablation["with_smote"] = results[best]

    # Feature importance of the chosen model (permutation importance on validation sample).
    names = list(pre.get_feature_names_out())
    rs = np.random.default_rng(seed)
    sample = rs.choice(len(y_va), size=min(6000, len(y_va)), replace=False)
    pi = permutation_importance(
        fitted[best], X_va[sample], y_va[sample], scoring="average_precision",
        n_repeats=3, random_state=seed, n_jobs=1,
    )
    importance = sorted(
        ({"feature": f, "importance": float(v)} for f, v in zip(names, pi.importances_mean)),
        key=lambda d: d["importance"], reverse=True,
    )

    # Dataset summary used by the dashboard/report.
    hours = feat.assign(hour=pd.to_datetime(feat["timestamp"]).dt.hour)
    by_hour = hours.groupby("hour")["is_fraud"].agg(["sum", "count"])
    dataset = {
        "rows_raw": clean_report["rows_in"], "rows_clean": clean_report["rows_out"],
        "fraud_rows": int(feat.is_fraud.sum()), "fraud_rate": float(feat.is_fraud.mean()),
        "cleaning": clean_report,
        "splits": {
            n: {"rows": int(len(s)), "fraud": int(s.is_fraud.sum()), "from": str(s.timestamp.min()), "to": str(s.timestamp.max())}
            for n, s in (("train", train), ("validation", val), ("test", test))
        },
        "smote": {"fraud_before": int(y_tr.sum()), "genuine_before": int((y_tr == 0).sum()),
                  "fraud_after": int(y_sm.sum()), "genuine_after": int((y_sm == 0).sum())},
        "fraud_rate_by_hour": [
            {"hour": int(h), "fraud": int(r["sum"]), "total": int(r["count"])} for h, r in by_hour.iterrows()
        ],
    }

    version = f"v{__version__}-seed{seed}-n{n_transactions}"
    summary = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_data": True,
        "seed": seed,
        "best_model": best,
        "boosting_backend": "xgboost" if "XGBoost" in results else "sklearn HistGradientBoosting",
        "thresholds": {"review": t_review, "block": t_block},
        "models": results,
        "ablation_smote": ablation,
        "feature_importance": importance,
        "dataset": dataset,
        "environment": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "total_seconds": round(time.time() - t_all, 1),
    }
    curves = {
        n: {"roc": _curve_points(y_te, p, "roc"), "pr": _curve_points(y_te, p, "pr")}
        for n, p in test_proba.items() if n != "Fixed rules (baseline)"
    }

    if out_dir:
        vdir = Path(out_dir) / version
        vdir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "version": version, "model_name": best, "preprocessor": pre, "model": fitted[best],
            "t_review": t_review, "t_block": t_block, "feature_columns": FEATURE_COLUMNS,
            "categorical_features": CATEGORICAL_FEATURES, "test_metrics": results[best],
        }
        joblib.dump(bundle, vdir / "model.joblib", compress=3)
        (vdir / "metadata.json").write_text(json.dumps(summary, indent=2))
        reg_path = Path(out_dir) / "registry.json"
        registry = json.loads(reg_path.read_text()) if reg_path.exists() else {"versions": {}}
        registry["latest"] = version
        registry["versions"][version] = {
            "model": best, "trained_at": summary["trained_at"],
            "test_f1": results[best]["f1"], "test_roc_auc": results[best]["roc_auc"],
        }
        reg_path.write_text(json.dumps(registry, indent=2))
        summary["model_dir"] = str(vdir)

    if reports_dir:
        rdir = Path(reports_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "metrics.json").write_text(json.dumps(summary, indent=2))
        (rdir / "curves.json").write_text(json.dumps(curves))
    summary["curves"] = curves
    return summary
