"""Command line entry point:  python -m fraud_detection.train"""

from __future__ import annotations

import argparse

from .modeling import run_training


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the fraud detection models")
    ap.add_argument("--n-transactions", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="smaller models for a quick run")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()
    summary = run_training(
        n_transactions=args.n_transactions, seed=args.seed, fast=args.fast,
        out_dir=args.out_dir, reports_dir=args.reports_dir,
    )
    best = summary["models"][summary["best_model"]]
    print(f"\nsaved {summary['version']} | best: {summary['best_model']} "
          f"(F1 {best['f1']:.3f}, ROC-AUC {best['roc_auc']:.3f}) | {summary['total_seconds']}s")


if __name__ == "__main__":
    main()
