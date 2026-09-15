"""Holdout evaluation and persisted training reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_models(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    """Evaluate probability predictions and persist a compact report bundle."""
    y_true = np.asarray(labels, dtype=int)
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    issues: list[str] = []

    for name, probabilities in predictions.items():
        probabilities = np.asarray(probabilities, dtype=float)
        predicted = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "auroc": float(roc_auc_score(y_true, probabilities)),
            "auprc": float(average_precision_score(y_true, probabilities)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
        metric_rows.append(row)
        confusion_rows.extend(
            [
                {"model": name, "actual": 0, "predicted": 0, "count": int(tn)},
                {"model": name, "actual": 0, "predicted": 1, "count": int(fp)},
                {"model": name, "actual": 1, "predicted": 0, "count": int(fn)},
                {"model": name, "actual": 1, "predicted": 1, "count": int(tp)},
            ]
        )
        if row["recall"] < 0.70:
            issues.append(f"{name}: recall is {row['recall']:.1%}; review missed positive cases.")
        if row["precision"] < 0.70:
            issues.append(f"{name}: precision is {row['precision']:.1%}; review false alarms.")
        if row["false_negatives"] > row["false_positives"]:
            issues.append(
                f"{name}: false negatives ({row['false_negatives']}) exceed false positives "
                f"({row['false_positives']})."
            )

    prevalence = float(y_true.mean())
    if prevalence < 0.20 or prevalence > 0.80:
        issues.append(f"Holdout prevalence is {prevalence:.1%}; accuracy may be misleading.")
    if not issues:
        issues.append("No threshold or class-balance warning was detected in this holdout run.")

    report = {
        "status": "trained_and_evaluated",
        "seed": int(seed),
        "n_holdout": int(len(y_true)),
        "positive_prevalence": prevalence,
        "models": metric_rows,
        "issues": issues,
    }
    _save_report(report, metric_rows, confusion_rows)
    return report


def _save_report(
    report: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
) -> None:
    report_dir = Path(__file__).resolve().parents[1] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metric_rows).to_csv(report_dir / "model_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(report_dir / "confusion_matrices.csv", index=False)
    (report_dir / "model_insights.json").write_text(
        json.dumps({"issues": report["issues"]}, indent=2), encoding="utf-8"
    )