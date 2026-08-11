"""Counterfactual 'what-if' recommendations for actionable clinical insight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.synthetic import FEATURE_LABELS
from src.models.ensemble import AdaptiveEnsemble


@dataclass
class CounterfactualAction:
    feature: str
    current: float
    proposed: float
    delta: float
    unit_hint: str
    new_risk: float
    risk_drop: float
    narrative: str


UNIT_HINTS = {
    "lactate": "mmol/L",
    "sofa": "points",
    "creatinine": "mg/dL",
    "wbc": "×10⁹/L",
    "glucose": "mg/dL",
    "bun": "mg/dL",
    "bilirubin": "mg/dL",
    "age": "years",
    "bmi": "kg/m²",
}


def generate_counterfactuals(
    ensemble: AdaptiveEnsemble,
    row: pd.Series,
    vitals: np.ndarray,
    base_risk: float,
    top_k: int = 4,
) -> list[CounterfactualAction]:
    """Greedy actionable nudges on lab/acuity features (demo-safe, not clinical advice)."""
    candidates = [
        ("lactate", -0.8, "Improve tissue perfusion / clear lactate"),
        ("sofa", -2.0, "Reduce organ dysfunction burden"),
        ("creatinine", -0.4, "Stabilize renal markers"),
        ("wbc", -2.5, "Control inflammatory burden"),
        ("glucose", -25.0, "Tighten glycemic control"),
        ("bun", -8.0, "Improve metabolic clearance"),
        ("bilirubin", -0.4, "Support hepatic recovery"),
    ]

    actions: list[CounterfactualAction] = []
    for key, step, why in candidates:
        if key not in row:
            continue
        current = float(row[key])
        proposed = float(max(current + step, 0.1))
        if abs(proposed - current) < 1e-6:
            continue

        cf = row.copy()
        cf[key] = proposed
        # Mild vital improvement when lactate/sofa improve
        v = vitals.copy()
        if key in {"lactate", "sofa"}:
            v[:, 0] = v[:, 0] - 3
            v[:, 1] = v[:, 1] + 4
            v[:, 4] = np.clip(v[:, 4] + 1.2, 70, 100)

        new_risk = ensemble.predict_one(cf, v).probability
        drop = base_risk - new_risk
        if drop <= 0.005:
            continue

        label = FEATURE_LABELS.get(key, key)
        actions.append(
            CounterfactualAction(
                feature=label,
                current=current,
                proposed=proposed,
                delta=proposed - current,
                unit_hint=UNIT_HINTS.get(key, ""),
                new_risk=new_risk,
                risk_drop=drop,
                narrative=f"{why}: {label} {current:.1f} → {proposed:.1f} lowers risk by {drop:.1%}.",
            )
        )

    actions.sort(key=lambda a: a.risk_drop, reverse=True)
    return actions[:top_k]
