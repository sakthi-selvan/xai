"""Synthetic multimodal ICU cohort for faculty demo (MIMIC/eICU-like)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEMOGRAPHIC_COLS = ["age", "gender", "bmi", "sofa", "charlson", "los_days"]
LAB_COLS = [
    "wbc",
    "hemoglobin",
    "platelets",
    "creatinine",
    "bun",
    "lactate",
    "bilirubin",
    "glucose",
    "sodium",
    "potassium",
]
VITAL_NAMES = ["hr", "sbp", "dbp", "rr", "spo2", "temp"]
HOURS = 24


@dataclass
class PatientCase:
    patient_id: str
    demographics: dict[str, float]
    labs: dict[str, float]
    vitals: np.ndarray  # (hours, n_vitals)
    label: int
    risk_tier: str
    summary: str


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_cohort(n: int = 240, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return tabular frame, vital tensors (n, 24, 6), and binary mortality labels."""
    rng = np.random.default_rng(seed)

    age = rng.normal(66, 14, n).clip(18, 95)
    gender = rng.integers(0, 2, n)
    bmi = rng.normal(27, 5, n).clip(16, 45)
    sofa = rng.integers(0, 16, n).astype(float)
    charlson = rng.integers(0, 10, n).astype(float)
    los_days = rng.lognormal(1.2, 0.7, n).clip(0.5, 40)

    wbc = rng.normal(11, 4, n).clip(2, 35)
    hemoglobin = rng.normal(10.5, 2.0, n).clip(6, 16)
    platelets = rng.normal(180, 80, n).clip(20, 500)
    creatinine = rng.normal(1.4, 0.9, n).clip(0.4, 8)
    bun = rng.normal(28, 16, n).clip(5, 120)
    lactate = rng.normal(2.4, 1.6, n).clip(0.5, 15)
    bilirubin = rng.normal(1.2, 1.1, n).clip(0.2, 12)
    glucose = rng.normal(145, 45, n).clip(60, 400)
    sodium = rng.normal(138, 5, n).clip(120, 155)
    potassium = rng.normal(4.2, 0.6, n).clip(2.5, 6.5)

    # Time-series vitals with patient-level drift
    vitals = np.zeros((n, HOURS, len(VITAL_NAMES)))
    base = np.column_stack(
        [
            rng.normal(92, 18, n),
            rng.normal(118, 22, n),
            rng.normal(68, 12, n),
            rng.normal(20, 5, n),
            rng.normal(95, 4, n),
            rng.normal(37.0, 0.7, n),
        ]
    )
    for i in range(n):
        severity = (sofa[i] / 15) + (lactate[i] / 10)
        trend = np.linspace(0, severity * 0.8, HOURS)
        noise = rng.normal(0, 1, (HOURS, len(VITAL_NAMES)))
        vitals[i] = base[i] + noise * np.array([6, 8, 5, 2, 1.2, 0.25])
        vitals[i, :, 0] += trend * 25  # HR rising
        vitals[i, :, 1] -= trend * 20  # SBP falling
        vitals[i, :, 3] += trend * 8  # RR rising
        vitals[i, :, 4] -= trend * 6  # SpO2 falling
        vitals[i, :, 5] += trend * 0.8  # Temp rising

    vitals[:, :, 0] = vitals[:, :, 0].clip(40, 200)
    vitals[:, :, 1] = vitals[:, :, 1].clip(60, 220)
    vitals[:, :, 2] = vitals[:, :, 2].clip(30, 130)
    vitals[:, :, 3] = vitals[:, :, 3].clip(8, 45)
    vitals[:, :, 4] = vitals[:, :, 4].clip(70, 100)
    vitals[:, :, 5] = vitals[:, :, 5].clip(35, 41)

    hr_last = vitals[:, -1, 0]
    sbp_last = vitals[:, -1, 1]
    spo2_last = vitals[:, -1, 4]
    rr_last = vitals[:, -1, 3]

    logit = (
        -4.2
        + 0.025 * (age - 60)
        + 0.22 * sofa
        + 0.12 * charlson
        + 0.35 * (lactate - 2)
        + 0.18 * (creatinine - 1)
        + 0.04 * (wbc - 10)
        + 0.02 * (hr_last - 90)
        - 0.015 * (sbp_last - 120)
        - 0.08 * (spo2_last - 95)
        + 0.05 * (rr_last - 18)
        + rng.normal(0, 0.45, n)
    )
    prob = _sigmoid(logit)
    labels = (rng.random(n) < prob).astype(int)

    frame = pd.DataFrame(
        {
            "patient_id": [f"ICU-{1000 + i}" for i in range(n)],
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "sofa": sofa,
            "charlson": charlson,
            "los_days": los_days,
            "wbc": wbc,
            "hemoglobin": hemoglobin,
            "platelets": platelets,
            "creatinine": creatinine,
            "bun": bun,
            "lactate": lactate,
            "bilirubin": bilirubin,
            "glucose": glucose,
            "sodium": sodium,
            "potassium": potassium,
            "mortality": labels,
            "true_risk": prob,
        }
    )
    return frame, vitals, labels


def vital_features(vitals: np.ndarray) -> np.ndarray:
    """Compact temporal descriptors for meta-learner / agents."""
    mean = vitals.mean(axis=1)
    std = vitals.std(axis=1)
    last = vitals[:, -1, :]
    first = vitals[:, 0, :]
    slope = (last - first) / max(HOURS - 1, 1)
    return np.hstack([mean, std, last, slope])


def risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "Critical"
    if prob >= 0.40:
        return "Elevated"
    if prob >= 0.20:
        return "Moderate"
    return "Low"


def patient_summary(row: pd.Series, prob: float) -> str:
    gender = "Male" if int(row["gender"]) == 1 else "Female"
    return (
        f"{int(row['age'])}y {gender} · SOFA {int(row['sofa'])} · "
        f"Lactate {row['lactate']:.1f} · estimated {risk_tier(prob)} risk"
    )


def build_showcase_cases(frame: pd.DataFrame, vitals: np.ndarray) -> list[dict[str, Any]]:
    """Curated demo patients spanning risk spectrum."""
    ranked = frame.assign(_r=frame["true_risk"]).sort_values("_r")
    picks = [
        ranked.iloc[5],
        ranked.iloc[len(ranked) // 3],
        ranked.iloc[2 * len(ranked) // 3],
        ranked.iloc[-8],
        ranked.iloc[-2],
    ]
    cases: list[dict[str, Any]] = []
    for row in picks:
        idx = int(frame.index[frame["patient_id"] == row["patient_id"]][0])
        dem = {c: float(row[c]) for c in DEMOGRAPHIC_COLS}
        labs = {c: float(row[c]) for c in LAB_COLS}
        cases.append(
            {
                "patient_id": row["patient_id"],
                "demographics": dem,
                "labs": labs,
                "vitals": vitals[idx],
                "label": int(row["mortality"]),
                "risk_tier": risk_tier(float(row["true_risk"])),
                "summary": patient_summary(row, float(row["true_risk"])),
                "row_index": idx,
            }
        )
    return cases


FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender (M=1)",
    "bmi": "BMI",
    "sofa": "SOFA score",
    "charlson": "Charlson index",
    "los_days": "ICU LOS (days)",
    "wbc": "WBC",
    "hemoglobin": "Hemoglobin",
    "platelets": "Platelets",
    "creatinine": "Creatinine",
    "bun": "BUN",
    "lactate": "Lactate",
    "bilirubin": "Bilirubin",
    "glucose": "Glucose",
    "sodium": "Sodium",
    "potassium": "Potassium",
}
