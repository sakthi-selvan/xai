"""Monte Carlo–style uncertainty quantification for clinical safety."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.ensemble import AdaptiveEnsemble


@dataclass
class UncertaintyReport:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    samples: np.ndarray
    needs_review: bool
    review_reason: str


def mc_dropout_estimate(
    ensemble: AdaptiveEnsemble,
    row,
    vitals_one: np.ndarray,
    n_samples: int = 40,
    seed: int = 7,
) -> UncertaintyReport:
    """Approximate epistemic uncertainty via stochastic forward passes."""
    rng = np.random.default_rng(seed)
    base = ensemble.predict_one(row, vitals_one)
    samples = []

    for _ in range(n_samples):
        # Perturb vitals slightly + jitter agent mixture (dropout-like)
        noise = rng.normal(0, 0.35, size=vitals_one.shape)
        scale = np.array([2.5, 3.0, 2.0, 0.8, 0.4, 0.08])
        noisy = vitals_one + noise * scale
        noisy[:, 4] = np.clip(noisy[:, 4], 70, 100)
        noisy[:, 5] = np.clip(noisy[:, 5], 35, 41)

        result = ensemble.predict_one(row, noisy)
        jitter = rng.normal(0, 0.03 + 0.04 * np.std(list(result.agent_probs.values())))
        samples.append(float(np.clip(result.probability + jitter, 0.01, 0.99)))

    arr = np.asarray(samples, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    ci_low, ci_high = np.percentile(arr, [5, 95]).tolist()

    # Also blend with base prediction for stability
    mean = float(0.7 * base.probability + 0.3 * mean)
    disagreement = float(np.std(list(base.agent_probs.values())))
    needs_review = std >= 0.07 or disagreement >= 0.16 or (0.35 <= mean <= 0.55)

    if needs_review:
        if 0.35 <= mean <= 0.55:
            reason = "Borderline probability band — clinician adjudication recommended."
        elif disagreement >= 0.16:
            reason = "High inter-agent disagreement — multimodal conflict detected."
        else:
            reason = "Wide confidence interval — epistemic uncertainty elevated."
    else:
        reason = "Calibrated confidence acceptable for triage support."

    return UncertaintyReport(
        mean=mean,
        std=std,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        samples=arr,
        needs_review=needs_review,
        review_reason=reason,
    )
