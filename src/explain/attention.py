"""Temporal attention visualization over 24h vital signs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.synthetic import VITAL_NAMES


@dataclass
class AttentionMaps:
    temporal: np.ndarray  # (24,)
    vital_importance: dict[str, float]
    heatmap: np.ndarray  # (6, 24) channel x time salience


def temporal_attention(vitals: np.ndarray, risk: float) -> AttentionMaps:
    """Soft attention over hours using deterioration cues."""
    # Normalize each channel
    x = vitals.copy()
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    z = (x - mu) / sd

    # Risk-aligned channel polarity (higher = worse for most; SpO2 inverted)
    polarity = np.array([1.0, -1.0, -0.6, 1.0, -1.2, 0.8])
    sal = z * polarity

    # Emphasize recent hours + rate of change
    hours = np.arange(vitals.shape[0])
    recency = 0.35 + 0.65 * (hours / hours.max())
    delta = np.abs(np.diff(sal, axis=0, prepend=sal[:1]))
    heatmap = np.maximum(0, sal.T) * recency + 0.35 * delta.T
    heatmap = heatmap / (heatmap.max() + 1e-9)

    temporal = heatmap.mean(axis=0)
    temporal = temporal / (temporal.sum() + 1e-9)

    # Slightly sharpen with risk
    temporal = temporal * (1 + 0.4 * risk)
    temporal = temporal / temporal.sum()

    vital_scores = heatmap.mean(axis=1)
    vital_scores = vital_scores / (vital_scores.sum() + 1e-9)
    vital_importance = {name.upper(): float(s) for name, s in zip(VITAL_NAMES, vital_scores)}

    return AttentionMaps(
        temporal=temporal,
        vital_importance=vital_importance,
        heatmap=heatmap,
    )
