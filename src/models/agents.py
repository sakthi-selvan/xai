"""Specialized clinical agents (TabNet / Bi-LSTM / CNN proxies for demo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.synthetic import DEMOGRAPHIC_COLS, LAB_COLS, vital_features


@dataclass
class AgentPrediction:
    name: str
    short: str
    probability: float
    confidence: float
    rationale: str
    modality: str


class ClinicalAgents:
    """Three modality specialists + shared preprocessing."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.tabnet = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=seed,
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.08,
                    ),
                ),
            ]
        )
        self.bilstm = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        max_iter=400,
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.15,
                    ),
                ),
            ]
        )
        self.cnn = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=8,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.fitted = False

    def _tabular(self, frame) -> np.ndarray:
        return frame[DEMOGRAPHIC_COLS + LAB_COLS].to_numpy(dtype=float)

    def _seq_stats(self, vitals: np.ndarray) -> np.ndarray:
        return vital_features(vitals)

    def _local_patterns(self, vitals: np.ndarray) -> np.ndarray:
        # CNN-style local windows: last 6h deltas + rolling means
        last6 = vitals[:, -6:, :]
        early6 = vitals[:, :6, :]
        mid = vitals[:, 9:15, :]
        return np.hstack(
            [
                last6.mean(axis=1),
                last6.std(axis=1),
                (last6.mean(axis=1) - early6.mean(axis=1)),
                mid.mean(axis=1),
                vitals[:, -1, :] - vitals[:, -2, :],
            ]
        )

    def fit(self, frame, vitals: np.ndarray, labels: np.ndarray) -> "ClinicalAgents":
        x_tab = self._tabular(frame)
        x_seq = self._seq_stats(vitals)
        x_cnn = self._local_patterns(vitals)
        self.tabnet.fit(x_tab, labels)
        self.bilstm.fit(x_seq, labels)
        self.cnn.fit(x_cnn, labels)
        self.fitted = True
        return self

    def _proba(self, model, x: np.ndarray) -> float:
        return float(model.predict_proba(x)[0, 1])

    def predict_one(self, row, vitals_one: np.ndarray) -> list[AgentPrediction]:
        assert self.fitted
        frame = row.to_frame().T
        x_tab = self._tabular(frame)
        x_seq = self._seq_stats(vitals_one[None, ...])
        x_cnn = self._local_patterns(vitals_one[None, ...])

        p_tab = self._proba(self.tabnet, x_tab)
        p_seq = self._proba(self.bilstm, x_seq)
        p_cnn = self._proba(self.cnn, x_cnn)

        lactate = float(row["lactate"])
        sofa = float(row["sofa"])
        hr_delta = float(vitals_one[-1, 0] - vitals_one[0, 0])
        spo2_min = float(vitals_one[:, 4].min())
        sbp_drop = float(vitals_one[0, 1] - vitals_one[-1, 1])

        return [
            AgentPrediction(
                name="TabNet Agent",
                short="TabNet",
                probability=p_tab,
                confidence=0.55 + 0.4 * abs(p_tab - 0.5) * 2,
                rationale=(
                    f"Structured view: SOFA {sofa:.0f}, lactate {lactate:.1f}. "
                    "Attends demographics + labs for static acuity."
                ),
                modality="Demographics + Labs",
            ),
            AgentPrediction(
                name="Bi-LSTM Agent",
                short="Bi-LSTM",
                probability=p_seq,
                confidence=0.55 + 0.4 * abs(p_seq - 0.5) * 2,
                rationale=(
                    f"Temporal trajectory: HR Δ {hr_delta:+.0f}, SpO₂ floor {spo2_min:.0f}%. "
                    "Models bidirectional vital-sign dynamics."
                ),
                modality="24h Vital Signs",
            ),
            AgentPrediction(
                name="CNN-1D Agent",
                short="CNN-1D",
                probability=p_cnn,
                confidence=0.55 + 0.4 * abs(p_cnn - 0.5) * 2,
                rationale=(
                    f"Local pattern scan: SBP drop {sbp_drop:.0f} mmHg in window. "
                    "Detects short-horizon deterioration motifs."
                ),
                modality="Local Vital Windows",
            ),
        ]


class CalibratedBaseline:
    """Simple calibrated baseline for comparison charts."""

    def __init__(self, seed: int = 42):
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=seed)),
            ]
        )

    def fit(self, frame, vitals: np.ndarray, labels: np.ndarray) -> "CalibratedBaseline":
        from src.data.synthetic import DEMOGRAPHIC_COLS, LAB_COLS

        x = np.hstack([frame[DEMOGRAPHIC_COLS + LAB_COLS].to_numpy(), vital_features(vitals)])
        self.model.fit(x, labels)
        return self

    def predict_proba(self, frame, vitals: np.ndarray) -> np.ndarray:
        from src.data.synthetic import DEMOGRAPHIC_COLS, LAB_COLS

        x = np.hstack([frame[DEMOGRAPHIC_COLS + LAB_COLS].to_numpy(), vital_features(vitals)])
        return self.model.predict_proba(x)[:, 1]
