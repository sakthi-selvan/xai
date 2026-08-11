"""Adaptive patient-specific meta-learner and cross-modal fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.synthetic import DEMOGRAPHIC_COLS, LAB_COLS, vital_features
from src.models.agents import ClinicalAgents


@dataclass
class EnsembleResult:
    probability: float
    weights: dict[str, float]
    agent_probs: dict[str, float]
    fused_embedding: np.ndarray
    attention: np.ndarray  # modality attention (3,)
    debate_notes: list[str]


class AdaptiveEnsemble:
    def __init__(self, agents: ClinicalAgents, seed: int = 42):
        self.agents = agents
        self.meta = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        max_iter=350,
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        self.gate = LogisticRegression(max_iter=400, random_state=seed)
        self.fitted = False

    def _agent_matrix(self, frame, vitals: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(len(frame)):
            row = frame.iloc[i]
            agents = self.agents.predict_one(row, vitals[i])
            preds.append([a.probability for a in agents])
        return np.asarray(preds, dtype=float)

    def _context(self, frame, vitals: np.ndarray) -> np.ndarray:
        tab = frame[DEMOGRAPHIC_COLS + LAB_COLS].to_numpy(dtype=float)
        seq = vital_features(vitals)
        return np.hstack([tab, seq])

    def fit(self, frame, vitals: np.ndarray, labels: np.ndarray) -> "AdaptiveEnsemble":
        agent_p = self._agent_matrix(frame, vitals)
        ctx = self._context(frame, vitals)
        meta_x = np.hstack([agent_p, ctx[:, :8]])  # agents + compact acuity context
        self.meta.fit(meta_x, labels)
        # Gate learns patient-specific mixture from acuity markers
        gate_x = np.column_stack(
            [
                frame["sofa"].to_numpy(),
                frame["lactate"].to_numpy(),
                frame["age"].to_numpy(),
                vitals[:, -1, 0],
                vitals[:, -1, 4],
                agent_p.std(axis=1),
            ]
        )
        self.gate.fit(gate_x, labels)
        self.fitted = True
        return self

    def _attention(self, agent_probs: np.ndarray, row, vitals_one: np.ndarray) -> np.ndarray:
        # Cross-modal attention proxy: acuity + disagreement aware
        sofa = float(row["sofa"]) / 15.0
        lactate = min(float(row["lactate"]) / 8.0, 1.5)
        volatility = float(np.std(vitals_one[:, 0]) / 25.0)
        disagreement = float(np.std(agent_probs))

        raw = np.array(
            [
                0.35 + 0.35 * sofa + 0.2 * lactate,  # TabNet / labs
                0.30 + 0.40 * volatility + 0.15 * disagreement,  # Bi-LSTM
                0.25 + 0.25 * (1 - vitals_one[-1, 4] / 100) + 0.2 * disagreement,  # CNN
            ],
            dtype=float,
        )
        raw = np.clip(raw, 0.08, None)
        return raw / raw.sum()

    def predict_one(self, row, vitals_one: np.ndarray) -> EnsembleResult:
        assert self.fitted
        agents = self.agents.predict_one(row, vitals_one)
        agent_probs = np.array([a.probability for a in agents], dtype=float)
        attn = self._attention(agent_probs, row, vitals_one)

        frame = row.to_frame().T
        ctx = self._context(frame, vitals_one[None, ...])
        meta_x = np.hstack([agent_probs[None, :], ctx[:, :8]])
        meta_p = float(self.meta.predict_proba(meta_x)[0, 1])

        # Blend meta-learner with attention-weighted agents (adaptive ensemble)
        weighted = float(np.dot(attn, agent_probs))
        probability = float(np.clip(0.65 * meta_p + 0.35 * weighted, 0.01, 0.99))

        notes = self._debate(agents, attn, probability)
        return EnsembleResult(
            probability=probability,
            weights={a.short: float(w) for a, w in zip(agents, attn)},
            agent_probs={a.short: a.probability for a in agents},
            fused_embedding=np.concatenate([agent_probs, attn]),
            attention=attn,
            debate_notes=notes,
        )

    def _debate(self, agents, attn: np.ndarray, final_p: float) -> list[str]:
        ordered = sorted(zip(agents, attn), key=lambda t: t[1], reverse=True)
        lead, lead_w = ordered[0]
        low = min(agents, key=lambda a: a.probability)
        high = max(agents, key=lambda a: a.probability)
        spread = high.probability - low.probability
        notes = [
            f"{lead.short} leads consensus (weight {lead_w:.0%}) via {lead.modality.lower()}.",
            f"Agent spread {spread:.0%} — {'escalating review' if spread > 0.18 else 'aligned views'}.",
            f"Collaborative risk settled at {final_p:.0%} after meta-learner adjudication.",
        ]
        if high.short != low.short and spread > 0.12:
            notes.append(
                f"Debate: {high.short} warns ({high.probability:.0%}) while "
                f"{low.short} is cautious ({low.probability:.0%})."
            )
        return notes

    def predict_batch(self, frame, vitals: np.ndarray) -> np.ndarray:
        return np.array(
            [self.predict_one(frame.iloc[i], vitals[i]).probability for i in range(len(frame))],
            dtype=float,
        )
