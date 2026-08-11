"""Train and cache the full EECRPS-XAI prototype pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.synthetic import build_showcase_cases, generate_cohort
from src.explain.shap_explain import ShapExplainer
from src.models.agents import CalibratedBaseline, ClinicalAgents
from src.models.ensemble import AdaptiveEnsemble


@dataclass
class PipelineBundle:
    frame: pd.DataFrame
    vitals: np.ndarray
    labels: np.ndarray
    agents: ClinicalAgents
    ensemble: AdaptiveEnsemble
    baseline: CalibratedBaseline
    shap: ShapExplainer
    showcase: list[dict]
    metrics: dict[str, float]
    train_idx: np.ndarray
    test_idx: np.ndarray


def build_pipeline(seed: int = 42) -> PipelineBundle:
    frame, vitals, labels = generate_cohort(n=280, seed=seed)
    idx = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=seed, stratify=labels
    )

    agents = ClinicalAgents(seed=seed)
    agents.fit(frame.iloc[train_idx], vitals[train_idx], labels[train_idx])

    ensemble = AdaptiveEnsemble(agents, seed=seed)
    ensemble.fit(frame.iloc[train_idx], vitals[train_idx], labels[train_idx])

    baseline = CalibratedBaseline(seed=seed)
    baseline.fit(frame.iloc[train_idx], vitals[train_idx], labels[train_idx])

    # Holdout metrics
    y_true = labels[test_idx]
    y_ens = ensemble.predict_batch(frame.iloc[test_idx].reset_index(drop=True), vitals[test_idx])
    y_base = baseline.predict_proba(frame.iloc[test_idx], vitals[test_idx])
    metrics = {
        "ensemble_auroc": float(roc_auc_score(y_true, y_ens)),
        "ensemble_auprc": float(average_precision_score(y_true, y_ens)),
        "baseline_auroc": float(roc_auc_score(y_true, y_base)),
        "baseline_auprc": float(average_precision_score(y_true, y_base)),
        "prevalence": float(labels.mean()),
        "n_patients": int(len(frame)),
        "n_test": int(len(test_idx)),
    }

    shap = ShapExplainer(agents.tabnet, frame.iloc[train_idx])
    showcase = build_showcase_cases(frame, vitals)

    return PipelineBundle(
        frame=frame,
        vitals=vitals,
        labels=labels,
        agents=agents,
        ensemble=ensemble,
        baseline=baseline,
        shap=shap,
        showcase=showcase,
        metrics=metrics,
        train_idx=train_idx,
        test_idx=test_idx,
    )
