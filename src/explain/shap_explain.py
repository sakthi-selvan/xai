"""Global / local SHAP explanations over tabular clinical features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.data.synthetic import DEMOGRAPHIC_COLS, FEATURE_LABELS, LAB_COLS


@dataclass
class ShapBundle:
    local: pd.DataFrame
    global_mean_abs: pd.DataFrame
    base_value: float
    feature_names: list[str]


def _pretty(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


class ShapExplainer:
    def __init__(self, model: Pipeline, background: pd.DataFrame, max_bg: int = 80):
        self.feature_names = DEMOGRAPHIC_COLS + LAB_COLS
        bg = background[self.feature_names].sample(
            n=min(max_bg, len(background)), random_state=42
        )
        self.background = bg
        # TreeExplainer on the GBDT step when available; else KernelExplainer
        clf = model.named_steps["model"]
        self._scaler = model.named_steps["scaler"]
        self._clf = clf
        try:
            self.explainer = shap.TreeExplainer(clf)
            self.mode = "tree"
        except Exception:
            scaled_bg = self._scaler.transform(bg.to_numpy())
            self.explainer = shap.KernelExplainer(
                lambda x: clf.predict_proba(x)[:, 1], scaled_bg
            )
            self.mode = "kernel"

    def explain(self, row: pd.Series, cohort: pd.DataFrame) -> ShapBundle:
        x = row[self.feature_names].to_numpy(dtype=float).reshape(1, -1)
        x_scaled = self._scaler.transform(x)

        if self.mode == "tree":
            values = self.explainer.shap_values(x_scaled)
            if isinstance(values, list):
                values = values[1]
            local_vals = np.asarray(values).reshape(-1)
            base = self.explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = float(np.asarray(base).reshape(-1)[-1])
            else:
                base = float(base)
        else:
            values = self.explainer.shap_values(x_scaled, nsamples=100)
            local_vals = np.asarray(values).reshape(-1)
            base = float(self.explainer.expected_value)

        local = pd.DataFrame(
            {
                "feature": [_pretty(f) for f in self.feature_names],
                "feature_key": self.feature_names,
                "value": x.reshape(-1),
                "shap": local_vals,
            }
        ).sort_values("shap", key=np.abs, ascending=False)

        # Global mean |SHAP| on a cohort sample
        sample = cohort[self.feature_names].sample(
            n=min(120, len(cohort)), random_state=0
        )
        xs = self._scaler.transform(sample.to_numpy())
        if self.mode == "tree":
            gvals = self.explainer.shap_values(xs)
            if isinstance(gvals, list):
                gvals = gvals[1]
            gvals = np.asarray(gvals)
        else:
            gvals = np.asarray(self.explainer.shap_values(xs, nsamples=50))

        global_df = (
            pd.DataFrame(
                {
                    "feature": [_pretty(f) for f in self.feature_names],
                    "mean_abs_shap": np.abs(gvals).mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=True)
            .reset_index(drop=True)
        )

        return ShapBundle(
            local=local.reset_index(drop=True),
            global_mean_abs=global_df,
            base_value=base,
            feature_names=self.feature_names,
        )
