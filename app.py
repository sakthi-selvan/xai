"""EECRPS-XAI — Streamlit faculty demo dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import (  # noqa: E402
    DEMOGRAPHIC_COLS,
    LAB_COLS,
    VITAL_NAMES,
    risk_tier,
)
from src.explain.attention import temporal_attention  # noqa: E402
from src.explain.counterfactual import generate_counterfactuals  # noqa: E402
from src.models.uncertainty import mc_dropout_estimate  # noqa: E402
from src.pipeline import build_pipeline  # noqa: E402
from src.ui.styles import CUSTOM_CSS, risk_badge_class  # noqa: E402

st.set_page_config(
    page_title="EECRPS-XAI | Clinical Risk Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

PALETTE = {
    "teal": "#0f766e",
    "ink": "#0f1c2e",
    "coral": "#c2410c",
    "amber": "#b45309",
    "sky": "#0369a1",
    "ok": "#047857",
    "muted": "#5b6b7c",
    "grid": "rgba(15,28,46,0.08)",
}


def plotly_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["ink"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig


@st.cache_resource(show_spinner="Training multimodal ensemble on synthetic ICU cohort…")
def load_bundle():
    return build_pipeline(seed=42)


def gauge_figure(prob: float, ci_low: float, ci_high: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 42, "family": "Source Serif 4, serif"}},
            title={"text": "Mortality risk", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": PALETTE["teal"]},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "#ecfdf5"},
                    {"range": [20, 40], "color": "#fffbeb"},
                    {"range": [40, 70], "color": "#fff7ed"},
                    {"range": [70, 100], "color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["coral"], "width": 3},
                    "thickness": 0.8,
                    "value": ci_high * 100,
                },
            },
        )
    )
    fig.add_annotation(
        text=f"90% CI {ci_low:.0%} – {ci_high:.0%}",
        x=0.5,
        y=0.0,
        showarrow=False,
        font=dict(size=12, color=PALETTE["muted"]),
    )
    return plotly_layout(fig, height=300)


def vitals_figure(vitals: np.ndarray, attention: np.ndarray) -> go.Figure:
    hours = np.arange(1, vitals.shape[0] + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#0f766e", "#0369a1", "#7c3aed", "#b45309", "#047857", "#c2410c"]
    for i, name in enumerate(VITAL_NAMES):
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=vitals[:, i],
                name=name.upper(),
                mode="lines",
                line=dict(width=2.2, color=colors[i % len(colors)]),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=attention,
            name="Temporal attention",
            marker=dict(color="rgba(15,118,110,0.22)"),
            opacity=0.9,
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Vital value", secondary_y=False)
    fig.update_yaxes(title_text="Attention", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Hour")
    return plotly_layout(fig, height=380)


def attention_heatmap(heatmap: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap,
            x=[f"H{i}" for i in range(1, heatmap.shape[1] + 1)],
            y=[n.upper() for n in VITAL_NAMES],
            colorscale=[
                [0.0, "#f8fafc"],
                [0.35, "#99f6e4"],
                [0.7, "#0d9488"],
                [1.0, "#115e59"],
            ],
            colorbar=dict(title="Salience"),
        )
    )
    return plotly_layout(fig, height=320)


def agent_bar(agent_probs: dict[str, float], weights: dict[str, float]) -> go.Figure:
    names = list(agent_probs.keys())
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Risk probability",
            x=names,
            y=[agent_probs[n] * 100 for n in names],
            marker_color=PALETTE["teal"],
            text=[f"{agent_probs[n]:.0%}" for n in names],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Adaptive weight",
            x=names,
            y=[weights[n] * 100 for n in names],
            mode="lines+markers",
            marker=dict(size=10, color=PALETTE["coral"]),
            line=dict(color=PALETTE["coral"], width=2.5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Agent risk (%)", range=[0, 100]),
        yaxis2=dict(title="Ensemble weight (%)", overlaying="y", side="right", range=[0, 100]),
        barmode="group",
    )
    return plotly_layout(fig, height=340)


def shap_local_figure(local_df: pd.DataFrame) -> go.Figure:
    top = local_df.head(10).iloc[::-1]
    colors = [PALETTE["coral"] if v > 0 else PALETTE["teal"] for v in top["shap"]]
    fig = go.Figure(
        go.Bar(
            x=top["shap"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in top["shap"]],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="SHAP value (→ higher risk)", yaxis_title="")
    return plotly_layout(fig, height=380)


def shap_global_figure(global_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=global_df["mean_abs_shap"],
            y=global_df["feature"],
            orientation="h",
            marker_color=PALETTE["sky"],
        )
    )
    fig.update_layout(xaxis_title="Mean |SHAP|", yaxis_title="")
    return plotly_layout(fig, height=380)


def uncertainty_hist(samples: np.ndarray, mean: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=samples * 100,
            nbinsx=18,
            marker_color="rgba(15,118,110,0.55)",
            name="MC samples",
        )
    )
    fig.add_vline(x=mean * 100, line_width=2.5, line_dash="dash", line_color=PALETTE["coral"])
    fig.update_layout(xaxis_title="Predicted risk (%)", yaxis_title="Count")
    return plotly_layout(fig, height=300)


def cohort_scatter(frame: pd.DataFrame, preds: np.ndarray, selected_id: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["sofa"],
            y=preds * 100,
            mode="markers",
            marker=dict(
                size=8,
                color=frame["lactate"],
                colorscale="Teal",
                showscale=True,
                colorbar=dict(title="Lactate"),
                opacity=0.75,
            ),
            text=frame["patient_id"],
            name="Cohort",
        )
    )
    mask = frame["patient_id"] == selected_id
    fig.add_trace(
        go.Scatter(
            x=frame.loc[mask, "sofa"],
            y=preds[mask.values] * 100,
            mode="markers",
            marker=dict(size=16, color=PALETTE["coral"], symbol="star"),
            name="Selected",
        )
    )
    fig.update_layout(xaxis_title="SOFA", yaxis_title="Ensemble risk (%)")
    return plotly_layout(fig, height=360)


def metrics_bars(metrics: dict[str, float]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                name="EECRPS Ensemble",
                x=["AUROC", "AUPRC"],
                y=[metrics["ensemble_auroc"], metrics["ensemble_auprc"]],
                marker_color=PALETTE["teal"],
                text=[
                    f"{metrics['ensemble_auroc']:.3f}",
                    f"{metrics['ensemble_auprc']:.3f}",
                ],
                textposition="outside",
            ),
            go.Bar(
                name="LogReg baseline",
                x=["AUROC", "AUPRC"],
                y=[metrics["baseline_auroc"], metrics["baseline_auprc"]],
                marker_color=PALETTE["amber"],
                text=[
                    f"{metrics['baseline_auroc']:.3f}",
                    f"{metrics['baseline_auprc']:.3f}",
                ],
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05], title="Score"))
    return plotly_layout(fig, height=340)


def render_hero():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Project Work Phase I · Batch 18 · Faculty Live Demo</div>
          <h1>EECRPS-XAI</h1>
          <p>
            Hierarchical Explainable Ensemble Clinical Risk Prediction System —
            multimodal agents (TabNet · Bi-LSTM · CNN-1D), adaptive consensus,
            SHAP / temporal attention / counterfactuals, and uncertainty-aware triage.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    render_hero()
    bundle = load_bundle()

    with st.sidebar:
        st.markdown("### Patient selector")
        st.caption("Synthetic MIMIC/eICU-like cohort for offline faculty demo.")
        case_labels = {
            f"{c['patient_id']} · {c['risk_tier']}": c for c in bundle.showcase
        }
        choice = st.selectbox("Showcase cases", list(case_labels.keys()))
        case = case_labels[choice]

        st.markdown("---")
        st.markdown("### Or pick from cohort")
        all_ids = bundle.frame["patient_id"].tolist()
        custom_id = st.selectbox("All patients", all_ids, index=all_ids.index(case["patient_id"]))
        use_custom = st.toggle("Use cohort patient", value=False)

        if use_custom:
            idx = int(bundle.frame.index[bundle.frame["patient_id"] == custom_id][0])
            row = bundle.frame.iloc[idx]
            vitals = bundle.vitals[idx]
            patient_id = custom_id
        else:
            idx = case["row_index"]
            row = bundle.frame.iloc[idx]
            vitals = bundle.vitals[idx]
            patient_id = case["patient_id"]

        st.markdown("---")
        st.markdown("### Demo controls")
        mc_samples = st.slider("MC uncertainty samples", 20, 80, 40, 5)
        st.caption("Python 3.11 · Streamlit prototype · open synthetic data")

    # Inference
    agents = bundle.agents.predict_one(row, vitals)
    ens = bundle.ensemble.predict_one(row, vitals)
    unc = mc_dropout_estimate(bundle.ensemble, row, vitals, n_samples=mc_samples)
    attn = temporal_attention(vitals, ens.probability)
    shap_bundle = bundle.shap.explain(row, bundle.frame)
    counterfactuals = generate_counterfactuals(
        bundle.ensemble, row, vitals, ens.probability
    )
    tier = risk_tier(ens.probability)
    badge = risk_badge_class(tier)

    # Cohort predictions cached lightly
    if "cohort_preds" not in st.session_state:
        st.session_state.cohort_preds = bundle.ensemble.predict_batch(
            bundle.frame, bundle.vitals
        )
    cohort_preds = st.session_state.cohort_preds

    # KPI row
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="label">Patient</div>
            <div class="value" style="font-size:1.25rem;">{patient_id}</div>
            <div class="hint">{int(row['age'])}y · SOFA {int(row['sofa'])} · Lactate {row['lactate']:.1f}</div>
          </div>
          <div class="metric-card">
            <div class="label">Ensemble risk</div>
            <div class="value">{ens.probability:.0%}</div>
            <div class="hint"><span class="badge {badge}">{tier}</span></div>
          </div>
          <div class="metric-card">
            <div class="label">Uncertainty (σ)</div>
            <div class="value">{unc.std:.3f}</div>
            <div class="hint">CI {unc.ci_low:.0%}–{unc.ci_high:.0%}</div>
          </div>
          <div class="metric-card">
            <div class="label">Holdout AUROC</div>
            <div class="value">{bundle.metrics['ensemble_auroc']:.3f}</div>
            <div class="hint">AUPRC {bundle.metrics['ensemble_auprc']:.3f} · n={bundle.metrics['n_patients']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if unc.needs_review:
        st.markdown(
            f'<div style="margin:0.8rem 0;"><span class="badge badge-review">Clinician review</span> '
            f'<span style="color:#5b6b7c; font-size:0.92rem;">{unc.review_reason}</span></div>',
            unsafe_allow_html=True,
        )

    tab_overview, tab_multi, tab_agents, tab_xai, tab_perf = st.tabs(
        [
            "Overview",
            "Multimodal signals",
            "Agent consensus",
            "Explainability",
            "Validation",
        ]
    )

    with tab_overview:
        c1, c2 = st.columns([1.05, 1.35])
        with c1:
            st.markdown('<div class="section-title">Risk & confidence</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_figure(ens.probability, unc.ci_low, unc.ci_high), use_container_width=True)
            st.plotly_chart(uncertainty_hist(unc.samples, unc.mean), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">Where this patient sits</div>', unsafe_allow_html=True)
            st.plotly_chart(
                cohort_scatter(bundle.frame, cohort_preds, patient_id),
                use_container_width=True,
            )
            st.markdown("#### Collaborative reasoning")
            for note in ens.debate_notes:
                st.markdown(f'<div class="debate-item">{note}</div>', unsafe_allow_html=True)

    with tab_multi:
        left, right = st.columns([1.4, 1])
        with left:
            st.markdown('<div class="section-title">24h vitals + temporal attention</div>', unsafe_allow_html=True)
            st.plotly_chart(vitals_figure(vitals, attn.temporal), use_container_width=True)
            st.plotly_chart(attention_heatmap(attn.heatmap), use_container_width=True)
        with right:
            st.markdown('<div class="section-title">Static clinical context</div>', unsafe_allow_html=True)
            dem = pd.DataFrame(
                {
                    "Feature": DEMOGRAPHIC_COLS,
                    "Value": [row[c] for c in DEMOGRAPHIC_COLS],
                }
            )
            labs = pd.DataFrame(
                {"Lab": LAB_COLS, "Value": [row[c] for c in LAB_COLS]}
            )
            st.dataframe(dem, hide_index=True, use_container_width=True)
            st.dataframe(labs, hide_index=True, use_container_width=True)
            vital_imp = pd.DataFrame(
                {
                    "Vital": list(attn.vital_importance.keys()),
                    "Attention share": list(attn.vital_importance.values()),
                }
            ).sort_values("Attention share", ascending=False)
            st.markdown("##### Channel importance")
            st.dataframe(
                vital_imp.style.format({"Attention share": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )

    with tab_agents:
        st.markdown(
            '<div class="section-title">Autonomous agents → adaptive consensus</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for col, agent in zip(cols, agents):
            with col:
                st.markdown(
                    f"""
                    <div class="agent-card">
                      <div class="name">{agent.name}</div>
                      <div class="mod">{agent.modality}</div>
                      <div style="font-family:Source Serif 4,serif; font-size:1.8rem; font-weight:700;">
                        {agent.probability:.0%}
                      </div>
                      <div style="color:#5b6b7c; font-size:0.82rem; margin-bottom:0.45rem;">
                        Confidence {agent.confidence:.0%} · weight {ens.weights[agent.short]:.0%}
                      </div>
                      <div style="font-size:0.9rem;">{agent.rationale}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.plotly_chart(agent_bar(ens.agent_probs, ens.weights), use_container_width=True)
        st.info(
            "The meta-learner blends agent probabilities with patient acuity context; "
            "cross-modal attention reweights specialists per case instead of fixed ensemble averages."
        )

    with tab_xai:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Local SHAP (this patient)</div>', unsafe_allow_html=True)
            st.plotly_chart(shap_local_figure(shap_bundle.local), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">Global SHAP (cohort)</div>', unsafe_allow_html=True)
            st.plotly_chart(shap_global_figure(shap_bundle.global_mean_abs), use_container_width=True)

        st.markdown('<div class="section-title">Counterfactual recommendations</div>', unsafe_allow_html=True)
        st.caption("Illustrative what-if nudges for demo explainability — not clinical advice.")
        if not counterfactuals:
            st.write("No beneficial counterfactuals found for this case.")
        for cf in counterfactuals:
            st.markdown(
                f"""
                <div class="cf-item">
                  <strong>{cf.feature}</strong>
                  {cf.current:.1f} → {cf.proposed:.1f} {cf.unit_hint}
                  · risk {ens.probability:.0%} → <strong>{cf.new_risk:.0%}</strong>
                  (−{cf.risk_drop:.1%})
                  <div style="color:#5b6b7c; margin-top:0.25rem;">{cf.narrative}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_perf:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown('<div class="section-title">Holdout discrimination</div>', unsafe_allow_html=True)
            st.plotly_chart(metrics_bars(bundle.metrics), use_container_width=True)
            st.markdown(
                f"""
                <div class="panel">
                  <h3>Prototype evaluation snapshot</h3>
                  <div class="subtle">Synthetic cohort proxy for MIMIC-III / eICU workflow</div>
                  <ul>
                    <li>Patients: <b>{bundle.metrics['n_patients']}</b> · Test: <b>{bundle.metrics['n_test']}</b></li>
                    <li>Event prevalence: <b>{bundle.metrics['prevalence']:.1%}</b></li>
                    <li>Ensemble AUROC / AUPRC: <b>{bundle.metrics['ensemble_auroc']:.3f}</b> /
                        <b>{bundle.metrics['ensemble_auprc']:.3f}</b></li>
                    <li>Baseline AUROC / AUPRC: <b>{bundle.metrics['baseline_auroc']:.3f}</b> /
                        <b>{bundle.metrics['baseline_auprc']:.3f}</b></li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown('<div class="section-title">Architecture in this demo</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="panel">
                  <ol>
                    <li><b>Multimodal inputs</b> — demographics, labs, 24h vitals</li>
                    <li><b>Specialist agents</b> — TabNet / Bi-LSTM / CNN-1D proxies</li>
                    <li><b>Cross-modal attention</b> — patient-specific agent weights</li>
                    <li><b>Meta-learner consensus</b> — collaborative risk adjudication</li>
                    <li><b>Explainability</b> — SHAP, temporal attention, counterfactuals</li>
                    <li><b>Uncertainty</b> — Monte Carlo confidence intervals + review flag</li>
                  </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.warning(
                "Faculty note: this build uses a legal synthetic cohort so the demo runs offline. "
                "Credentialed MIMIC-III / eICU wiring is the next validation step."
            )

    st.markdown(
        '<div class="footnote">EECRPS-XAI · Dept. of AI &amp; DS · Batch XVIII · '
        "Supervisor: Mr. P. Ramprakash · Demo prototype for Phase-I review</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
