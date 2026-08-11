"""EECRPS-XAI — Admin dashboard + patient risk + faculty live demo."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def html(markup: str) -> None:
    """Render HTML in Streamlit without markdown treating indents as code blocks."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)

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
    page_title="EECRPS-XAI | Admin Clinical Dashboard",
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


def get_cohort_preds(bundle):
    if "cohort_preds" not in st.session_state:
        with st.spinner("Scoring ICU cohort for admin stats…"):
            st.session_state.cohort_preds = bundle.ensemble.predict_batch(
                bundle.frame, bundle.vitals
            )
    return st.session_state.cohort_preds


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
    colors = ["#0f766e", "#0369a1", "#0e7490", "#b45309", "#047857", "#c2410c"]
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
    return plotly_layout(fig, height=360)


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
    return plotly_layout(fig, height=300)


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
    )
    return plotly_layout(fig, height=320)


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
    return plotly_layout(fig, height=360)


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
    return plotly_layout(fig, height=360)


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
    return plotly_layout(fig, height=280)


def risk_distribution_figure(preds: np.ndarray) -> go.Figure:
    tiers = [risk_tier(p) for p in preds]
    order = ["Low", "Moderate", "Elevated", "Critical"]
    colors = {
        "Low": "#047857",
        "Moderate": "#b45309",
        "Elevated": "#c2410c",
        "Critical": "#b91c1c",
    }
    counts = pd.Series(tiers).value_counts().reindex(order).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=order,
            y=counts.values,
            marker_color=[colors[o] for o in order],
            text=[int(v) for v in counts.values],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="Patients", xaxis_title="Risk tier")
    return plotly_layout(fig, height=320)


def sofa_risk_figure(frame: pd.DataFrame, preds: np.ndarray) -> go.Figure:
    fig = go.Figure(
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
    fig.update_layout(xaxis_title="SOFA", yaxis_title="Predicted risk (%)")
    return plotly_layout(fig, height=320)


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
    return plotly_layout(fig, height=320)


def age_risk_figure(frame: pd.DataFrame, preds: np.ndarray) -> go.Figure:
    bins = pd.cut(frame["age"], bins=[18, 40, 55, 70, 85, 100], right=False)
    df = pd.DataFrame({"age_bin": bins.astype(str), "risk": preds})
    summary = df.groupby("age_bin", observed=False)["risk"].mean().reset_index()
    fig = go.Figure(
        go.Bar(
            x=summary["age_bin"],
            y=summary["risk"] * 100,
            marker_color=PALETTE["sky"],
            text=[f"{v:.0f}%" for v in summary["risk"] * 100],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Age group", yaxis_title="Mean predicted risk (%)")
    return plotly_layout(fig, height=300)


def render_hero():
    html(
        """
        <div class="hero">
          <div class="hero-kicker">EECRPS-XAI · Clinical Admin Console · Batch 18</div>
          <h1>ICU Risk Command Center</h1>
          <p>
            Admin stats for the ICU cohort, patient-level risk prediction with explainability,
            and a one-click Live Demo tab for faculty walkthrough.
          </p>
        </div>
        """
    )


def kpi_cards(items: list[tuple[str, str, str]]):
    cards = "".join(
        (
            '<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="hint">{hint}</div>'
            "</div>"
        )
        for label, value, hint in items
    )
    html(f'<div class="metric-grid">{cards}</div>')


def predict_patient(bundle, row, vitals, mc_samples: int = 40):
    agents = bundle.agents.predict_one(row, vitals)
    ens = bundle.ensemble.predict_one(row, vitals)
    unc = mc_dropout_estimate(bundle.ensemble, row, vitals, n_samples=mc_samples)
    attn = temporal_attention(vitals, ens.probability)
    shap_bundle = bundle.shap.explain(row, bundle.frame)
    counterfactuals = generate_counterfactuals(
        bundle.ensemble, row, vitals, ens.probability
    )
    return agents, ens, unc, attn, shap_bundle, counterfactuals


def render_admin_dashboard(bundle, preds: np.ndarray):
    frame = bundle.frame.copy()
    frame["pred_risk"] = preds
    frame["risk_tier"] = [risk_tier(p) for p in preds]
    critical = int((frame["risk_tier"] == "Critical").sum())
    elevated = int((frame["risk_tier"] == "Elevated").sum())
    review_band = int(((preds >= 0.35) & (preds <= 0.55)).sum())
    high_risk = frame.sort_values("pred_risk", ascending=False).head(8)

    kpi_cards(
        [
            ("ICU patients", f"{bundle.metrics['n_patients']}", f"Holdout n={bundle.metrics['n_test']}"),
            ("Critical alerts", f"{critical}", f"Elevated also: {elevated}"),
            ("Needs review", f"{review_band}", "Borderline 35–55% band"),
            ("Model AUROC", f"{bundle.metrics['ensemble_auroc']:.3f}", f"AUPRC {bundle.metrics['ensemble_auprc']:.3f}"),
        ]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-title">Risk tier census</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_distribution_figure(preds), use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">SOFA vs predicted risk</div>', unsafe_allow_html=True)
        st.plotly_chart(sofa_risk_figure(frame, preds), use_container_width=True)
    with c3:
        st.markdown('<div class="section-title">Model discrimination</div>', unsafe_allow_html=True)
        st.plotly_chart(metrics_bars(bundle.metrics), use_container_width=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="section-title">High-risk watchlist</div>', unsafe_allow_html=True)
        show = high_risk[
            ["patient_id", "age", "sofa", "lactate", "pred_risk", "risk_tier", "mortality"]
        ].copy()
        show["pred_risk"] = show["pred_risk"].map(lambda x: f"{x:.0%}")
        show = show.rename(
            columns={
                "patient_id": "Patient",
                "age": "Age",
                "sofa": "SOFA",
                "lactate": "Lactate",
                "pred_risk": "Risk",
                "risk_tier": "Tier",
                "mortality": "Label",
            }
        )
        st.dataframe(show, hide_index=True, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Mean risk by age</div>', unsafe_allow_html=True)
        st.plotly_chart(age_risk_figure(frame, preds), use_container_width=True)
        html(
            f"""
            <div class="panel">
              <h3>Admin snapshot</h3>
              <div class="subtle">Synthetic ICU cohort · offline faculty prototype</div>
              <ul>
                <li>Event prevalence: <b>{bundle.metrics['prevalence']:.1%}</b></li>
                <li>Mean predicted risk: <b>{preds.mean():.0%}</b></li>
                <li>Max predicted risk: <b>{preds.max():.0%}</b></li>
                <li>Agents online: <b>TabNet · Bi-LSTM · CNN-1D</b></li>
              </ul>
            </div>
            """
        )


def render_patient_prediction(bundle, row, vitals, patient_id: str, mc_samples: int):
    agents, ens, unc, attn, shap_bundle, counterfactuals = predict_patient(
        bundle, row, vitals, mc_samples=mc_samples
    )
    tier = risk_tier(ens.probability)
    badge = risk_badge_class(tier)

    kpi_cards(
        [
            ("Patient", patient_id, f"{int(row['age'])}y · SOFA {int(row['sofa'])}"),
            ("Predicted risk", f"{ens.probability:.0%}", f'<span class="badge {badge}">{tier}</span>'),
            ("Uncertainty σ", f"{unc.std:.3f}", f"CI {unc.ci_low:.0%}–{unc.ci_high:.0%}"),
            ("Lactate / HR", f"{row['lactate']:.1f}", f"HR now {vitals[-1, 0]:.0f}"),
        ]
    )

    if unc.needs_review:
        html(
            f'<div style="margin:0.5rem 0 0.9rem;"><span class="badge badge-review">Clinician review</span> '
            f'<span style="color:#5b6b7c;">{unc.review_reason}</span></div>'
        )

    sub1, sub2, sub3, sub4 = st.tabs(
        ["Risk result", "Vitals & context", "Agent consensus", "Explainability"]
    )

    with sub1:
        c1, c2 = st.columns([1.05, 1.2])
        with c1:
            st.plotly_chart(
                gauge_figure(ens.probability, unc.ci_low, unc.ci_high),
                use_container_width=True,
            )
            st.plotly_chart(uncertainty_hist(unc.samples, unc.mean), use_container_width=True)
        with c2:
            st.markdown("#### Collaborative reasoning")
            for note in ens.debate_notes:
                html(f'<div class="debate-item">{note}</div>')
            st.markdown("#### Counterfactual actions")
            if not counterfactuals:
                st.caption("No strong risk-reducing nudges for this case.")
            for cf in counterfactuals[:3]:
                html(
                    f"""
                    <div class="cf-item">
                      <strong>{cf.feature}</strong>
                      {cf.current:.1f} → {cf.proposed:.1f} {cf.unit_hint}
                      · risk {ens.probability:.0%} → <strong>{cf.new_risk:.0%}</strong>
                    </div>
                    """
                )

    with sub2:
        left, right = st.columns([1.4, 1])
        with left:
            st.plotly_chart(vitals_figure(vitals, attn.temporal), use_container_width=True)
            st.plotly_chart(attention_heatmap(attn.heatmap), use_container_width=True)
        with right:
            dem = pd.DataFrame(
                {"Feature": DEMOGRAPHIC_COLS, "Value": [row[c] for c in DEMOGRAPHIC_COLS]}
            )
            labs = pd.DataFrame({"Lab": LAB_COLS, "Value": [row[c] for c in LAB_COLS]})
            st.dataframe(dem, hide_index=True, use_container_width=True)
            st.dataframe(labs, hide_index=True, use_container_width=True)

    with sub3:
        cols = st.columns(3)
        for col, agent in zip(cols, agents):
            with col:
                html(
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
                    """
                )
        st.plotly_chart(agent_bar(ens.agent_probs, ens.weights), use_container_width=True)

    with sub4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(shap_local_figure(shap_bundle.local), use_container_width=True)
        with c2:
            st.plotly_chart(shap_global_figure(shap_bundle.global_mean_abs), use_container_width=True)
        for cf in counterfactuals:
            html(
                f"""
                <div class="cf-item">
                  <strong>{cf.feature}</strong> — {cf.narrative}
                </div>
                """
            )


def render_live_demo(bundle, mc_samples: int):
    html(
        """
        <div class="panel">
          <h3>Faculty Live Demo</h3>
          <div class="subtle">
            Guided walkthrough — pick a showcase patient and present risk, agents, and explanations in one screen.
          </div>
        </div>
        """
    )

    labels = {
        f"{c['patient_id']} · {c['risk_tier']} — {c['summary']}": c for c in bundle.showcase
    }
    # Prefer a critical/elevated case as default for impactful demo
    default_idx = 0
    for i, c in enumerate(bundle.showcase):
        if c["risk_tier"] in {"Critical", "Elevated"}:
            default_idx = i
            break

    choice = st.selectbox(
        "Demo patient",
        list(labels.keys()),
        index=default_idx,
        help="Curated cases spanning Low → Critical risk",
    )
    case = labels[choice]
    idx = case["row_index"]
    row = bundle.frame.iloc[idx]
    vitals = bundle.vitals[idx]

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        run = st.button("▶ Run live prediction", type="primary", use_container_width=True)
    with col_b:
        st.caption("Shows end-to-end: multimodal → agents → consensus → XAI")
    with col_c:
        st.caption("Use this tab during viva / first review")

    if run or st.session_state.get("demo_autorun"):
        st.session_state["demo_autorun"] = True
        agents, ens, unc, attn, shap_bundle, counterfactuals = predict_patient(
            bundle, row, vitals, mc_samples=mc_samples
        )
        tier = risk_tier(ens.probability)
        badge = risk_badge_class(tier)

        st.markdown("### 1 · Patient story")
        html(
            f"""
            <div class="panel">
              <b>{case['patient_id']}</b> — {case['summary']}<br/>
              Ground-truth label (demo): <b>{'Mortality event' if case['label'] else 'Survived'}</b>
              · Predicted tier: <span class="badge {badge}">{tier}</span>
            </div>
            """
        )

        st.markdown("### 2 · Multimodal risk prediction")
        kpi_cards(
            [
                ("Ensemble risk", f"{ens.probability:.0%}", f"CI {unc.ci_low:.0%}–{unc.ci_high:.0%}"),
                ("TabNet", f"{ens.agent_probs['TabNet']:.0%}", f"Weight {ens.weights['TabNet']:.0%}"),
                ("Bi-LSTM", f"{ens.agent_probs['Bi-LSTM']:.0%}", f"Weight {ens.weights['Bi-LSTM']:.0%}"),
                ("CNN-1D", f"{ens.agent_probs['CNN-1D']:.0%}", f"Weight {ens.weights['CNN-1D']:.0%}"),
            ]
        )

        g1, g2 = st.columns([1, 1.2])
        with g1:
            st.plotly_chart(
                gauge_figure(ens.probability, unc.ci_low, unc.ci_high),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(agent_bar(ens.agent_probs, ens.weights), use_container_width=True)

        st.markdown("### 3 · Why this prediction? (Explainability)")
        x1, x2 = st.columns(2)
        with x1:
            st.plotly_chart(shap_local_figure(shap_bundle.local), use_container_width=True)
        with x2:
            st.plotly_chart(vitals_figure(vitals, attn.temporal), use_container_width=True)

        st.markdown("### 4 · Agent debate + what-if actions")
        d1, d2 = st.columns([1.1, 1])
        with d1:
            for note in ens.debate_notes:
                html(f'<div class="debate-item">{note}</div>')
            if unc.needs_review:
                st.warning(unc.review_reason)
            else:
                st.success(unc.review_reason)
        with d2:
            if not counterfactuals:
                st.info("Stable case — no strong counterfactual levers.")
            for cf in counterfactuals[:3]:
                html(
                    f"""
                    <div class="cf-item">
                      <strong>{cf.feature}</strong><br/>{cf.narrative}
                    </div>
                    """
                )

        with st.expander("Talking points for faculty"):
            st.markdown(
                """
                1. **Admin dashboard** shows ICU census & model quality.  
                2. **Patient prediction** is how a clinician would query one case.  
                3. **This Live Demo** stitches multimodal agents + adaptive ensemble + SHAP/attention/counterfactuals + uncertainty.  
                4. Synthetic data today; same schema plugs into MIMIC-III / eICU next.
                """
            )
    else:
        st.info("Click **Run live prediction** to start the faculty demo for the selected patient.")


def main():
    render_hero()
    bundle = load_bundle()
    preds = get_cohort_preds(bundle)

    with st.sidebar:
        st.markdown("### Navigation help")
        st.markdown(
            """
            **Admin Dashboard** — cohort stats & alerts  
            **Patient Prediction** — risk for one patient  
            **Live Demo** — faculty walkthrough
            """
        )
        st.markdown("---")
        st.markdown("### Prediction settings")
        mc_samples = st.slider("MC uncertainty samples", 20, 80, 40, 5)
        st.caption("Python 3.11 · Streamlit · synthetic ICU cohort")

        st.markdown("---")
        st.markdown("### Patient for prediction tab")
        all_ids = bundle.frame["patient_id"].tolist()
        # Default to a mid/high risk showcase
        default_pid = bundle.showcase[-2]["patient_id"]
        patient_id = st.selectbox(
            "Select patient",
            all_ids,
            index=all_ids.index(default_pid),
        )
        idx = int(bundle.frame.index[bundle.frame["patient_id"] == patient_id][0])
        row = bundle.frame.iloc[idx]
        vitals = bundle.vitals[idx]

    tab_admin, tab_patient, tab_demo = st.tabs(
        ["Admin Dashboard", "Patient Prediction", "Live Demo"]
    )

    with tab_admin:
        render_admin_dashboard(bundle, preds)

    with tab_patient:
        render_patient_prediction(bundle, row, vitals, patient_id, mc_samples)

    with tab_demo:
        render_live_demo(bundle, mc_samples)

    html(
        '<div class="footnote">EECRPS-XAI · Dept. of AI &amp; DS · Batch XVIII · '
        "Supervisor: Mr. P. Ramprakash · Phase-I review prototype</div>"
    )


if __name__ == "__main__":
    main()
