"""Modern clinical dashboard styling for Streamlit."""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

:root {
  --ink: #0f1c2e;
  --muted: #5b6b7c;
  --line: rgba(15, 28, 46, 0.10);
  --panel: rgba(255, 255, 255, 0.78);
  --teal: #0f766e;
  --teal-deep: #115e59;
  --coral: #c2410c;
  --amber: #b45309;
  --ok: #047857;
  --sky: #0369a1;
  --bg1: #e8eef5;
  --bg2: #f4f7fb;
  --bg3: #dceae7;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, #c7e0db 0%, transparent 55%),
    radial-gradient(900px 420px at 90% 0%, #d5e3f4 0%, transparent 50%),
    linear-gradient(180deg, var(--bg1), var(--bg2) 40%, var(--bg3));
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 1280px;
}

#MainMenu, footer, header {visibility: hidden;}

.hero {
  background: linear-gradient(135deg, rgba(15, 28, 46, 0.94), rgba(17, 94, 89, 0.88));
  border-radius: 22px;
  padding: 1.55rem 1.7rem 1.35rem;
  color: #f8fafc;
  box-shadow: 0 18px 40px rgba(15, 28, 46, 0.18);
  margin-bottom: 1.1rem;
  position: relative;
  overflow: hidden;
}

.hero::after {
  content: "";
  position: absolute;
  inset: auto -40px -60px auto;
  width: 220px;
  height: 220px;
  background: radial-gradient(circle, rgba(255,255,255,0.16), transparent 70%);
}

.hero-kicker {
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 0.72rem;
  opacity: 0.78;
  margin-bottom: 0.45rem;
}

.hero h1 {
  font-family: 'Source Serif 4', serif;
  font-size: 1.85rem;
  line-height: 1.15;
  margin: 0 0 0.45rem 0;
  font-weight: 700;
}

.hero p {
  margin: 0;
  max-width: 62ch;
  color: rgba(248, 250, 252, 0.86);
  font-size: 0.98rem;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 24px rgba(15, 28, 46, 0.05);
  margin-bottom: 0.85rem;
}

.panel h3 {
  margin: 0 0 0.35rem 0;
  font-size: 1.02rem;
  font-weight: 650;
}

.panel .subtle {
  color: var(--muted);
  font-size: 0.88rem;
  margin-bottom: 0.7rem;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}

.metric-card {
  background: rgba(255,255,255,0.9);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.85rem 0.95rem;
}

.metric-card .label {
  color: var(--muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.metric-card .value {
  font-family: 'Source Serif 4', serif;
  font-size: 1.55rem;
  font-weight: 700;
  margin-top: 0.15rem;
}

.metric-card .hint {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 0.15rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  border-radius: 999px;
  padding: 0.28rem 0.7rem;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.badge-low { background: #ecfdf5; color: var(--ok); border-color: #a7f3d0; }
.badge-moderate { background: #fffbeb; color: var(--amber); border-color: #fde68a; }
.badge-elevated { background: #fff7ed; color: var(--coral); border-color: #fdba74; }
.badge-critical { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.badge-review { background: #eff6ff; color: var(--sky); border-color: #bfdbfe; }

.agent-card {
  border-radius: 16px;
  border: 1px solid var(--line);
  background: white;
  padding: 0.9rem 1rem;
  height: 100%;
}

.agent-card .name {
  font-weight: 650;
  margin-bottom: 0.15rem;
}

.agent-card .mod {
  color: var(--muted);
  font-size: 0.8rem;
  margin-bottom: 0.55rem;
}

.debate-item {
  padding: 0.55rem 0.7rem;
  border-left: 3px solid var(--teal);
  background: rgba(15, 118, 110, 0.06);
  border-radius: 0 10px 10px 0;
  margin-bottom: 0.45rem;
  font-size: 0.9rem;
}

.cf-item {
  padding: 0.7rem 0.8rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  margin-bottom: 0.5rem;
  background: white;
}

.cf-item strong { color: var(--teal-deep); }

.section-title {
  font-family: 'Source Serif 4', serif;
  font-size: 1.25rem;
  margin: 0.4rem 0 0.2rem;
}

.footnote {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 1rem;
}

div[data-testid="stTabs"] button {
  font-weight: 600;
}

@media (max-width: 900px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero h1 { font-size: 1.45rem; }
}
"""


def risk_badge_class(tier: str) -> str:
    return {
        "Low": "badge-low",
        "Moderate": "badge-moderate",
        "Elevated": "badge-elevated",
        "Critical": "badge-critical",
    }.get(tier, "badge-moderate")
