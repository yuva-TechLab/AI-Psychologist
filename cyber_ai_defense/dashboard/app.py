import requests
import sys
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberAI Defense — Attacker Psychology Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
  }
  code, pre, .monospace {
    font-family: 'JetBrains Mono', monospace;
  }

  /* Dark terminal theme */
  .stApp { background: #0a0e1a; }

  /* Metric cards */
  .metric-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6580;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 32px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Risk level colours */
  .SAFE     { color: #22c55e; }
  .LOW      { color: #3b82f6; }
  .MEDIUM   { color: #f59e0b; }
  .HIGH     { color: #f97316; }
  .CRITICAL { color: #ef4444; }

  /* Kill-chain stage pills */
  .stage-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
  }
  .stage-active   { background: #1e3a5f; color: #60a5fa; border: 1px solid #3b82f6; }
  .stage-inactive { background: #111827; color: #374151; border: 1px solid #1f2937; }
  .stage-predicted{ background: #451a1a; color: #fca5a5; border: 1px solid #ef4444;
                    animation: pulse-border 1.5s ease-in-out infinite; }

  @keyframes pulse-border {
    0%,100% { border-color: #ef4444; box-shadow: 0 0 0 0 rgba(239,68,68,0.0); }
    50%     { border-color: #fca5a5; box-shadow: 0 0 0 4px rgba(239,68,68,0.2); }
  }

  /* Alert box */
  .alert-critical {
    background: #1c0a0a;
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fca5a5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
  }
  .alert-high {
    background: #1c110a;
    border: 1px solid #f97316;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fdba74;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
  }
  .alert-low {
    background: #0a0f1c;
    border: 1px solid #3b82f6;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    color: #93c5fd;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
  }
  .alert-medium {
    background: #1a140a;
    border: 1px solid #f59e0b;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fde68a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
  }

  /* Action box */
  .action-box {
    background: #0d1f0d;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 14px 18px;
    color: #86efac;
    font-size: 13px;
    line-height: 1.7;
  }

  /* Section headers */
  .section-header {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a6580;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 6px;
    margin-bottom: 14px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #060b16 !important;
    border-right: 1px solid #1e2d4a;
  }

  /* Plotly container background */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS & STAGE DATA
# ─────────────────────────────────────────────

KNOWN_STAGES = [
    "Reconnaissance", "Credential Access", "Exploitation",
    "Privilege Escalation", "Lateral Movement",
    "Command & Control", "Data Exfiltration", "Impact",
    "Persistence", "Discovery",
]

STAGE_BASE_SCORES = {
    "Benign": 0, "Reconnaissance": 20, "Discovery": 25,
    "Credential Access": 40, "Exploitation": 55, "Persistence": 55,
    "Privilege Escalation": 65, "Defense Evasion": 65,
    "Lateral Movement": 70, "Command & Control": 75,
    "Data Exfiltration": 80, "Impact": 80, "Unknown": 30,
}

STAGE_DEFENSES = {
    "Reconnaissance":       "Block port scanning sources. Enable IDS signatures. Rate-limit ICMP.",
    "Credential Access":    "Enable MFA immediately. Lock accounts after 5 failed attempts. Alert SOC.",
    "Exploitation":         "Patch affected services urgently. Deploy WAF rules. Activate honeypots.",
    "Privilege Escalation": "Enforce least-privilege. Monitor sudo/admin activity. Deploy UEBA alerts.",
    "Lateral Movement":     "Segment network immediately. Monitor SMB/RDP/WMI. Disable unnecessary shares.",
    "Command & Control":    "Block suspicious outbound. Monitor DNS beaconing. Isolate host.",
    "Data Exfiltration":    "Activate DLP rules. Block large outbound transfers. Isolate systems NOW.",
    "Impact":               "ISOLATE affected systems. Activate IR plan. Restore from clean backup.",
    "Persistence":          "Audit startup items, scheduled tasks, and registry run keys. Enable FIM.",
    "Discovery":            "Alert on AD enumeration commands. Restrict LDAP queries.",
}

KILL_CHAIN_ORDER = [
    "Reconnaissance", "Discovery", "Credential Access", "Exploitation",
    "Persistence", "Privilege Escalation", "Lateral Movement",
    "Command & Control", "Data Exfiltration", "Impact",
]

RISK_COLOR = {
    "SAFE": "#22c55e", "LOW": "#3b82f6",
    "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444",
}


# ─────────────────────────────────────────────
# DEMO PREDICTION ENGINE (no model files needed)
# ─────────────────────────────────────────────

TRANSITION_PROBS = {
    "Reconnaissance":       {"Credential Access": 0.44, "Exploitation": 0.36, "Discovery": 0.12, "Persistence": 0.08},
    "Credential Access":    {"Exploitation": 0.34, "Privilege Escalation": 0.32, "Lateral Movement": 0.22, "Discovery": 0.12},
    "Exploitation":         {"Privilege Escalation": 0.50, "Persistence": 0.28, "Lateral Movement": 0.14, "Reconnaissance": 0.08},
    "Privilege Escalation": {"Lateral Movement": 0.51, "Discovery": 0.28, "Command & Control": 0.14, "Impact": 0.07},
    "Lateral Movement":     {"Data Exfiltration": 0.44, "Command & Control": 0.28, "Discovery": 0.20, "Impact": 0.08},
    "Command & Control":    {"Data Exfiltration": 0.70, "Lateral Movement": 0.18, "Impact": 0.12},
    "Discovery":            {"Lateral Movement": 0.50, "Data Exfiltration": 0.30, "Impact": 0.20},
    "Persistence":          {"Privilege Escalation": 0.40, "Command & Control": 0.30, "Lateral Movement": 0.20, "Impact": 0.10},
    "Data Exfiltration":    {"Impact": 0.80, "Command & Control": 0.20},
    "Impact":               {"Reconnaissance": 0.50, "Data Exfiltration": 0.50},
}

def predict_next(sequence: list) -> tuple:
    """Weighted prediction from transition probs + sequence context."""
    if not sequence:
        return "Reconnaissance", 0.60, [("Reconnaissance", 0.60), ("Discovery", 0.40)]

    last = sequence[-1]
    nexts = dict(TRANSITION_PROBS.get(last, {"Unknown": 1.0}))

    # Slight boost to deeper stages if sequence is long
    depth_bonus = min(len(sequence) * 0.02, 0.15)
    for stage in ["Data Exfiltration", "Command & Control", "Impact"]:
        if stage in nexts:
            nexts[stage] = min(nexts[stage] + depth_bonus, 0.99)

    total = sum(nexts.values())
    nexts = {k: v / total for k, v in nexts.items()}
    sorted_n = sorted(nexts.items(), key=lambda x: -x[1])
    best, conf = sorted_n[0]
    return best, conf, sorted_n[:5]

def compute_score(sequence: list, predicted: str, confidence: float) -> tuple:
    """Return (score, risk_level)."""
    base = STAGE_BASE_SCORES.get(predicted, 30)
    conf_boost = confidence * 10
    depth = max((KILL_CHAIN_ORDER.index(s) + 1
                 for s in sequence if s in KILL_CHAIN_ORDER), default=0)
    depth_boost = (depth / len(KILL_CHAIN_ORDER)) * 10
    score = float(np.clip(base + conf_boost + depth_boost, 0, 100))
    if score >= 85:   return score, "CRITICAL"
    elif score >= 70: return score, "HIGH"
    elif score >= 50: return score, "MEDIUM"
    elif score >= 30: return score, "LOW"
    else:             return score, "SAFE"


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

def gauge_chart(score: float, risk_level: str) -> go.Figure:
    score = min(float(score), 100.0)   # ← add this line
    ...
    color = RISK_COLOR.get(risk_level, "#3b82f6")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 36, "color": color, "family": "JetBrains Mono"},
                "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0,
                     "tickcolor": "#1e2d4a", "tickfont": {"color": "#4a6580", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#0a0e1a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#0a0e1a"},
                {"range": [30, 50], "color": "#0d1525"},
                {"range": [50, 70], "color": "#0f1a2e"},
                {"range": [70, 85], "color": "#120e1a"},
                {"range": [85, 100],"color": "#150a0a"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": score},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8ba3c0"},
    )
    return fig

def topk_bar_chart(topk: list) -> go.Figure:
    stages = [s for s, _ in topk][::-1]
    probs  = [p for _, p in topk][::-1]
    colors = [RISK_COLOR.get(
        "CRITICAL" if STAGE_BASE_SCORES.get(s, 0) >= 75 else
        "HIGH"     if STAGE_BASE_SCORES.get(s, 0) >= 65 else
        "MEDIUM"   if STAGE_BASE_SCORES.get(s, 0) >= 50 else
        "LOW"      if STAGE_BASE_SCORES.get(s, 0) >= 30 else "SAFE", "#3b82f6")
        for s in stages]

    fig = go.Figure(go.Bar(
        x=probs, y=stages, orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p:.0%}" for p in probs],
        textposition="outside",
        textfont={"color": "#8ba3c0", "size": 11, "family": "JetBrains Mono"},
    ))
    fig.update_layout(
        height=220, margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1.1], showgrid=False, showticklabels=False,
                   zeroline=False),
        yaxis=dict(showgrid=False, tickfont={"color": "#8ba3c0", "size": 11,
                                              "family": "JetBrains Mono"}),
        bargap=0.35,
    )
    return fig

def score_history_chart(history: list) -> go.Figure:
    if not history:
        return go.Figure()
    scores = [h["score"] for h in history]
    stages = [h["detected"] for h in history]

    level_colors = {
        "SAFE": "#22c55e", "LOW": "#3b82f6",
        "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444",
    }
    colors = [level_colors.get(h["level"], "#3b82f6") for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))), y=scores,
        mode="lines+markers",
        line=dict(color="#1e3a5f", width=2),
        marker=dict(color=colors, size=8, line=dict(width=0)),
        hovertemplate="Step %{x}: %{y:.0f}/100<br>Stage: " +
                      "<br>".join(stages) + "<extra></extra>",
    ))
    # Risk band fills
    for y0, y1, col, alpha in [
        (0, 30, "#22c55e", 0.03), (30, 50, "#3b82f6", 0.03),
        (50, 70, "#f59e0b", 0.05), (70, 85, "#f97316", 0.07),
        (85, 100, "#ef4444", 0.08),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=alpha, line_width=0)

    fig.update_layout(
        height=200, margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont={"color": "#4a6580", "size": 10}),
        yaxis=dict(range=[0, 100], showgrid=True,
                   gridcolor="#1e2d4a", tickfont={"color": "#4a6580", "size": 10}),
    )
    return fig

def kill_chain_viz(sequence: list, predicted: str) -> str:
    """Build HTML kill-chain pill display."""
    html = '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:8px 0">'
    for i, stage in enumerate(KILL_CHAIN_ORDER):
        if stage in sequence:
            cls = "stage-active"
        elif stage == predicted:
            cls = "stage-predicted"
        else:
            cls = "stage-inactive"
        html += f'<span class="stage-pill {cls}">{stage}</span>'
        if i < len(KILL_CHAIN_ORDER) - 1:
            html += '<span style="color:#1e2d4a;font-size:14px">›</span>'
    html += "</div>"
    return html


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "sequence" not in st.session_state:
    st.session_state.sequence = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px">
      <div style="font-size:11px;letter-spacing:0.2em;color:#4a6580;text-transform:uppercase">
        CyberAI Defense
      </div>
      <div style="font-size:18px;font-weight:600;color:#e2e8f0;margin-top:4px">
        Attacker Psychology Engine
      </div>
      <div style="font-size:11px;color:#4a6580;margin-top:4px">
        MITRE ATT&CK · Predictive Defense
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-header">Build Attack Sequence</div>',
                unsafe_allow_html=True)

    stage = st.selectbox("Add next detected stage:", ["— select —"] + KNOWN_STAGES)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Stage", use_container_width=True):
            if stage != "— select —":
                st.session_state.sequence.append(stage)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sequence = []
            st.session_state.history  = []

    st.divider()

    st.markdown('<div class="section-header">Quick Load Scenarios</div>',
                unsafe_allow_html=True)

    scenarios = {
        "Early Recon":          ["Reconnaissance"],
        "Cred Brute Force":     ["Reconnaissance", "Credential Access"],
        "Post-Exploit PrivEsc": ["Reconnaissance", "Credential Access", "Exploitation"],
        "Advanced APT":         ["Reconnaissance", "Exploitation", "Persistence",
                                 "Privilege Escalation", "Lateral Movement"],
        "Critical — C2 Active": ["Reconnaissance", "Credential Access", "Exploitation",
                                  "Privilege Escalation", "Command & Control"],
    }
    for name, seq in scenarios.items():
        if st.button(name, use_container_width=True):
            st.session_state.sequence = seq.copy()
            st.session_state.history  = []

    st.divider()
    st.markdown(
        '<div style="font-size:10px;color:#2a3f55;text-align:center">'
        'Educational prototype · Defensive use only</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:flex-end;
            border-bottom:1px solid #1e2d4a;padding-bottom:14px;margin-bottom:20px">
  <div>
    <div style="font-size:11px;letter-spacing:0.2em;color:#4a6580;
                text-transform:uppercase;margin-bottom:4px">
      Threat Intelligence Dashboard
    </div>
    <div style="font-size:24px;font-weight:600;color:#e2e8f0">
      🛡️ Predictive Cyber Defense
    </div>
  </div>
  <div style="font-size:11px;color:#2a3f55;font-family:'JetBrains Mono',monospace">
    Model: Transformer · MITRE ATT&CK v14
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

sequence = st.session_state.sequence

# Run prediction
if sequence:
    response = requests.post(
        "http://127.0.0.1:8000/score",
        json={"sequence": sequence}
    )

    if response.status_code == 200:
        data = response.json()
        score = data["threat_score"]
        risk_level = data["risk_level"]

        # Keep local prediction logic
        predicted, confidence, topk = predict_next(sequence)

    else:
        st.error("Backend API error")
        score = 0
        risk_level = "SAFE"
        predicted, confidence, topk = "—", 0.0, []

    # ✅ ADD THIS (you missed defense)
    defense = STAGE_DEFENSES.get(predicted, "Escalate to SOC")

    # ✅ FIXED: history append (correct place)
    st.session_state.history.append({
        "step": len(st.session_state.history) + 1,
        "detected": sequence[-1],
        "predicted": predicted,
        "score": score,
        "level": risk_level,
        "confidence": confidence,
    })

else:
    predicted   = "—"
    confidence  = 0.0
    topk        = []
    score       = 0.0
    risk_level  = "SAFE"
    defense     = "Add attack stages in the sidebar to begin analysis."

# ── Row 1: KPI metrics ───────────────────────
c1, c2, c3, c4 = st.columns(4)

risk_color = RISK_COLOR.get(risk_level, "#3b82f6")

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Detected Stage</div>
      <div class="metric-value" style="font-size:18px;color:#60a5fa">
        {sequence[-1] if sequence else "—"}
      </div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Predicted Next</div>
      <div class="metric-value" style="font-size:18px;color:#fca5a5">
        {predicted}
      </div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Threat Score</div>
      <div class="metric-value" style="color:{risk_color}">
        {score:.0f}<span style="font-size:16px;color:#4a6580">/100</span>
      </div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Risk Level</div>
      <div class="metric-value" style="color:{risk_color};font-size:22px">
        {risk_level}
      </div>
    </div>""", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ── Row 2: Kill chain + gauge ────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-header">Kill Chain Progression</div>',
                unsafe_allow_html=True)
    if sequence:
        st.markdown(kill_chain_viz(sequence, predicted), unsafe_allow_html=True)

        # Current sequence as code block
        seq_str = " → ".join(sequence)
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#4a6580;margin-top:8px;padding:8px 12px;
                    background:#060b16;border-radius:6px;border:1px solid #1e2d4a">
          INPUT  : {seq_str}<br>
          PREDICT: <span style="color:#fca5a5">{predicted}</span>
                   &nbsp; CONF=<span style="color:#60a5fa">{confidence:.0%}</span>
                   &nbsp; SCORE=<span style="color:{risk_color}">{score:.0f}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#2a3f55;font-size:13px;padding:20px 0">'
            'Add stages in the sidebar to begin attack sequence analysis.'
            '</div>',
            unsafe_allow_html=True
        )

with col_right:
    st.markdown('<div class="section-header">Threat Score Gauge</div>',
                unsafe_allow_html=True)
    st.plotly_chart(gauge_chart(score, risk_level),
                    use_container_width=True, config={"displayModeBar": False})


# ── Row 3: Top-K predictions + alert ─────────
col_a, col_b = st.columns([2, 3])

with col_a:
    st.markdown('<div class="section-header">Next-Step Probabilities</div>',
                unsafe_allow_html=True)
    if topk:
        st.plotly_chart(topk_bar_chart(topk),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div style="color:#2a3f55;font-size:13px">No prediction yet.</div>',
                    unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-header">Security Alert & Recommended Action</div>',
                unsafe_allow_html=True)

    if sequence:
        alert_cls = f"alert-{risk_level.lower()}"
        alert_msg = (
            f"[{risk_level}] Attack progression detected: {' → '.join(sequence)}<br>"
            f"AI predicts next step: <strong>{predicted}</strong> "
            f"(confidence {confidence:.0%}, score {score:.0f}/100)"
        )
        st.markdown(f'<div class="{alert_cls}">{alert_msg}</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="action-box">⚡ <strong>Recommended Action</strong><br>{defense}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="color:#2a3f55;font-size:13px">No active alert.</div>',
                    unsafe_allow_html=True)


# ── Row 4: Score history chart ────────────────
if len(st.session_state.history) > 1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Score History — Session Timeline</div>',
                unsafe_allow_html=True)
    st.plotly_chart(score_history_chart(st.session_state.history),
                    use_container_width=True, config={"displayModeBar": False})

    # History table
    with st.expander("View full event log"):
        rows = []
        for h in st.session_state.history:
            rows.append({
                "Step"      : h["step"],
                "Detected"  : h["detected"],
                "Predicted" : h["predicted"],
                "Score"     : f"{h['score']:.0f}",
                "Risk Level": h["level"],
                "Confidence": f"{h['confidence']:.0%}",
            })
        st.table(rows)
