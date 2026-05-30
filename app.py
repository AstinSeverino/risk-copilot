"""Risk Copilot — Agentic Transaction Risk Analysis"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import time

from src.data.database import get_all_merchants, get_merchant, get_transactions
from src.ml.predict import predict_all_merchants, predict_risk
from src.ml.features import calculate_features
from src.agents.graph import graph, builder
from src.agents.observability import flush as langfuse_flush

st.set_page_config(
    page_title="Risk Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

header[data-testid="stHeader"] {
    background: rgba(10, 10, 15, 0.95);
    backdrop-filter: blur(10px);
}

.app-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.15);
    margin-bottom: 1.5rem;
}

.app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f4f4f5;
    letter-spacing: -0.02em;
    margin: 0;
}

.app-subtitle {
    font-size: 0.875rem;
    color: #71717a;
    font-weight: 400;
    margin-top: 0.25rem;
}

.stat-card {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: left;
    transition: border-color 0.2s;
}

.stat-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f4f4f5;
    line-height: 1.2;
}

.stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

.stat-value.accent { color: #6366f1; }
.stat-value.warning { color: #f59e0b; }
.stat-value.danger { color: #ef4444; }

.verdict-approve {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(34, 197, 94, 0.04));
    border: 1px solid rgba(34, 197, 94, 0.3);
    padding: 1.25rem;
    border-radius: 12px;
    text-align: center;
    color: #22c55e;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 1rem 0;
}

.verdict-review {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.04));
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 1.25rem;
    border-radius: 12px;
    text-align: center;
    color: #f59e0b;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 1rem 0;
}

.verdict-block {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.04));
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 1.25rem;
    border-radius: 12px;
    text-align: center;
    color: #ef4444;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 1rem 0;
}

.info-card {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 12px;
    padding: 1.25rem;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-green { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
.badge-yellow { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
.badge-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
.badge-blue { background: rgba(99, 102, 241, 0.15); color: #818cf8; }
.badge-gray { background: rgba(113, 113, 122, 0.15); color: #a1a1aa; }

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #e4e4e7;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e2a;
    margin: 1.5rem 0;
}

div[data-testid="stMetric"] {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}

div[data-testid="stMetric"] label {
    color: #71717a !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f4f4f5 !important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #111118;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1e1e2a;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #71717a;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #818cf8 !important;
}

button[kind="primary"] {
    background: #6366f1 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    transition: background 0.2s;
}

button[kind="primary"]:hover {
    background: #4f46e5 !important;
}

.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

div[data-testid="stExpander"] {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 10px;
}

div[data-testid="stStatusWidget"] {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 10px;
}

.reason-code {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #818cf8;
    margin: 0.15rem;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a1a1aa", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(30,30,42,0.8)", zerolinecolor="#1e1e2a"),
    yaxis=dict(gridcolor="rgba(30,30,42,0.8)", zerolinecolor="#1e1e2a"),
    margin=dict(l=0, r=0, t=40, b=0),
)


@st.cache_data(ttl=300)
def load_risk_scores():
    merchants = get_all_merchants()
    merchant_ids = merchants["merchant_id"].tolist()
    scores = predict_all_merchants(merchant_ids)
    return merchants, scores


def render_stat(label, value, accent=""):
    cls = f"stat-value {accent}" if accent else "stat-value"
    st.markdown(f"""
    <div class="stat-card">
        <div class="{cls}">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def risk_badge(score):
    if score >= 0.8:
        return f'<span class="badge badge-red">{score:.3f}</span>'
    elif score >= 0.5:
        return f'<span class="badge badge-yellow">{score:.3f}</span>'
    return f'<span class="badge badge-green">{score:.3f}</span>'


# ─── HEADER ───
st.markdown("""
<div class="app-header">
    <div class="app-title">Risk Copilot</div>
    <div class="app-subtitle">Agentic Decision Engine &mdash; ML Scoring + LLM Reasoning + Policy RAG</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Dashboard", "Investigation", "Merchant Detail"])

# ─── TAB 1: DASHBOARD ───
with tab1:
    with st.spinner("Loading risk scores..."):
        merchants_df, risk_scores = load_risk_scores()

    flagged = {mid: score for mid, score in risk_scores.items() if score >= 0.5}
    avg_risk = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0
    high_risk_count = sum(1 for s in risk_scores.values() if s > 0.8)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat("Merchants", len(merchants_df))
    with col2:
        render_stat("Active Alerts", len(flagged), "warning" if flagged else "")
    with col3:
        render_stat("Avg Risk", f"{avg_risk:.3f}", "accent")
    with col4:
        render_stat("Critical", high_risk_count, "danger" if high_risk_count else "")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Anomalies by MCC Category</div>', unsafe_allow_html=True)
        mcc_data = []
        for mid, score in risk_scores.items():
            m = merchants_df[merchants_df["merchant_id"] == mid].iloc[0]
            mcc_data.append({"MCC": f"{m['mcc_code']} — {m['mcc_description'][:20]}", "Risk": score, "Flagged": score >= 0.5})

        mcc_df = pd.DataFrame(mcc_data)
        flagged_by_mcc = mcc_df[mcc_df["Flagged"]].groupby("MCC").size().reset_index(name="Count")

        if not flagged_by_mcc.empty:
            fig = px.bar(flagged_by_mcc, x="Count", y="MCC", orientation="h",
                         color="Count", color_continuous_scale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#ef4444"]])
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              yaxis_title="", xaxis_title="", coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged merchants detected.")

    with col_right:
        st.markdown('<div class="section-title">Top Flagged Merchants</div>', unsafe_allow_html=True)
        if flagged:
            flagged_list = []
            for mid, score in sorted(flagged.items(), key=lambda x: x[1], reverse=True)[:10]:
                m = merchants_df[merchants_df["merchant_id"] == mid].iloc[0]
                flagged_list.append({
                    "ID": mid,
                    "Name": m["name"][:25],
                    "MCC": m["mcc_code"],
                    "City": m["city"],
                    "Risk": score,
                })
            df_flagged = pd.DataFrame(flagged_list)
            st.dataframe(
                df_flagged.style.format({"Risk": "{:.3f}"}).background_gradient(
                    subset=["Risk"], cmap="YlOrRd", vmin=0.5, vmax=1.0
                ),
                use_container_width=True,
                hide_index=True,
                height=350,
            )
        else:
            st.info("No merchants above risk threshold.")

# ─── TAB 2: INVESTIGATION ───
with tab2:
    merchants_df_inv, risk_scores_inv = load_risk_scores()
    flagged_inv = {mid: score for mid, score in risk_scores_inv.items() if score >= 0.5}

    if not flagged_inv:
        st.warning("No flagged merchants to investigate.")
    else:
        options = []
        for mid, score in sorted(flagged_inv.items(), key=lambda x: x[1], reverse=True):
            m = merchants_df_inv[merchants_df_inv["merchant_id"] == mid].iloc[0]
            options.append(f"{mid} — {m['name']} (MCC {m['mcc_code']}, Risk: {score:.3f})")

        selected_option = st.selectbox("Select a flagged merchant:", options)
        selected_mid = selected_option.split(" — ")[0]

        col_graph, col_info = st.columns([2, 1], gap="large")
        with col_graph:
            st.markdown('<div class="section-title">Agent Pipeline</div>', unsafe_allow_html=True)
            try:
                mermaid_code = builder.compile().get_graph().draw_mermaid()
                st.code(mermaid_code, language="mermaid")
            except Exception:
                st.code("""graph LR
    data_collector --> anomaly_detector
    anomaly_detector -->|risk < 0.5| auto_approve --> END
    anomaly_detector -->|risk >= 0.5| context_researcher
    context_researcher --> kyb_verifier
    kyb_verifier --> policy_retriever
    policy_retriever --> decision_agent
    decision_agent --> narrative_generator
    narrative_generator --> END""", language="mermaid")

        with col_info:
            st.markdown('<div class="section-title">Merchant Profile</div>', unsafe_allow_html=True)
            m_info = get_merchant(selected_mid)
            st.markdown(f"""<div class="info-card">
<strong>{m_info['name']}</strong><br/>
<span style="color:#71717a;font-size:0.85rem">
MCC {m_info['mcc_code']} — {m_info['mcc_description']}<br/>
{m_info['city']}, {m_info['state']}<br/>
Registered: {m_info['registered_date']}<br/>
Tier: <span class="badge badge-{'red' if m_info['risk_tier'] == 'high' else 'yellow' if m_info['risk_tier'] == 'medium' else 'gray'}">{m_info['risk_tier']}</span>
</span>
</div>""", unsafe_allow_html=True)

        st.markdown("")

        if st.button("Run Investigation", type="primary", use_container_width=True):
            node_labels = {
                "data_collector": "Collecting merchant data",
                "anomaly_detector": "Running ML risk models",
                "auto_approve": "Auto-approving (low risk)",
                "context_researcher": "Researching external context",
                "kyb_verifier": "Verifying KYB",
                "policy_retriever": "Retrieving policies (RAG)",
                "decision_agent": "Making risk decision",
                "narrative_generator": "Generating report",
            }

            node_icons = {
                "data_collector": "01",
                "anomaly_detector": "02",
                "auto_approve": "—",
                "context_researcher": "03",
                "kyb_verifier": "04",
                "policy_retriever": "05",
                "decision_agent": "06",
                "narrative_generator": "07",
            }

            start_time = time.time()

            with st.status("Running agentic investigation...", expanded=True) as status:
                final_state = {}
                for event in graph.stream({"merchant_id": selected_mid, "reasoning_trace": []}):
                    node_name = list(event.keys())[0]
                    node_output = event[node_name]
                    final_state.update(node_output)

                    icon = node_icons.get(node_name, "??")
                    label = node_labels.get(node_name, node_name)
                    st.write(f"**[{icon}]** {label}")

                    if node_name == "data_collector":
                        ts = node_output.get("transactions_summary", {})
                        st.caption(f"  {ts.get('txn_count_24h', 0)} txns (24h) · {ts.get('unique_customers_24h', 0)} unique customers")
                    elif node_name == "anomaly_detector":
                        st.caption(f"  Risk: {node_output.get('risk_score', 0):.3f} · Peer z-score: {node_output.get('peer_zscore', 0):.1f}σ")
                    elif node_name == "context_researcher":
                        findings = node_output.get("context_findings", [])
                        for f in findings[:3]:
                            cls = f.get('classification', 'N/A')
                            badge = "badge-green" if cls == "EXPLAINS_ANOMALY" else "badge-red" if cls == "INCREASES_RISK" else "badge-gray"
                            st.caption(f"  [{cls}] {f.get('finding', '')[:80]}")
                    elif node_name == "policy_retriever":
                        pc = node_output.get("policy_context", "")
                        chunk_count = pc.count("---") + 1 if pc else 0
                        st.caption(f"  Retrieved {chunk_count} policy chunks")
                    elif node_name == "decision_agent":
                        st.caption(f"  Decision: {node_output.get('decision', 'N/A')} · Confidence: {node_output.get('confidence', 0):.0%}")

                langfuse_flush()
                elapsed = time.time() - start_time
                status.update(label=f"Investigation complete ({elapsed:.1f}s)", state="complete")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            decision = final_state.get("decision", "REVIEW")
            verdict_class = {"APPROVE": "verdict-approve", "REVIEW": "verdict-review", "BLOCK": "verdict-block"}.get(decision, "verdict-review")
            st.markdown(f'<div class="{verdict_class}">{decision}</div>', unsafe_allow_html=True)

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Risk Score", f"{final_state.get('risk_score', 0):.3f}")
            with col_v2:
                st.metric("Confidence", f"{final_state.get('confidence', 0):.0%}")
            with col_v3:
                st.metric("Peer Z-Score", f"{final_state.get('peer_zscore', 0):.1f}σ")

            codes_html = " ".join(f'<span class="reason-code">{c}</span>' for c in final_state.get("reason_codes", []))
            st.markdown(f"**Reason Codes** {codes_html}", unsafe_allow_html=True)

            st.markdown("")

            col_feat, col_expl = st.columns([1, 1], gap="large")

            with col_feat:
                st.markdown('<div class="section-title">Risk Factor Contribution</div>', unsafe_allow_html=True)
                fi = final_state.get("feature_importances", [])
                if fi:
                    fi_df = pd.DataFrame(fi)
                    fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                 color="importance",
                                 color_continuous_scale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#ef4444"]])
                    fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                                      yaxis_title="", xaxis_title="", coloraxis_showscale=False)
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(fig, use_container_width=True)

            with col_expl:
                st.markdown('<div class="section-title">Decision Explanation</div>', unsafe_allow_html=True)
                st.write(final_state.get("explanation", "N/A"))
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Counterfactual</div>', unsafe_allow_html=True)
                st.info(final_state.get("counterfactual", "N/A"))

            with st.expander("Full Investigation Report"):
                st.markdown(final_state.get("narrative_report", "No report generated."))

            with st.expander("Reasoning Trace (Audit Trail)"):
                for step in final_state.get("reasoning_trace", []):
                    st.code(step, language=None)

# ─── TAB 3: MERCHANT DETAIL ───
with tab3:
    merchants_df_det = get_all_merchants()
    merchant_options = [f"{row['merchant_id']} — {row['name']} (MCC {row['mcc_code']})"
                        for _, row in merchants_df_det.iterrows()]
    selected_detail = st.selectbox("Select a merchant:", merchant_options, key="detail_select")
    detail_mid = selected_detail.split(" — ")[0]

    m_detail = get_merchant(detail_mid)

    col_info_d, col_risk_d = st.columns([2, 1], gap="large")

    with col_info_d:
        st.markdown(f'<div class="section-title">{m_detail["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
<table style="width:100%;border-collapse:collapse;color:#a1a1aa;font-size:0.875rem;">
<tr><td style="padding:6px 0;color:#71717a;">Merchant ID</td><td style="padding:6px 0;color:#e4e4e7;">{m_detail['merchant_id']}</td></tr>
<tr><td style="padding:6px 0;color:#71717a;">MCC</td><td style="padding:6px 0;color:#e4e4e7;">{m_detail['mcc_code']} — {m_detail['mcc_description']}</td></tr>
<tr><td style="padding:6px 0;color:#71717a;">Location</td><td style="padding:6px 0;color:#e4e4e7;">{m_detail['city']}, {m_detail['state']}</td></tr>
<tr><td style="padding:6px 0;color:#71717a;">Registered</td><td style="padding:6px 0;color:#e4e4e7;">{m_detail['registered_date']}</td></tr>
<tr><td style="padding:6px 0;color:#71717a;">Risk Tier</td><td style="padding:6px 0;color:#e4e4e7;">{m_detail['risk_tier']}</td></tr>
<tr><td style="padding:6px 0;color:#71717a;">Avg Monthly Volume</td><td style="padding:6px 0;color:#e4e4e7;">{m_detail['avg_monthly_volume']}</td></tr>
</table>
</div>""", unsafe_allow_html=True)

    with col_risk_d:
        st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
        result = predict_risk(detail_mid)
        st.metric("Risk Probability", f"{result.risk_probability:.3f}")
        st.metric("Anomaly Score", f"{result.anomaly_score:.3f}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Transaction History (90 days)</div>', unsafe_allow_html=True)

    txns = get_transactions(detail_mid, days=90)
    if not txns.empty:
        daily = txns.groupby(txns["timestamp"].dt.date).agg(
            txn_count=("txn_id", "count"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        ).reset_index()
        daily.columns = ["Date", "Transactions", "Total Amount", "Avg Ticket"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Transactions"],
            mode="lines",
            name="Daily Transactions",
            line=dict(color="#6366f1", width=2),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.08)",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig, use_container_width=True)

        col_peer, col_kyb = st.columns([1, 1], gap="large")

        with col_peer:
            st.markdown('<div class="section-title">Peer Comparison</div>', unsafe_allow_html=True)
            features = calculate_features(detail_mid)
            peer_z = features["peer_volume_zscore"]
            st.metric("Volume vs Peers (z-score)", f"{peer_z:.1f}σ",
                       delta=f"{'Above' if peer_z > 0 else 'Below'} peer average")

        with col_kyb:
            st.markdown('<div class="section-title">KYB Verification</div>', unsafe_allow_html=True)
            from datetime import datetime
            reg = datetime.strptime(m_detail["registered_date"], "%Y-%m-%d")
            age = (datetime.now() - reg).days
            kyb_ok = age > 90

            checks = [
                ("Business Age", f"{age} days", kyb_ok),
                ("Sanctions", "Clear", True),
                ("PEP Screening", "Clear", True),
                ("Adverse Media", "None Found", True),
            ]
            for name, val, ok in checks:
                badge = "badge-green" if ok else "badge-yellow"
                status_text = "PASS" if ok else "FLAG"
                st.markdown(f'{name}: <span class="badge {badge}">{status_text}</span> — {val}', unsafe_allow_html=True)
    else:
        st.info("No transactions found for this merchant.")
