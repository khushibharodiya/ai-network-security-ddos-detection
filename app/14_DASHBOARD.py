
import warnings
warnings.filterwarnings("ignore")

import ipaddress
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from inference_engine import load_artifacts, predict, generate_security_report


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Network Attack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK / CYBER SECURITY THEME
# ============================================================
st.markdown(
    """
<style>
/* ---------- APP ---------- */
.stApp {
    background: #05070b;
    color: #f8fafc;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: #090d14;
    border-right: 1px solid #182235;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* ---------- HERO ---------- */
.hero {
    background:
        radial-gradient(circle at 85% 20%, rgba(14,165,233,.20), transparent 28%),
        radial-gradient(circle at 10% 90%, rgba(37,99,235,.15), transparent 30%),
        linear-gradient(135deg, #0b1220 0%, #070b12 100%);
    border: 1px solid #1d4ed8;
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 22px;
    box-shadow: 0 0 35px rgba(14,165,233,.08);
}

.hero h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 800;
    color: #e0f2fe;
}

.hero p {
    color: #94a3b8;
    margin: 8px 0 0 0;
    font-size: 15px;
}

/* ---------- SECTION CARDS ---------- */
.section-card {
    background: #090d14;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px;
    margin: 12px 0;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #e2e8f0;
    margin-bottom: 14px;
}

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0c1422, #080d15);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 15px 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,.25);
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

div[data-testid="stMetricValue"] {
    color: #e0f2fe !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #38bdf8, #3b82f6);
    color: white;
}

/* ---------- INPUTS ---------- */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #0b1220 !important;
    color: #f8fafc !important;
    border-color: #26364d !important;
}

label {
    color: #cbd5e1 !important;
}

/* ---------- VERDICT ---------- */
.verdict-box {
    border-radius: 15px;
    padding: 22px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    margin: 15px 0;
}

.verdict-attack {
    background: linear-gradient(135deg, #3b0710, #160306);
    border: 1px solid #ef4444;
    color: #fecaca;
    box-shadow: 0 0 25px rgba(239,68,68,.12);
}

.verdict-benign {
    background: linear-gradient(135deg, #032e1b, #03150c);
    border: 1px solid #22c55e;
    color: #bbf7d0;
}

/* ---------- RISK LEVEL ---------- */
.risk-critical { color: #ff4d5e; font-weight: 800; }
.risk-high     { color: #ff8a3d; font-weight: 800; }
.risk-medium   { color: #facc15; font-weight: 800; }
.risk-low      { color: #4ade80; font-weight: 800; }

/* ---------- REASON CHIP ---------- */
.reason-chip {
    display: block;
    background: #0b1220;
    border-left: 3px solid #0ea5e9;
    border-radius: 7px;
    padding: 9px 12px;
    margin-bottom: 7px;
    color: #dbeafe;
}

/* ---------- INFO BOX ---------- */
.info-box {
    background: #07111e;
    border: 1px solid #164e63;
    border-radius: 10px;
    padding: 12px 15px;
    color: #bae6fd;
    margin: 8px 0 14px 0;
}

/* ---------- TABLE ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 10px;
}

/* ---------- TABS ---------- */
button[data-baseweb="tab"] {
    color: #94a3b8;
    font-weight: 700;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8;
}

/* ---------- DIVIDER ---------- */
hr {
    border-color: #1e293b !important;
}

/* ---------- SMALL LABEL ---------- */
.small-label {
    color: #64748b;
    font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
    <h1>🛡️ Network Attack Detection</h1>
    <p>
        XGBoost-based traffic classifier — live prediction, risk scoring,
        traffic analysis, rule-based explanation, and security reporting
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    artifacts = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Run your final model-training script first so that model_artifacts/ "
        "contains the saved model, threshold, metrics and feature information."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()


metrics = artifacts.get("metrics", {})

# Use the saved threshold whenever possible.
THRESHOLD = float(metrics.get("threshold", 0.88))

# These are read from the saved final evaluation whenever available.
TEST_PRECISION = metrics.get("test_precision", metrics.get("precision", np.nan))
TEST_RECALL = metrics.get("test_recall", metrics.get("recall", np.nan))
TEST_ACCURACY = metrics.get(
    "test_accuracy",
    metrics.get(
        "accuracy",
        metrics.get("test_acc", metrics.get("test_accuracy_score", np.nan)),
    ),
)
TEST_F1 = metrics.get("test_f1", metrics.get("f1", np.nan))
TEST_ROC_AUC = metrics.get(
    "test_roc_auc",
    metrics.get(
        "roc_auc",
        metrics.get(
            "test_roc_auc_score",
            metrics.get(
                "roc_auc_score",
                metrics.get("test_auc", metrics.get("auc", np.nan)),
            ),
        ),
    ),
)


def fmt_percent(value, decimals=2):
    """Convert a 0-1 metric to a percentage string."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📊 Model Info")

    st.metric("Decision threshold", f"{THRESHOLD:.2f}")
    st.metric("Test Accuracy", fmt_percent(TEST_ACCURACY, 2))
    st.metric("Test Precision", fmt_percent(TEST_PRECISION, 2))
    st.metric("Test Recall", fmt_percent(TEST_RECALL, 2))
    st.metric("Test F1-Score", fmt_percent(TEST_F1, 2))
    st.metric("Test ROC-AUC", fmt_percent(TEST_ROC_AUC, 2))

    st.divider()

    st.markdown("### 🤖 Final Model")
    st.write("**XGBoost**")
    st.caption(
        "Final classifier with the saved project feature set. "
        "Prediction logic remains inside inference_engine.py."
    )

    st.divider()
    st.caption(
        "Project: AI-Powered Network Security & DDoS Detection Platform"
    )


# ============================================================
# HELPERS
# ============================================================
def validate_ip(value):
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def risk_class(level):
    return {
        "Critical": "risk-critical",
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(str(level), "")


def make_gauge(value, title, color="#0ea5e9"):
    value = float(np.clip(value, 0, 100))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 15, "color": "#cbd5e1"}},
            number={"suffix": "%", "font": {"size": 31, "color": "#f8fafc"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#64748b",
                    "tickfont": {"color": "#94a3b8"},
                },
                "bar": {"color": color},
                "bgcolor": "#0b1220",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#052e16"},
                    {"range": [25, 50], "color": "#422006"},
                    {"range": [50, 75], "color": "#431407"},
                    {"range": [75, 100], "color": "#450a0a"},
                ],
            },
        )
    )

    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=55, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
    )
    return fig


def make_risk_gauge(value, title, color="#ef4444"):
    value = float(np.clip(value, 0, 100))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 15, "color": "#cbd5e1"}},
            number={
                "suffix": " / 100",
                "font": {"size": 31, "color": "#f8fafc"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#64748b",
                    "tickfont": {"color": "#94a3b8"},
                },
                "bar": {"color": color},
                "bgcolor": "#0b1220",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#052e16"},
                    {"range": [25, 50], "color": "#422006"},
                    {"range": [50, 75], "color": "#431407"},
                    {"range": [75, 100], "color": "#450a0a"},
                ],
            },
        )
    )

    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=55, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
    )
    return fig


def create_pdf_report(title, report_text, result_df=None):
    """
    Create a simple PDF security report.
    Requires reportlab.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return None

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SecurityTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        spaceAfter=15,
    )

    body_style = ParagraphStyle(
        "SecurityBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=5,
    )

    story = [
        Paragraph(title, title_style),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            body_style,
        ),
        Spacer(1, 10),
    ]

    for line in str(report_text).splitlines():
        safe_line = (
            str(line)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.append(Paragraph(safe_line if safe_line.strip() else "&nbsp;", body_style))

    if result_df is not None and len(result_df) > 0:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Traffic Result", styles["Heading2"]))

        cols = [
            c for c in [
                "protocol",
                "src_port",
                "dst_port",
                "bytes_sent",
                "bytes_received",
                "prediction",
                "probability",
                "risk_score",
                "risk_band",
            ]
            if c in result_df.columns
        ]

        if cols:
            table_data = [cols]
            for _, row in result_df[cols].head(20).iterrows():
                table_data.append([str(row[c]) for c in cols])

            table = Table(table_data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#f1f5f9")]),
                    ]
                )
            )
            story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def show_model_information():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Model Information</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Final Model", "XGBoost")
    c2.metric("Threshold", f"{THRESHOLD:.2f}")
    c3.metric("Accuracy", fmt_percent(TEST_ACCURACY))
    c4.metric("Precision", fmt_percent(TEST_PRECISION))
    c5.metric("Recall", fmt_percent(TEST_RECALL))
    c6.metric("F1 Score", fmt_percent(TEST_F1))
    c7.metric("ROC-AUC", fmt_percent(TEST_ROC_AUC))
    if pd.isna(TEST_ROC_AUC):
        st.caption(
            "⚠️ ROC-AUC is N/A because the saved model_artifacts metrics "
            "do not contain a ROC-AUC value. The dashboard cannot invent "
            "the final evaluation metric."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TOP NAVIGATION
# ============================================================
tab_overview, tab_live, tab_batch = st.tabs(
    ["📊 Security Overview", "🔎 Live Detection", "📁 Batch Analysis"]
)


# ============================================================
# TAB 1 — SECURITY OVERVIEW
# ============================================================
with tab_overview:
    st.subheader("Security Overview")

    st.markdown(
        """
<div class="info-box">
This dashboard provides an overview of analyzed traffic, detected attacks,
risk levels, model performance and the current security posture.
</div>
""",
        unsafe_allow_html=True,
    )

    # Try to use the project CSV automatically.
    try:
        overview_df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "cybersecurity.csv")
        overview_results = predict(overview_df, artifacts)
    except Exception:
        overview_df = None
        overview_results = None

    if overview_results is not None and len(overview_results) > 0:
        total = len(overview_results)
        attacks = int((overview_results["prediction"] == "Attack").sum())
        benign = total - attacks
        attack_pct = attacks / total * 100 if total else 0

        critical = int((overview_results["risk_band"] == "Critical").sum())
        high = int((overview_results["risk_band"] == "High").sum())
        medium = int((overview_results["risk_band"] == "Medium").sum())
        low = int((overview_results["risk_band"] == "Low").sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Traffic Analyzed", f"{total:,}")
        k2.metric("Attacks Detected", f"{attacks:,}")
        k3.metric("Benign Traffic", f"{benign:,}")
        k4.metric("Attack Percentage", f"{attack_pct:.2f}%")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("### 📊 Risk Analysis")

            risk_counts = pd.DataFrame(
                {
                    "Risk Level": ["Critical", "High", "Medium", "Low"],
                    "Count": [critical, high, medium, low],
                }
            )

            fig_risk = px.bar(
                risk_counts,
                x="Risk Level",
                y="Count",
                color="Risk Level",
                color_discrete_map={
                    "Critical": "#ff3b4d",
                    "High": "#ff7a30",
                    "Medium": "#facc15",
                    "Low": "#22c55e",
                },
            )

            fig_risk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e5e7eb"},
                showlegend=False,
            )

            st.plotly_chart(fig_risk, use_container_width=True, key="overview_risk_chart")

        with right:
            st.markdown("### 📈 Risk Score Distribution")

            fig_dist = px.histogram(
                overview_results,
                x="risk_score",
                nbins=30,
            )

            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e5e7eb"},
                xaxis_title="Risk Score",
                yaxis_title="Traffic Count",
            )

            st.plotly_chart(fig_dist, use_container_width=True, key="overview_risk_distribution_chart")

        st.divider()
        show_model_information()

    else:
        st.warning(
            "cybersecurity.csv could not be loaded for the automatic overview. "
            "Use Batch Analysis to upload your traffic CSV."
        )
        show_model_information()


# ============================================================
# TAB 2 — LIVE DETECTION
# ============================================================
with tab_live:
    st.subheader("🔍 Live Traffic Detection")

    st.markdown(
        """
<div class="info-box">
Enter one network-traffic record and let the final XGBoost model calculate
the attack probability, risk score and risk level.
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("live_detection_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
            src_port = st.number_input(
                "Source Port",
                min_value=0,
                max_value=65535,
                value=51234,
            )
            dst_port = st.number_input(
                "Destination Port",
                min_value=0,
                max_value=65535,
                value=443,
            )

        with c2:
            bytes_sent = st.number_input(
                "Bytes Sent",
                min_value=0.0,
                value=15000.0,
            )
            bytes_received = st.number_input(
                "Bytes Received",
                min_value=0.0,
                value=45000.0,
            )
            is_internal = (
                st.selectbox("Internal Traffic?", ["No", "Yes"]) == "Yes"
            )

        with c3:
            src_ip = st.text_input(
                "Source IP",
                value="203.0.113.10",
            )
            dst_ip = st.text_input(
                "Destination IP",
                value="192.168.1.10",
            )

        user_agent = st.text_input(
            "User-Agent (optional)",
            value="",
        )
        url = st.text_input(
            "Request URL (optional)",
            value="",
        )

        submitted = st.form_submit_button(
            "🔍 ANALYZE TRAFFIC",
            use_container_width=True,
        )

    if submitted:
        if not validate_ip(src_ip) or not validate_ip(dst_ip):
            st.error(
                "Source IP and Destination IP must be valid IP addresses."
            )
        else:
            raw_row = pd.DataFrame(
                [
                    {
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "bytes_sent": bytes_sent,
                        "bytes_received": bytes_received,
                        "is_internal_traffic": int(is_internal),
                        "protocol": protocol,
                        "timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "user_agent": user_agent if user_agent else None,
                        "url": url if url else None,
                    }
                ]
            )

            try:
                result_df = predict(raw_row, artifacts)
                result = result_df.iloc[0]

                st.divider()

                prediction = str(result["prediction"])
                probability = float(result["probability"]) * 100
                risk_score = float(result["risk_score"])
                risk_band = str(result["risk_band"])

                is_attack = prediction.lower() == "attack"

                verdict_class = (
                    "verdict-attack"
                    if is_attack
                    else "verdict-benign"
                )
                verdict_icon = "🚨" if is_attack else "✅"

                st.markdown(
                    f"""
<div class="verdict-box {verdict_class}">
    {verdict_icon} {prediction.upper()}
</div>
""",
                    unsafe_allow_html=True,
                )

                r1, r2, r3 = st.columns(3)

                with r1:
                    st.metric(
                        "Probability",
                        f"{probability:.2f}%",
                    )

                with r2:
                    st.metric(
                        "Risk Score",
                        f"{risk_score:.1f} / 100",
                    )

                with r3:
                    st.metric(
                        "Risk Level",
                        risk_band.upper(),
                    )

                g1, g2 = st.columns(2)

                with g1:
                    st.plotly_chart(
                        make_gauge(
                            probability,
                            "Attack Probability",
                            "#ff3b4d" if is_attack else "#22c55e",
                        ),
                        use_container_width=True,
                        key="live_attack_probability_gauge",
                    )

                with g2:
                    risk_colors = {
                        "Low": "#22c55e",
                        "Medium": "#facc15",
                        "High": "#ff7a30",
                        "Critical": "#ff3b4d",
                    }

                    st.plotly_chart(
                        make_risk_gauge(
                            risk_score,
                            "Risk Score",
                            risk_colors.get(
                                risk_band,
                                "#0ea5e9",
                            ),
                        ),
                        use_container_width=True,
                        key="live_risk_score_gauge",
                    )

                st.markdown(
                    f"""
<div style="font-size:18px; margin:5px 0 15px 0;">
    Risk Level:
    <span class="{risk_class(risk_band)}">{risk_band.upper()}</span>
</div>
""",
                    unsafe_allow_html=True,
                )

                # ---------- Traffic details ----------
                st.markdown("### 📋 Traffic Details")

                traffic_display = pd.DataFrame(
                    [
                        {
                            "Source Port": src_port,
                            "Destination Port": dst_port,
                            "Protocol": protocol,
                            "Bytes Sent": bytes_sent,
                            "Bytes Received": bytes_received,
                            "Prediction": prediction,
                            "Probability": f"{probability:.2f}%",
                            "Risk Score": f"{risk_score:.1f}",
                            "Risk Level": risk_band,
                        }
                    ]
                )

                st.dataframe(
                    traffic_display,
                    use_container_width=True,
                    hide_index=True,
                )

                # ---------- Explanation ----------
                st.markdown("### 🧠 Rule-Based Explanation")

                explanation = str(
                    result.get(
                        "explanation",
                        "No explanation was returned by the inference engine.",
                    )
                )

                for reason in explanation.split("; "):
                    st.markdown(
                        f'<div class="reason-chip">• {reason}</div>',
                        unsafe_allow_html=True,
                    )

                # ---------- Security report ----------
                st.markdown("### 📄 Security Report")

                report_text = generate_security_report(result_df)

                st.code(
                    report_text,
                    language=None,
                )

                pdf_data = create_pdf_report(
                    "Network Security — Individual Traffic Report",
                    report_text,
                    result_df,
                )

                d1, d2 = st.columns(2)

                with d1:
                    st.download_button(
                        "📄 Download TXT Report",
                        data=report_text,
                        file_name="security_report_single.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

                with d2:
                    if pdf_data is not None:
                        st.download_button(
                            "📑 Generate PDF Report",
                            data=pdf_data,
                            file_name="security_report_single.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.warning(
                            "Install reportlab to enable PDF reports: "
                            "pip install reportlab"
                        )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ============================================================
# TAB 3 — BATCH ANALYSIS
# ============================================================
with tab_batch:
    st.subheader("📁 Batch Traffic Analysis")

    source_col1, source_col2 = st.columns([1, 2])

    with source_col1:
        source = st.radio(
            "Data source",
            ["Demo sample", "Upload CSV"],
        )

    with source_col2:
        uploaded_file = None
        sample_size = 200

        if source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload network traffic CSV",
                type="csv",
            )
        else:
            sample_size = st.slider(
                "Sample size",
                50,
                2000,
                200,
                step=50,
            )

    if source == "Upload CSV":
        if uploaded_file is None:
            st.info(
                "Upload a CSV above to see batch results, "
                "or switch to Demo sample."
            )
            st.stop()

        raw_df = pd.read_csv(uploaded_file)

    else:
        try:
            full_df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "cybersecurity.csv")
            raw_df = full_df.sample(
                min(sample_size, len(full_df)),
                random_state=42,
            )
        except FileNotFoundError:
            st.error(
                "cybersecurity.csv was not found. "
                "Upload the CSV instead."
            )
            st.stop()

    with st.spinner("Scoring traffic..."):
        try:
            results = predict(raw_df, artifacts)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            st.stop()

    total = len(results)
    attacks = int((results["prediction"] == "Attack").sum())
    benign = total - attacks

    attack_pct = attacks / total * 100 if total else 0
    avg_risk = float(results["risk_score"].mean()) if total else 0

    critical = int((results["risk_band"] == "Critical").sum())
    high = int((results["risk_band"] == "High").sum())
    medium = int((results["risk_band"] == "Medium").sum())
    low = int((results["risk_band"] == "Low").sum())

    # ---------- Security overview ----------
    st.markdown("### 1. 🛡️ Security Overview")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Traffic Analyzed", f"{total:,}")
    k2.metric("Attacks Detected", f"{attacks:,}")
    k3.metric("Benign Traffic", f"{benign:,}")
    k4.metric("Attack Percentage", f"{attack_pct:.2f}%")

    st.divider()

    # ---------- Risk analysis ----------
    st.markdown("### 2. 📊 Risk Analysis")

    risk_counts = pd.DataFrame(
        {
            "Risk Level": ["Critical", "High", "Medium", "Low"],
            "Count": [critical, high, medium, low],
        }
    )

    rc1, rc2 = st.columns(2)

    with rc1:
        fig_risk = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={
                "Critical": "#ff3b4d",
                "High": "#ff7a30",
                "Medium": "#facc15",
                "Low": "#22c55e",
            },
        )

        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e7eb"},
            showlegend=False,
        )

        st.plotly_chart(fig_risk, use_container_width=True, key="batch_risk_chart")

    with rc2:
        st.metric("Average Risk Score", f"{avg_risk:.1f} / 100")

        fig_dist = px.histogram(
            results,
            x="risk_score",
            nbins=30,
        )

        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e7eb"},
            xaxis_title="Risk Score",
            yaxis_title="Traffic Count",
        )

        st.plotly_chart(fig_dist, use_container_width=True, key="batch_risk_distribution_chart")

    st.divider()

    # ---------- Model information ----------
    st.markdown("### 3. 🤖 Model Information")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

    m1.metric("Model", "XGBoost")
    m2.metric("Threshold", f"{THRESHOLD:.2f}")
    m3.metric("Accuracy", fmt_percent(TEST_ACCURACY))
    m4.metric("Precision", fmt_percent(TEST_PRECISION))
    m5.metric("Recall", fmt_percent(TEST_RECALL))
    m6.metric("F1 Score", fmt_percent(TEST_F1))
    m7.metric("ROC-AUC", fmt_percent(TEST_ROC_AUC))

    if pd.isna(TEST_ROC_AUC):
        st.caption(
            "⚠️ ROC-AUC is N/A because it was not saved in the final "
            "evaluation metrics used by this dashboard."
        )

    st.divider()

    # ---------- Traffic details ----------
    st.markdown("### 4. 🔍 Traffic Details")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        show_only = st.multiselect(
            "Prediction",
            ["Attack", "Benign"],
            default=["Attack", "Benign"],
        )

    with filter_col2:
        min_risk = st.slider(
            "Minimum Risk Score",
            0,
            100,
            0,
        )

    filtered = results[
        results["prediction"].isin(show_only)
        & (results["risk_score"] >= min_risk)
    ]

    desired_cols = [
        "src_port",
        "dst_port",
        "protocol",
        "bytes_sent",
        "bytes_received",
        "prediction",
        "probability",
        "risk_score",
        "risk_band",
    ]

    available_cols = [
        c for c in desired_cols if c in filtered.columns
    ]

    display_df = filtered[available_cols].copy()

    if "probability" in display_df.columns:
        display_df["probability"] = (
            display_df["probability"] * 100
        ).round(2).astype(str) + "%"

    if "risk_score" in display_df.columns:
        display_df["risk_score"] = (
            display_df["risk_score"].round(1)
        )

    display_df = display_df.rename(
        columns={
            "src_port": "Source Port",
            "dst_port": "Destination Port",
            "protocol": "Protocol",
            "bytes_sent": "Bytes Sent",
            "bytes_received": "Bytes Received",
            "prediction": "Prediction",
            "probability": "Probability",
            "risk_score": "Risk Score",
            "risk_band": "Risk Level",
        }
    )

    st.dataframe(
        display_df.sort_values(
            "Risk Score",
            ascending=False,
        ) if "Risk Score" in display_df.columns else display_df,
        use_container_width=True,
        height=390,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(filtered):,} of {total:,} analyzed records."
    )

    st.divider()

    # ---------- Security report ----------
    st.markdown("### 5. 📄 Security Report")

    report_text = generate_security_report(results)

    st.code(
        report_text,
        language=None,
    )

    pdf_data = create_pdf_report(
        "Network Security — Batch Security Report",
        report_text,
        results,
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "📄 Download TXT Report",
            data=report_text,
            file_name="security_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "📊 Download Scored CSV",
            data=results.to_csv(index=False),
            file_name="scored_traffic.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        if pdf_data is not None:
            st.download_button(
                "📑 Generate PDF Report",
                data=pdf_data,
                file_name="security_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning(
                "Install reportlab: pip install reportlab"
            )