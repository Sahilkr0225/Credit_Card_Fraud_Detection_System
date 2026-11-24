"""
🛡️ Intelligent Credit Card Fraud Detection & Risk Assessment System
Streamlit Dashboard (Module 2: Security & Application Layer)
"""

import os
import sys
from datetime import datetime, timezone
import random
from typing import Dict, Any, List

# Ensure repository root is on sys.path for direct streamlit execution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from src.security.sim_swap_engine import SimSwapEngine
from src.security.velocity_monitor import VelocityMonitor
from src.security.otp_monitor import OtpMonitor
from src.security.device_location_rules import DeviceLocationRules
from src.security.signals_engine import SecuritySignalsEngine
from src.security.simulator import SyntheticTransactionGenerator, CITIES, MERCHANTS
from src.risk_engine import UnifiedRiskEngine
from app.components.alert_visualizations import (
    render_risk_gauge,
    render_risk_distribution,
    render_amount_vs_risk_scatter,
    render_flags_breakdown,
    render_score_breakdown_bar,
)
from app.components.transaction_monitor import (
    render_transaction_feed,
    render_transaction_details,
)


# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Intelligent Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0E131F;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Session State Initialization
# ==========================================
if "security_engine" not in st.session_state:
    st.session_state.security_engine = SecuritySignalsEngine()

if "risk_engine" not in st.session_state:
    st.session_state.risk_engine = UnifiedRiskEngine(w_ml=0.45, w_anomaly=0.15, w_security=0.40)

if "generator" not in st.session_state:
    st.session_state.generator = SyntheticTransactionGenerator(seed=42)

if "transactions_df" not in st.session_state:
    st.session_state.transactions_df = pd.DataFrame()


def evaluate_and_record_batch(raw_transactions: List[Dict[str, Any]]) -> None:
    """Evaluate batch of raw transactions and append to session state."""
    records: List[Dict[str, Any]] = []

    for tx in raw_transactions:
        # 1. Security Telemetry Evaluation (Member 2)
        sec_res = st.session_state.security_engine.evaluate_transaction(tx)

        # 2. ML Prediction (Member 1 integration / Contract mock fallback)
        is_fraud = tx.get("is_ground_truth_fraud", 0) == 1
        if is_fraud:
            ml_prob = round(random.uniform(0.78, 0.96), 3)
            is_anom = True
            anom_score = round(random.uniform(-0.40, -0.15), 3)
            ml_tier = "HIGH" if ml_prob < 0.85 else "CRITICAL"
        else:
            ml_prob = round(random.uniform(0.01, 0.09), 3)
            is_anom = False
            anom_score = round(random.uniform(0.05, 0.25), 3)
            ml_tier = "LOW"

        ml_res = {
            "transaction_id": tx["transaction_id"],
            "fraud_probability": ml_prob,
            "ml_prediction": 1 if is_fraud else 0,
            "anomaly_score": anom_score,
            "is_anomaly": is_anom,
            "ml_risk_tier": ml_tier,
            "model_version": "hybrid_rf_iforest_v1.0",
        }

        # 3. Unified Risk Assessment Engine
        assessment = st.session_state.risk_engine.assess(
            ml_result=ml_res,
            security_result=sec_res,
            timestamp=tx.get("timestamp"),
        )

        # Merge for tabular viewing
        flat_record = {
            "transaction_id": tx["transaction_id"],
            "user_id": tx.get("user_id", "USR_UNKNOWN"),
            "timestamp": tx.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "Amount": tx.get("amount", tx.get("Amount", 0.0)),
            "merchant": tx.get("merchant", "General Merchant"),
            "merchant_category": tx.get("merchant_category", "retail"),
            "city": tx.get("city", "Unknown"),
            "country": tx.get("country", "Unknown"),
            "device_id": tx.get("device_id", "DEV_UNKNOWN"),
            "scenario_label": tx.get("scenario_label", "NORMAL"),
            "unified_risk_score": assessment["unified_risk_score"],
            "risk_level": assessment["risk_level"],
            "decision": assessment["decision"],
            "ml_fraud_probability": ml_res["fraud_probability"],
            "ml_contribution": assessment["breakdown"]["ml_contribution"],
            "anomaly_score": ml_res["anomaly_score"],
            "anomaly_contribution": assessment["breakdown"]["anomaly_contribution"],
            "is_anomaly": ml_res["is_anomaly"],
            "security_risk_score": sec_res["security_risk_score"],
            "security_contribution": assessment["breakdown"]["security_contribution"],
            "sim_swap_detected": sec_res["sim_swap_detected"],
            "sim_swap_age_hours": sec_res["sim_swap_age_hours"],
            "otp_failures_count": sec_res["otp_failures_count"],
            "otp_verified": tx.get("otp_verified", True),
            "velocity_score": sec_res["velocity_score"],
            "carding_flag": sec_res["carding_flag"],
            "impossible_travel_detected": sec_res.get("impossible_travel_detected", False),
            "implied_speed_kmh": sec_res.get("implied_speed_kmh", 0.0),
            "critical_flags": assessment["breakdown"]["critical_flags"],
        }
        records.append(flat_record)

    new_df = pd.DataFrame(records)
    if st.session_state.transactions_df.empty:
        st.session_state.transactions_df = new_df
    else:
        st.session_state.transactions_df = pd.concat(
            [new_df, st.session_state.transactions_df], ignore_index=True
        )


# Populate initial demo stream if empty
if st.session_state.transactions_df.empty:
    initial_batch = st.session_state.generator.generate_batch(size=25, fraud_ratio=0.20)
    evaluate_and_record_batch(initial_batch)


# ==========================================
# Sidebar: System Status & Quick Controls
# ==========================================
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Status-Live%20Monitoring-2EA44F?style=for-the-badge",
        use_container_width=True,
    )
    st.title("🛡️ Security Operations")
    st.caption("Module 2: Security & Application Layer")

    st.markdown("---")
    st.subheader("⚡ Attack Simulator Studio")
    st.write("Trigger synthetic fraud scenarios to test the detection pipelines in real-time:")

    if st.button("📱 Simulate SIM-Swap Attack", use_container_width=True):
        attack = st.session_state.generator.generate_sim_swap_attack()
        evaluate_and_record_batch([attack])
        st.success("SIM-Swap account takeover simulated!")

    if st.button("⚡ Simulate Carding Burst (5 txns)", use_container_width=True):
        burst = st.session_state.generator.generate_carding_burst(burst_size=5)
        evaluate_and_record_batch(burst)
        st.success("Rapid card-testing attack simulated!")

    if st.button("✈️ Simulate Impossible Travel", use_container_width=True):
        pair = st.session_state.generator.generate_impossible_travel()
        evaluate_and_record_batch(pair)
        st.success("Geo-velocity jump simulated (NY -> Tokyo)!")

    if st.button("🔑 Simulate OTP Brute-Force", use_container_width=True):
        otp_atk = st.session_state.generator.generate_otp_brute_force_attack()
        evaluate_and_record_batch([otp_atk])
        st.success("Repeated OTP failure attack simulated!")

    if st.button("🟢 Generate 15 Normal Txns", use_container_width=True):
        normal_batch = [
            st.session_state.generator.generate_normal_transaction() for _ in range(15)
        ]
        evaluate_and_record_batch(normal_batch)
        st.info("Added 15 legitimate transactions.")

    if st.button("🎲 Generate Mixed Stream (30)", use_container_width=True):
        mixed = st.session_state.generator.generate_batch(size=30, fraud_ratio=0.15)
        evaluate_and_record_batch(mixed)
        st.info("Generated 30 mixed transactions.")

    st.markdown("---")
    if st.button("🗑️ Reset Stream Data", use_container_width=True):
        st.session_state.transactions_df = pd.DataFrame()
        st.session_state.security_engine.velocity_monitor.clear()
        st.session_state.security_engine.otp_monitor.clear()
        st.session_state.security_engine.device_location_rules.clear()
        # Repopulate minimal baseline
        fresh_batch = st.session_state.generator.generate_batch(size=10, fraud_ratio=0.10)
        evaluate_and_record_batch(fresh_batch)
        st.rerun()


# ==========================================
# Main Dashboard Header & KPI Metrics
# ==========================================
st.title("Intelligent Credit Card Fraud Detection & Risk Assessment System")
st.markdown(
    "**Hybrid Financial Defense Platform:** Supervised ML + Anomaly Detection + Real-time SIM-Swap, "
    "Velocity, and Impossible Travel Telemetry."
)

df = st.session_state.transactions_df

# Top KPI Summary Bar
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_tx = len(df)
fraud_tx = len(df[df["decision"] == "DECLINE"])
challenged_tx = len(df[df["decision"].isin(["CHALLENGE_OTP", "MANUAL_REVIEW"])])
prevented_amount = df[df["decision"] == "DECLINE"]["Amount"].sum() if not df.empty else 0.0
avg_score = df["unified_risk_score"].mean() if not df.empty else 0.0

with kpi1:
    st.metric("Total Evaluated", f"{total_tx:,}")
with kpi2:
    st.metric("Declined (Blocked)", f"{fraud_tx:,}", delta=f"{(fraud_tx/max(1, total_tx))*100:.1f}%")
with kpi3:
    st.metric("Challenged / Review", f"{challenged_tx:,}")
with kpi4:
    st.metric("Fraud Volume Prevented", f"${prevented_amount:,.2f}")
with kpi5:
    st.metric("Average Risk Score", f"{avg_score:.1f}/100")

st.markdown("---")


# ==========================================
# Navigation Tabs
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🖥️ Live Transaction Feed & Monitor",
    "📊 Alert & Security Visualizations",
    "🧪 Live Transaction Sandbox",
    "⚙️ Risk Engine Configuration",
])


# ----------------------------------------------------
# TAB 1: Live Transaction Feed & Monitor
# ----------------------------------------------------
with tab1:
    st.subheader("🔴 Real-Time Transaction Stream")
    selected_id = render_transaction_feed(df)

    if selected_id:
        selected_row = df[df["transaction_id"] == selected_id].iloc[0]
        render_transaction_details(selected_row)


# ----------------------------------------------------
# TAB 2: Alert & Security Visualizations
# ----------------------------------------------------
with tab2:
    st.subheader("📊 Analytical Threat Telemetry & Visualizations")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.plotly_chart(render_risk_distribution(df), use_container_width=True)
    with row1_c2:
        st.plotly_chart(render_amount_vs_risk_scatter(df), use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.plotly_chart(render_flags_breakdown(df), use_container_width=True)
    with row2_c2:
        st.markdown("#### 🛡️ Defense Engine Architecture Breakdown")
        st.markdown(
            """
            The unified system continuously correlates three distinct layers of defense:
            1. **Supervised ML Probability ($S_{ML}$):** Pre-trained Random Forest and Logistic Regression
               models identifying non-linear patterns across scaled PCA features $V_1 - V_{28}$ and $Amount$.
            2. **Unsupervised Outlier Metric ($S_{Anomaly}$):** Isolation Forest identifying zero-day
               spending anomalies without needing historical labels.
            3. **Behavioral & Telecom Security Rules ($S_{Security}$):**
               - **SIM-Swap Engine:** High-value transfer within 24-48h of carrier SIM porting triggers immediate critical flags.
               - **Velocity Monitor:** Rolling window frequency tracking to thwart automated card-testing bots.
               - **OTP Failure Tracker:** Monitors brute-force credential stuffing and social engineering pressure.
               - **Device & Location Rules:** Great-Circle Haversine calculations detecting impossible supersonic travel.
            """
        )


# ----------------------------------------------------
# TAB 3: Live Transaction Sandbox (Manual Evaluator)
# ----------------------------------------------------
with tab3:
    st.subheader("🧪 Single Transaction Risk Sandbox")
    st.write(
        "Interactively simulate custom transaction parameters to verify how the Unified Risk Engine "
        "and security rules respond in real-time."
    )

    sb_col1, sb_col2, sb_col3 = st.columns(3)

    with sb_col1:
        st.markdown("##### 💳 Payment Details")
        sb_user = st.text_input("User ID", value="USR_VIVA_TEST")
        sb_amount = st.number_input("Amount ($)", min_value=0.50, max_value=50000.0, value=1250.0, step=50.0)
        sb_cat = st.selectbox("Merchant Category", ["retail", "crypto", "gambling", "p2p", "travel", "grocery"])
        sb_merchant = st.text_input("Merchant Name", value="Instant Crypto Gateway")

    with sb_col2:
        st.markdown("##### 📱 Telephony & Auth Signals")
        sb_sim_swap = st.checkbox("SIM Swap Detected?", value=True)
        sb_sim_age = st.slider("SIM Swap Age (Hours)", min_value=0.5, max_value=120.0, value=3.5, step=0.5)
        sb_otp_verified = st.checkbox("OTP Verified Successfully?", value=True, help="Even if verified, recent SIM swap indicates intercepted OTP!")
        sb_otp_failures = st.slider("Failed OTP Retries", min_value=0, max_value=5, value=0)

    with sb_col3:
        st.markdown("##### 📍 Device & ML Signals")
        sb_new_device = st.checkbox("Unrecognized New Device?", value=True)
        city_names = [c["city"] for c in CITIES]
        sb_city = st.selectbox("Transaction City", city_names, index=6)  # Tokyo
        sb_speed = st.number_input("Geo Velocity (Implied km/h)", min_value=0.0, max_value=25000.0, value=6500.0)
        sb_ml_prob = st.slider("Supervised ML Fraud Probability (%)", min_value=0, max_value=100, value=35)

    if st.button("⚡ Evaluate Custom Transaction Now", use_container_width=True, type="primary"):
        # Synthesize custom event
        custom_tx = {
            "transaction_id": f"TXN_MANUAL_{random.randint(100, 999)}",
            "user_id": sb_user,
            "Amount": sb_amount,
            "amount": sb_amount,
            "merchant": sb_merchant,
            "merchant_category": sb_cat,
            "city": sb_city,
            "country": "Simulated",
            "device_id": "DEV_MANUAL_TEST" if not sb_new_device else "DEV_UNKNOWN_NEW",
            "sim_swap_detected": sb_sim_swap,
            "sim_swap_age_hours": sb_sim_age,
            "otp_verified": sb_otp_verified,
            "otp_failures_count": sb_otp_failures,
            "is_ground_truth_fraud": 1 if (sb_sim_swap and sb_sim_age < 24) or sb_speed > 850 else 0,
        }

        # Security evaluation
        sec_eval = st.session_state.security_engine.evaluate_transaction(custom_tx)

        # Inject manual override for impossible travel speed if specified
        if sb_speed > 850.0:
            sec_eval["impossible_travel_detected"] = True
            sec_eval["implied_speed_kmh"] = sb_speed
            if "IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED" not in sec_eval["flags"]:
                sec_eval["flags"].append("IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED")
            sec_eval["security_risk_score"] = max(sec_eval["security_risk_score"], 0.90)

        # ML simulation
        ml_eval = {
            "transaction_id": custom_tx["transaction_id"],
            "fraud_probability": sb_ml_prob / 100.0,
            "ml_prediction": 1 if sb_ml_prob > 50 else 0,
            "anomaly_score": -0.25 if sb_ml_prob > 50 else 0.10,
            "is_anomaly": sb_ml_prob > 50,
            "ml_risk_tier": "HIGH" if sb_ml_prob > 60 else "LOW",
        }

        # Unified assessment
        eval_result = st.session_state.risk_engine.assess(ml_eval, sec_eval)

        st.markdown("#### 🎯 Real-Time Evaluation Result")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.plotly_chart(
                render_risk_gauge(eval_result["unified_risk_score"], eval_result["decision"]),
                use_container_width=True,
            )

        with res_col2:
            dec = eval_result["decision"]
            lvl = eval_result["risk_level"]
            score = eval_result["unified_risk_score"]

            if dec == "DECLINE":
                st.error(f"### 🔴 DECISION: DECLINE ({lvl})\n**Action:** Transaction immediately rejected. Critical security alert raised.")
            elif dec == "MANUAL_REVIEW":
                st.warning(f"### 🟠 DECISION: MANUAL REVIEW ({lvl})\n**Action:** Transaction held in risk operations queue for human analyst inspection.")
            elif dec == "CHALLENGE_OTP":
                st.warning(f"### 🟡 DECISION: CHALLENGE OTP ({lvl})\n**Action:** Step-up biometric or out-of-band verification required.")
            else:
                st.success(f"### 🟢 DECISION: APPROVE ({lvl})\n**Action:** Transaction approved. Standard post-authorization logging.")

            st.plotly_chart(
                render_score_breakdown_bar(eval_result["breakdown"]),
                use_container_width=True,
            )

            crit_flags = eval_result["breakdown"]["critical_flags"]
            if crit_flags:
                st.markdown("**Triggered Critical Signals:**")
                badges = " ".join([
                    f"<span style='background-color:#EF4444; color:white; padding:4px 10px; border-radius:12px; margin-right:8px; font-weight:600; font-size:12px;'>⚠️ {f}</span>"
                    for f in crit_flags
                ])
                st.markdown(badges, unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 4: Risk Engine Configuration
# ----------------------------------------------------
with tab4:
    st.subheader("⚙️ Unified Risk Engine Multi-Factor Weights")
    st.write(
        "Configure the mathematical weights governing the composite risk equation: "
        r"$$R_{\text{total}} = w_1 \cdot S_{\text{ML}} + w_2 \cdot S_{\text{Anomaly}} + w_3 \cdot S_{\text{Security}}$$"
    )

    wc1, wc2, wc3 = st.columns(3)

    with wc1:
        new_w_ml = st.slider("Supervised ML Weight (w1)", 0.0, 1.0, 0.45, 0.05)
    with wc2:
        new_w_anom = st.slider("Anomaly Detection Weight (w2)", 0.0, 1.0, 0.15, 0.05)
    with wc3:
        new_w_sec = st.slider("Behavioral/Security Rules Weight (w3)", 0.0, 1.0, 0.40, 0.05)

    tot = new_w_ml + new_w_anom + new_w_sec
    if tot > 0:
        norm_ml = new_w_ml / tot
        norm_anom = new_w_anom / tot
        norm_sec = new_w_sec / tot
        st.info(
            f"Normalized Weights: **ML: {norm_ml:.2f}** | **Anomaly: {norm_anom:.2f}** | **Security: {norm_sec:.2f}** (Sum = 1.0)"
        )

        if st.button("💾 Apply Updated Risk Weights"):
            st.session_state.risk_engine = UnifiedRiskEngine(
                w_ml=norm_ml, w_anomaly=norm_anom, w_security=norm_sec
            )
            st.success("Risk weights updated successfully for all subsequent transaction assessments!")

    st.markdown("---")
    st.markdown("#### 🛡️ Viva & Defense Reference Table")
    st.markdown(
        r"""
        | Attack Scenario | Vulnerability in Standard OTP Systems | How Our System Detects It |
        | :--- | :--- | :--- |
        | **SIM-Swap Hijacking** | OTP is intercepted on ported SIM; transaction authentication passes. | Telephony age check (< 24-48h) + amount escalation overrides OTP success. |
        | **Card-Testing Bots (Carding)** | Low-value test charges ($1-$10) go unnoticed by simple thresholds. | Sliding-window velocity tracking flags 3+ micro-charges in 5 minutes. |
        | **Impossible Travel** | Stolen credentials used overseas without physical card cloning. | Haversine distance over $\Delta t$ calculates velocity exceeding aviation limits (>850 km/h). |
        | **Credential Stuffing** | Automated brute-forcing of security codes. | Clustered failed OTP counters trigger automated auth channel lockout. |
        """
    )
