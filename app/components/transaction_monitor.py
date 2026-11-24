"""
Interactive Transaction Monitor and Inspection UI component.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st


def render_transaction_feed(df: pd.DataFrame) -> Optional[str]:
    """
    Renders filterable live transaction feed and returns selected transaction_id if any.
    """
    if df.empty:
        st.info("No transactions recorded in the live stream yet. Use the simulation panel to generate events.")
        return None

    # Filter Controls Bar
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 2.0])

    with col1:
        risk_options = ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        selected_risk = st.selectbox("Filter Risk Level", risk_options, index=0)

    with col2:
        decision_options = ["ALL", "APPROVE", "CHALLENGE_OTP", "MANUAL_REVIEW", "DECLINE"]
        selected_decision = st.selectbox("Filter Decision", decision_options, index=0)

    with col3:
        scenarios = ["ALL"] + sorted(df["scenario_label"].unique().tolist()) if "scenario_label" in df.columns else ["ALL"]
        selected_scenario = st.selectbox("Filter Scenario", scenarios, index=0)

    with col4:
        search_query = st.text_input("Search TXN or User ID", placeholder="e.g. TXN_ or USR_")

    # Apply filters
    filtered_df = df.copy()

    if selected_risk != "ALL":
        filtered_df = filtered_df[filtered_df["risk_level"] == selected_risk]

    if selected_decision != "ALL":
        filtered_df = filtered_df[filtered_df["decision"] == selected_decision]

    if selected_scenario != "ALL":
        filtered_df = filtered_df[filtered_df["scenario_label"] == selected_scenario]

    if search_query:
        query = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["transaction_id"].str.lower().str.contains(query, na=False)
            | filtered_df["user_id"].str.lower().str.contains(query, na=False)
        ]

    st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** total transactions")

    # Display columns
    display_cols = [
        "transaction_id",
        "user_id",
        "timestamp",
        "Amount",
        "merchant",
        "scenario_label",
        "unified_risk_score",
        "risk_level",
        "decision",
    ]
    present_cols = [c for c in display_cols if c in filtered_df.columns]

    # Render formatted table
    st.dataframe(
        filtered_df[present_cols].sort_values(by="unified_risk_score", ascending=False),
        column_config={
            "Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
            "unified_risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                help="Composite Unified Risk Score (0-100)",
                format="%d",
                min_value=0,
                max_value=100,
            ),
            "risk_level": st.column_config.TextColumn("Risk Level"),
            "decision": st.column_config.TextColumn("Decision"),
            "scenario_label": st.column_config.TextColumn("Scenario Profile"),
        },
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    # Export CSV button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Stream to CSV",
        data=csv_bytes,
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )

    # Transaction selection dropdown for deep inspection
    tx_ids = filtered_df["transaction_id"].tolist()
    if tx_ids:
        selected_tx_id = st.selectbox(
            "Select Transaction for In-Depth Telemetry Inspection:",
            options=tx_ids,
            index=0,
        )
        return selected_tx_id

    return None


def render_transaction_details(tx_row: pd.Series) -> None:
    """
    Renders detailed breakdown for an individual selected transaction.
    """
    st.markdown("---")
    st.subheader(f"🔍 Deep Telemetry Inspection: `{tx_row.get('transaction_id')}`")

    # Status Pill Banner
    decision = tx_row.get("decision", "UNKNOWN")
    risk_level = tx_row.get("risk_level", "UNKNOWN")
    score = tx_row.get("unified_risk_score", 0.0)

    if decision == "DECLINE":
        st.error(f"🔴 **DECISION: {decision}** | Severity: **{risk_level}** | Score: **{score:.1f}/100**")
    elif decision == "MANUAL_REVIEW":
        st.warning(f"🟠 **DECISION: {decision}** | Severity: **{risk_level}** | Score: **{score:.1f}/100**")
    elif decision == "CHALLENGE_OTP":
        st.warning(f"🟡 **DECISION: {decision}** | Severity: **{risk_level}** | Score: **{score:.1f}/100**")
    else:
        st.success(f"🟢 **DECISION: {decision}** | Severity: **{risk_level}** | Score: **{score:.1f}/100**")

    # 4 Telemetry Columns
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("#### 🤖 ML Intelligence")
        ml_prob = tx_row.get("ml_fraud_probability", 0.0)
        st.metric("Fraud Probability", f"{ml_prob * 100:.1f}%")
        st.caption(f"ML Contribution: **{tx_row.get('ml_contribution', 0.0):.1f} pts**")
        st.caption(f"Anomaly Outlier: **{tx_row.get('is_anomaly', False)}**")

    with c2:
        st.markdown("#### 📱 Telephony & SIM")
        sim_swapped = tx_row.get("sim_swap_detected", False)
        sim_age = tx_row.get("sim_swap_age_hours", 999.0)
        st.metric("SIM Swapped?", "YES ⚠️" if sim_swapped else "NO")
        st.caption(f"SIM Swap Age: **{sim_age:.1f} hours**")
        st.caption(f"OTP Verified: **{tx_row.get('otp_verified', True)}**")

    with c3:
        st.markdown("#### ⚡ Velocity & Carding")
        vel_score = tx_row.get("velocity_score", 0.0)
        carding = tx_row.get("carding_flag", False)
        st.metric("Velocity Score", f"{vel_score:.2f}")
        st.caption(f"Carding Attack: **{'DETECTED 🚨' if carding else 'No'}**")
        st.caption(f"5m Rolling Count: **{tx_row.get('tx_count_5m', 1)} txns**")

    with c4:
        st.markdown("#### 📍 Device & Geo")
        speed = tx_row.get("implied_speed_kmh", 0.0)
        impossible = tx_row.get("impossible_travel_detected", False)
        st.metric("Geo Speed", f"{speed:.0f} km/h")
        st.caption(f"Impossible Travel: **{'DETECTED 🚨' if impossible else 'No'}**")
        st.caption(f"Location: **{tx_row.get('city', 'Unknown')}, {tx_row.get('country', 'Unknown')}**")

    # Critical Flags List
    flags = tx_row.get("critical_flags")
    if isinstance(flags, str) and flags:
        flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    elif isinstance(flags, list):
        flag_list = flags
    else:
        flag_list = []

    if flag_list:
        st.markdown("##### 🚩 Active Risk Signals & Triggered Rules:")
        badge_html = " ".join([
            f"<span style='background-color:#EF4444; color:white; padding:4px 10px; border-radius:12px; margin-right:8px; font-weight:600; font-size:12px;'>⚠️ {f}</span>"
            for f in flag_list
        ])
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        st.caption("No critical security or rule triggers activated.")
