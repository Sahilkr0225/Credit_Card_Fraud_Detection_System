"""
Alert and Risk Visualizations component using Plotly.
"""

from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_risk_gauge(score: float, decision: str) -> go.Figure:
    """Render an interactive radial risk score gauge."""
    color_map = {
        "APPROVE": "#10B981",         # Emerald Green
        "CHALLENGE_OTP": "#F59E0B",   # Amber
        "MANUAL_REVIEW": "#F97316",   # Orange
        "DECLINE": "#EF4444",         # Red
    }
    bar_color = color_map.get(decision, "#3B82F6")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Unified Risk Score ({decision})", "font": {"size": 20, "color": "#F3F4F6"}},
            number={"suffix": "/100", "font": {"size": 36, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9CA3AF"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#1F2937",
                "borderwidth": 1,
                "bordercolor": "#374151",
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [30, 65], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [65, 80], "color": "rgba(249, 115, 22, 0.25)"},
                    {"range": [80, 100], "color": "rgba(239, 68, 68, 0.30)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 4},
                    "thickness": 0.8,
                    "value": 80.0,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
    )
    return fig


def render_risk_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of unified risk scores across simulated events."""
    if df.empty or "unified_risk_score" not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x="unified_risk_score",
        nbins=25,
        color="risk_level",
        color_discrete_map={
            "LOW": "#10B981",
            "MEDIUM": "#F59E0B",
            "HIGH": "#F97316",
            "CRITICAL": "#EF4444",
        },
        title="Unified Risk Score Distribution",
        labels={"unified_risk_score": "Risk Score (0 - 100)", "count": "Transactions"},
    )

    fig.add_vline(x=30, line_dash="dash", line_color="#10B981", annotation_text="Approve Threshold")
    fig.add_vline(x=65, line_dash="dash", line_color="#F59E0B", annotation_text="Challenge Threshold")
    fig.add_vline(x=80, line_dash="dash", line_color="#EF4444", annotation_text="Decline Threshold")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=320,
    )
    return fig


def render_amount_vs_risk_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot correlating transaction amount with unified risk score."""
    if df.empty or "Amount" not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df,
        x="Amount",
        y="unified_risk_score",
        color="risk_level",
        size="Amount",
        hover_data=["transaction_id", "user_id", "merchant", "decision", "scenario_label"],
        color_discrete_map={
            "LOW": "#10B981",
            "MEDIUM": "#F59E0B",
            "HIGH": "#F97316",
            "CRITICAL": "#EF4444",
        },
        title="Transaction Amount vs. Unified Risk Score",
        labels={"Amount": "Amount ($)", "unified_risk_score": "Risk Score (0 - 100)"},
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=320,
    )
    return fig


def render_flags_breakdown(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing distribution of triggered security and ML flags."""
    if df.empty or "critical_flags" not in df.columns:
        return go.Figure()

    all_flags: List[str] = []
    for item in df["critical_flags"].dropna():
        if isinstance(item, list):
            all_flags.extend(item)
        elif isinstance(item, str) and item:
            all_flags.extend([f.strip() for f in item.split(",") if f.strip()])

    if not all_flags:
        fig = go.Figure()
        fig.add_annotation(
            text="No active flags triggered in current feed",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"color": "#9CA3AF", "size": 14},
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        return fig

    flag_counts = pd.Series(all_flags).value_counts().reset_index()
    flag_counts.columns = ["Flag", "Count"]

    fig = px.pie(
        flag_counts,
        values="Count",
        names="Flag",
        hole=0.45,
        title="Security & Rule Triggers Breakdown",
        color_discrete_sequence=px.colors.sequential.Sunsetdark,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=320,
    )
    return fig


def render_score_breakdown_bar(breakdown: Dict[str, Any]) -> go.Figure:
    """Horizontal stacked bar chart showing component contributions."""
    ml_contrib = breakdown.get("ml_contribution", 0.0)
    anomaly_contrib = breakdown.get("anomaly_contribution", 0.0)
    sec_contrib = breakdown.get("security_contribution", 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Risk Breakdown"],
        x=[ml_contrib],
        name=f"Supervised ML ({ml_contrib} pts)",
        orientation="h",
        marker={"color": "#6366F1"},
    ))
    fig.add_trace(go.Bar(
        y=["Risk Breakdown"],
        x=[anomaly_contrib],
        name=f"Anomaly Outlier ({anomaly_contrib} pts)",
        orientation="h",
        marker={"color": "#8B5CF6"},
    ))
    fig.add_trace(go.Bar(
        y=["Risk Breakdown"],
        x=[sec_contrib],
        name=f"Security Signals ({sec_contrib} pts)",
        orientation="h",
        marker={"color": "#EC4899"},
    ))

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        xaxis={"range": [0, 100], "title": "Score Contribution (Total 100)"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=140,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.1, "xanchor": "left", "x": 0},
    )
    return fig
