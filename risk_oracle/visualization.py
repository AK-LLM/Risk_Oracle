"""
Plotly chart helpers — one chart per logical view, all returning go.Figure.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go


def loss_distribution(samples: np.ndarray, title: str = "Impact distribution") -> go.Figure:
    nonzero = samples[samples > 0]
    if len(nonzero) == 0:
        fig = go.Figure()
        fig.update_layout(title=title + " (no impact in any simulation)")
        return fig
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=nonzero,
        nbinsx=50,
        marker_color="#4c8eda",
        opacity=0.75,
    ))
    p50 = np.median(nonzero)
    p95 = np.quantile(nonzero, 0.95)
    p99 = np.quantile(nonzero, 0.99)
    for label, val, dash in [("median", p50, "solid"), ("VaR 95%", p95, "dash"),
                              ("VaR 99%", p99, "dot")]:
        fig.add_vline(x=val, line_dash=dash,
                      annotation_text=f"{label}: ${val:,.0f}",
                      annotation_position="top",
                      line_color="#d85a30")
    fig.update_layout(
        title=title,
        xaxis_title="Impact (USD, illustrative)",
        yaxis_title="Frequency in 20k simulations",
        showlegend=False,
        bargap=0.02,
    )
    return fig


def probability_band(point: float, low: float, high: float,
                     primary_p: float, critic_p: float,
                     market_p: float = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[high - low], y=["Probability"],
        base=[low], orientation="h",
        marker_color="rgba(76, 142, 218, 0.3)",
        name="Confidence band",
        hovertemplate=f"Band: {low:.1%} – {high:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[point], y=["Probability"],
        mode="markers", marker=dict(size=20, color="#d85a30", symbol="diamond"),
        name=f"Reconciled: {point:.1%}",
    ))
    fig.add_trace(go.Scatter(
        x=[primary_p], y=["Probability"],
        mode="markers", marker=dict(size=12, color="#4c8eda", symbol="circle"),
        name=f"Primary: {primary_p:.1%}",
    ))
    fig.add_trace(go.Scatter(
        x=[critic_p], y=["Probability"],
        mode="markers", marker=dict(size=12, color="#1d9e75", symbol="circle"),
        name=f"Critic: {critic_p:.1%}",
    ))
    if market_p is not None:
        fig.add_trace(go.Scatter(
            x=[market_p], y=["Probability"],
            mode="markers", marker=dict(size=12, color="#888780", symbol="x"),
            name=f"Market: {market_p:.1%}",
        ))
    fig.update_layout(
        xaxis_range=[0, 1],
        xaxis_tickformat=".0%",
        showlegend=True,
        height=200,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig


def hazard_curve(times: np.ndarray, probs: np.ndarray,
                 horizon_label: str = "months from now") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=probs, mode="lines+markers",
        line=dict(color="#4c8eda", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(76, 142, 218, 0.15)",
    ))
    fig.update_layout(
        title="Probability of event still active over time",
        xaxis_title=horizon_label,
        yaxis_title="P(event active at time t)",
        yaxis_tickformat=".0%",
        height=320,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig


def contagion_chart(spillover: Dict[str, Dict[str, float]],
                    source_loss: float) -> go.Figure:
    cats = list(spillover.keys())
    losses = [spillover[c]["spillover_loss"] for c in cats]
    weights = [spillover[c]["weight"] for c in cats]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats,
        x=losses,
        orientation="h",
        marker_color="#7f77dd",
        text=[f"weight {w:.0%}" for w in weights],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Spillover expected loss by category (source loss: ${source_loss:,.0f})",
        xaxis_title="Expected spillover loss (USD)",
        height=350,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig
