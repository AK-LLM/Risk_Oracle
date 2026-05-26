"""
Risk Oracle — Streamlit entrypoint (v2 with portfolio, evidence assistant,
explanation layer, watchlist, comparison view).
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from risk_oracle import models as models_mod
from risk_oracle import router as router_mod
from risk_oracle import pipeline as pipe_mod
from risk_oracle import calibration as cal_mod
from risk_oracle import decision as decision_mod
from risk_oracle import portfolio as pf_mod
from risk_oracle import watchlist as wl_mod
from risk_oracle import evidence_assistant as evi_mod
from risk_oracle import explanation as exp_mod
from risk_oracle import polymarket as pm_mod
from risk_oracle import bet_tracker as bet_mod
from risk_oracle import visualization as viz
from risk_oracle.taxonomy import TAXONOMY, get_category, CATEGORY_KEYS


st.set_page_config(
    page_title="Risk Oracle",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_secrets() -> Dict[str, str]:
    secrets: Dict[str, str] = {}
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FRED_API_KEY",
                "ACLED_API_KEY", "ACLED_EMAIL", "EXTERNAL_DB_URL"):
        try:
            val = st.secrets.get(key, "")
        except Exception:
            val = ""
        if not val:
            val = os.environ.get(key, "")
        secrets[key] = val
    return secrets


def render_sidebar(secrets: Dict[str, str]):
    st.sidebar.title("Risk Oracle")
    st.sidebar.caption("Probabilistic risk modeling with calibration feedback")

    st.sidebar.subheader("Status")
    rows = [
        ("Smart router (LLM)",
         "✓ active" if (secrets["ANTHROPIC_API_KEY"] or secrets["OPENAI_API_KEY"])
         else "○ rule-based fallback"),
        ("Evidence assistant",
         "✓ active" if (secrets["ANTHROPIC_API_KEY"] or secrets["OPENAI_API_KEY"])
         else "○ needs LLM key"),
        ("Explanation layer",
         "✓ active" if (secrets["ANTHROPIC_API_KEY"] or secrets["OPENAI_API_KEY"])
         else "○ template fallback"),
        ("FRED macro data", "✓ active" if secrets["FRED_API_KEY"] else "○ skipped"),
        ("ACLED conflict data", "✓ active" if secrets["ACLED_API_KEY"] else "○ skipped"),
        ("GDELT news", "✓ active (free)"),
        ("Polymarket / Manifold / Metaculus", "✓ active (free)"),
        ("World Bank / USGS / NOAA", "✓ active (free)"),
    ]
    for label, status in rows:
        st.sidebar.text(f"{status}   {label}")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**No API keys?** Core forecasting still works using only free sources. "
        "Add `ANTHROPIC_API_KEY` for smart routing, evidence elicitation, and explanations."
    )


# =============================================================================
# Session state helpers
# =============================================================================

def _init_session_state():
    if "trigger_text" not in st.session_state:
        st.session_state.trigger_text = ""
    if "situation_text" not in st.session_state:
        st.session_state.situation_text = ""
    if "evidence_factors" not in st.session_state:
        st.session_state.evidence_factors = []
    if "extracted_evidence" not in st.session_state:
        st.session_state.extracted_evidence = None
    if "last_forecast" not in st.session_state:
        st.session_state.last_forecast = None


def _set_example(text: str):
    st.session_state.trigger_text = text


def _set_evidence_from_extraction(extracted: list):
    st.session_state.evidence_factors = [
        {
            "name": e.name,
            "likelihood_ratio": e.likelihood_ratio,
            "confidence": e.confidence,
        }
        for e in extracted
    ]
    st.session_state.extracted_evidence = extracted


# =============================================================================
# Main
# =============================================================================

def main():
    _init_session_state()
    secrets = _get_secrets()
    render_sidebar(secrets)

    st.title("Risk Oracle")
    st.markdown(
        "Smart router → primary + critic models → reconciliation → OSINT verification → "
        "calibrated probability with explanation, portfolio context, and external comparison."
    )

    tabs = st.tabs([
        "🎯 New forecast",
        "🟣 Polymarket",
        "💰 Bets",
        "👁 Watchlist",
        "💼 Portfolio",
        "📋 History",
        "📊 Calibration",
        "ℹ About",
    ])

    with tabs[0]:
        render_forecast_tab(secrets)
    with tabs[1]:
        render_polymarket_tab(secrets)
    with tabs[2]:
        render_bets_tab(secrets)
    with tabs[3]:
        render_watchlist_tab(secrets)
    with tabs[4]:
        render_portfolio_tab()
    with tabs[5]:
        render_history_tab()
    with tabs[6]:
        render_calibration_tab()
    with tabs[7]:
        render_about_tab()


# =============================================================================
# Forecast tab
# =============================================================================

def render_forecast_tab(secrets: Dict[str, str]):
    st.subheader("Run a new forecast")

    # ----- Trigger input -----
    trigger = st.text_area(
        "Trigger question — be specific and date-bound",
        height=80,
        placeholder="e.g., 'Will the S&P 500 close more than 20% below its current level at any point in 2026?'",
        key="trigger_text",
        help="Vague questions can't be scored. Include a specific event and a specific date.",
    )

    with st.expander("📋 Or click to use a preset example", expanded=False):
        examples = [
            "Will direct Iran-Israel strikes still be occurring on October 1, 2026?",
            "Will the US enter a recession (two consecutive negative GDP quarters) by Q4 2026?",
            "Will a major H5N1 outbreak (>1000 human cases in one country) be declared by WHO before end of 2026?",
            "Will a Cat 4+ hurricane make US landfall during the 2026 Atlantic season?",
            "Will the S&P 500 close more than 20% below its current level at any point in 2026?",
            "Will the Bank of Canada raise its overnight policy rate at its next scheduled meeting?",
        ]
        for i, ex in enumerate(examples):
            st.button(ex, key=f"ex_btn_{i}", on_click=_set_example, args=(ex,),
                      use_container_width=True)

    # ----- Routing preview -----
    if trigger.strip():
        with st.expander("Router preview (classification before you run)", expanded=False):
            decision = router_mod.route(trigger, secrets=secrets)
            col1, col2, col3 = st.columns(3)
            col1.metric("Primary category", get_category(decision.primary_category).name)
            col2.metric("Router confidence", f"{decision.confidence:.0%}")
            col3.metric("Mode", "LLM" if decision.used_llm else "Rule-based")
            if decision.is_anomaly:
                st.warning("Black-swan / anomaly flag: " + "; ".join(decision.anomaly_reasons))
            if decision.secondary_categories:
                st.info("Cross-category spillover candidates: " +
                        ", ".join(get_category(c).name for c in decision.secondary_categories))
            st.caption(decision.reasoning)
    else:
        decision = None

    # ----- Evidence assistant (LLM) -----
    st.markdown("### Evidence")
    has_llm = bool(secrets["ANTHROPIC_API_KEY"] or secrets["OPENAI_API_KEY"])

    if has_llm:
        st.markdown(
            "**Option A — describe the situation in plain English, let the assistant extract evidence factors.**"
        )
        situation = st.text_area(
            "What do you know about the current situation?",
            height=100,
            placeholder="e.g., 'The Fed has signaled it's done hiking but Powell's tone last week was hawkish. Unemployment ticked up to 4.2% from 4.0%. Housing starts collapsed. Yield curve still inverted but flattening.'",
            key="situation_text",
        )
        if st.button("🪄 Extract evidence factors", disabled=not (situation.strip() and trigger.strip())):
            if decision is None:
                decision = router_mod.route(trigger, secrets=secrets)
            with st.spinner("Asking the assistant…"):
                extracted = evi_mod.extract_evidence(
                    trigger=trigger,
                    situation=situation,
                    category=decision.primary_category,
                    secrets=secrets,
                )
            if extracted:
                _set_evidence_from_extraction(extracted)
                st.success(f"Extracted {len(extracted)} evidence factors. Review and edit below.")
                st.rerun()
            else:
                st.error("Couldn't extract evidence. Try a more detailed description, or fill in evidence manually below.")
    else:
        st.info("Set `ANTHROPIC_API_KEY` to enable the assistant. Manual entry below.")

    st.markdown("**Prior probability and evidence factors** — adjust as needed.")
    if decision is None and trigger.strip():
        decision = router_mod.route(trigger, secrets=secrets)
    if decision is not None:
        spec = decision.primary_spec
        prior_default = float(spec.base_rate_default)
        ref_note = spec.reference_class_note
    else:
        prior_default = 0.5
        ref_note = "Run a trigger to anchor the prior to a reference class."

    prior = st.slider(
        "Prior probability (base rate before specific evidence)",
        min_value=0.01, max_value=0.99, step=0.01, value=prior_default,
    )
    st.caption(f"Reference class: {ref_note}")

    # ----- Render evidence editor -----
    evidence_factors = st.session_state.evidence_factors or [
        {"name": "Active signal supporting event", "likelihood_ratio": 2.0, "confidence": 0.7},
        {"name": "Counter-signal toward resolution", "likelihood_ratio": 0.6, "confidence": 0.7},
        {"name": "Structural/institutional incentive", "likelihood_ratio": 1.5, "confidence": 0.7},
    ]
    new_factors = []
    n = st.number_input("Number of evidence factors", min_value=0, max_value=10,
                        value=len(evidence_factors))
    # Extend with empty defaults if user increased count
    while len(evidence_factors) < n:
        evidence_factors.append(
            {"name": "New factor", "likelihood_ratio": 1.0, "confidence": 0.7}
        )
    for i in range(int(n)):
        ev = evidence_factors[i]
        cols = st.columns([3, 2, 1])
        name = cols[0].text_input(f"Factor {i+1}", value=ev.get("name", ""), key=f"ev_n_{i}")
        lr = cols[1].number_input(
            "Likelihood ratio", min_value=0.05, max_value=20.0, step=0.1,
            value=float(ev.get("likelihood_ratio", 1.0)), key=f"ev_lr_{i}",
        )
        conf = cols[2].slider(
            "Conf.", min_value=0.0, max_value=1.0, step=0.05,
            value=float(ev.get("confidence", 0.7)), key=f"ev_c_{i}",
        )
        # Show rationale if from extraction
        if (st.session_state.extracted_evidence
                and i < len(st.session_state.extracted_evidence)
                and name == st.session_state.extracted_evidence[i].name):
            st.caption("💡 " + st.session_state.extracted_evidence[i].rationale)
        new_factors.append({"name": name, "likelihood_ratio": lr, "confidence": conf})
    st.session_state.evidence_factors = new_factors

    # ----- Run button -----
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_button = st.button("Run forecast", type="primary",
                               disabled=not trigger.strip(),
                               use_container_width=True)
    with col2:
        expected_resolution = st.date_input(
            "Expected resolution date",
            value=datetime.utcnow().date() + timedelta(days=180),
        )
    with col3:
        save_to_watchlist = st.checkbox("Save to watchlist after running", value=False)

    if not run_button:
        return

    # ----- Run the pipeline -----
    if decision is None:
        decision = router_mod.route(trigger, secrets=secrets)

    ev_objects = [
        models_mod.EvidenceFactor(
            name=f["name"], likelihood_ratio=f["likelihood_ratio"],
            confidence=f["confidence"],
        ) for f in new_factors
    ]

    with st.spinner("Running primary + critic models, OSINT verification, and market comparison…"):
        fr = pipe_mod.run_forecast(
            trigger=trigger,
            category=decision.primary_category,
            prior=prior,
            evidence=ev_objects,
            secrets=secrets,
        )
    st.session_state.last_forecast = fr

    # Persist as a logged prediction
    pred_id = cal_mod.log_prediction(
        cal_mod.Prediction(
            trigger=trigger,
            category=decision.primary_category,
            primary_p=fr.primary_p,
            critic_p=fr.critic_p,
            reconciled_p=fr.point_p,
            band_low=fr.band_low,
            band_high=fr.band_high,
            expected_resolution=str(expected_resolution),
            metadata={"market_prob": fr.market_prob, "tail": fr.tail_metrics},
        ),
    )

    if save_to_watchlist:
        wl_id = wl_mod.add_item(wl_mod.WatchlistItem(
            trigger=trigger,
            category=decision.primary_category,
            prior=prior,
            evidence=new_factors,
            alert_threshold=0.05,
        ))
        wl_mod.record_refresh(wl_id, fr.point_p, fr.band_low, fr.band_high, fr.market_prob)
        st.info(f"Added to watchlist (id={wl_id}). Refresh from the Watchlist tab.")

    _render_forecast_result(fr, decision, secrets, pred_id)


def _render_forecast_result(fr: pipe_mod.ForecastResult, decision, secrets: Dict[str, str],
                            pred_id: int):
    spec = get_category(fr.category)

    st.divider()
    st.markdown("### Output")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reconciled P", f"{fr.point_p:.1%}")
    col2.metric("Band low", f"{fr.band_low:.1%}")
    col3.metric("Band high", f"{fr.band_high:.1%}")
    if fr.market_prob is not None:
        col4.metric("Markets avg", f"{fr.market_prob:.1%}",
                    delta=f"{(fr.point_p - fr.market_prob):+.1%}")
    else:
        col4.metric("Markets avg", "—")

    st.plotly_chart(
        viz.probability_band(
            fr.point_p, fr.band_low, fr.band_high,
            fr.primary_p, fr.critic_p, market_p=fr.market_prob,
        ),
        use_container_width=True,
    )

    for note in fr.notes:
        st.info(note)

    # ----- Explanation -----
    st.markdown("### Explanation")
    with st.spinner("Generating explanation…"):
        explanation = exp_mod.explain(
            pipe_mod.forecast_to_explanation_state(fr),
            secrets=secrets,
        )
    st.markdown(explanation)

    # ----- Comparison view -----
    st.markdown("### Comparison against external markets")
    if fr.comparison and fr.comparison.items:
        rows = []
        for it in fr.comparison.items:
            rows.append({
                "Source": it.source,
                "Probability": f"{it.probability:.1%}" if it.probability is not None else "—",
                "Δ vs us": f"{(it.probability - fr.point_p):+.1%}" if it.probability is not None else "—",
                "Matched questions": (it.question_matched[:120] + "…")
                                     if len(it.question_matched) > 120
                                     else it.question_matched or "—",
                "Note": it.note,
            })
        rows.insert(0, {
            "Source": "Our reconciled",
            "Probability": f"{fr.point_p:.1%}",
            "Δ vs us": "—",
            "Matched questions": "—",
            "Note": f"primary {fr.primary_p:.1%}, critic {fr.critic_p:.1%}",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        md = fr.comparison.max_disagreement()
        if md is not None and md > 0.15:
            st.warning(
                f"You disagree with the market consensus by {md:.1%}. "
                f"That's a strong signal — either you see something they don't, "
                f"or your evidence is mis-weighted. Worth a second pass."
            )

    # ----- Portfolio context -----
    st.markdown("### Portfolio context")
    exposure, contributing = pf_mod.exposure_by_category(fr.category)
    positions = pf_mod.list_positions()
    if not positions:
        st.caption("No positions in your portfolio. Add some on the Portfolio tab to see "
                   "category-specific exposure here.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Net exposure to category", f"${exposure:,.0f}")
        # Drawdown estimates — derive from tail risk relative to a reference asset value of $1M
        expected_dd = min(40, 100 * fr.tail_metrics["expected_loss"] / 1e9) if fr.tail_metrics["expected_loss"] else 5
        tail_dd = min(60, 100 * fr.tail_metrics["VaR_99"] / 1e9) if fr.tail_metrics["VaR_99"] else 12
        loss = pf_mod.estimate_loss_given_event(
            exposure_usd=max(0, exposure),
            expected_drawdown_pct=expected_dd,
            p99_drawdown_pct=tail_dd,
        )
        col2.metric("Expected loss (typical)", f"${loss['expected_loss']:,.0f}")
        col3.metric("Tail loss (99th pct)", f"${loss['tail_loss']:,.0f}")

        if contributing:
            with st.expander(f"Positions contributing to this category ({len(contributing)})"):
                rows = []
                for p in contributing:
                    sens = p.category_sensitivities.get(fr.category, 0.0)
                    rows.append({
                        "Ticker": p.ticker,
                        "Notional": f"${p.notional_usd:,.0f}",
                        "Direction": p.direction,
                        "Sensitivity": f"{sens:+.1%}",
                        "Effective exposure": f"${p.signed_notional() * sens:,.0f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Hedge ideas (directional only, not advice):**")
        for h in pf_mod.hedge_ideas(fr.category):
            st.markdown(f"- {h}")

    # ----- Polymarket edge analysis -----
    st.markdown("### 🟣 Polymarket edge analysis")
    with st.spinner("Searching Polymarket for matching markets…"):
        pm_matches = pm_mod.search_markets(fr.trigger, limit=5)

    if not pm_matches:
        st.caption(
            "No matching Polymarket markets found. Try the Polymarket tab to browse "
            "the most active markets and pick one as a trigger."
        )
    else:
        st.caption(f"Found {len(pm_matches)} matching market(s). Edge analysis below.")
        # Bankroll setting via session state
        if "pm_bankroll" not in st.session_state:
            st.session_state.pm_bankroll = 10_000.0
        c1, c2 = st.columns([1, 3])
        with c1:
            bankroll = st.number_input(
                "Bankroll USD", min_value=100.0, step=500.0,
                value=float(st.session_state.pm_bankroll),
                key="pm_bankroll_input",
            )
            st.session_state.pm_bankroll = bankroll

        for i, m in enumerate(pm_matches):
            if m.yes_price is None:
                continue
            rec = decision_mod.polymarket_recommendation(
                our_probability=fr.point_p,
                band_low=fr.band_low,
                band_high=fr.band_high,
                market_yes_price=m.yes_price,
                bankroll_usd=bankroll,
                available_liquidity_usd=max(50.0, m.liquidity_usd),
            )
            with st.container(border=True):
                st.markdown(f"**{m.question}**")
                st.caption(
                    f"vol 24h ${m.volume_24h_usd:,.0f} · liq ${m.liquidity_usd:,.0f}"
                    + (f" · closes in {m.days_until_close}d" if m.days_until_close is not None else "")
                    + f" · [open on polymarket.com →]({m.url})"
                )
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Market YES", f"{m.yes_price:.1%}")
                c2.metric("Our P", f"{fr.point_p:.1%}")
                c3.metric("Edge", f"{rec.edge:.1%}",
                          delta=rec.side if rec.side != "PASS" else None,
                          delta_color="normal" if rec.side != "PASS" else "off")
                c4.metric("EV per $1", f"${rec.expected_value_per_dollar:+.2f}")
                c5.metric("Recommended size", f"${rec.recommended_size_usd:,.0f}")
                if rec.side != "PASS":
                    log_key = f"log_bet_{m.id}_{i}"
                    if st.button(
                        f"Log {rec.side} bet at ${rec.recommended_size_usd:,.0f}",
                        key=log_key,
                        type="secondary",
                    ):
                        bet_id = bet_mod.log_bet(bet_mod.Bet(
                            placed_at=datetime.utcnow().isoformat(),
                            market_id=m.id, market_question=m.question,
                            market_url=m.url, side=rec.side,
                            entry_price=m.yes_price if rec.side == "YES" else (1 - m.yes_price),
                            size_usd=rec.recommended_size_usd,
                            our_probability=fr.point_p,
                            edge_at_entry=rec.edge,
                            expected_value_usd=rec.recommended_size_usd * rec.expected_value_per_dollar,
                            forecast_id=pred_id,
                            notes="",
                        ))
                        st.success(f"Bet #{bet_id} logged. See Bets tab.")
                for note in rec.notes:
                    st.caption("• " + note)
                if rec.liquidity_warning:
                    st.warning(
                        "Liquidity warning — slippage will be material. "
                        "Stack the order or split across sessions."
                    )

    # ----- Tail risk -----
    st.markdown("### Tail-explicit risk (illustrative $ units)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected loss", f"${fr.tail_metrics['expected_loss']:,.0f}")
    col2.metric("VaR 95%", f"${fr.tail_metrics['VaR_95']:,.0f}")
    col3.metric("VaR 99%", f"${fr.tail_metrics['VaR_99']:,.0f}")
    col4.metric("Worst 1% mean", f"${fr.tail_metrics['worst_1pct_mean']:,.0f}")
    st.plotly_chart(viz.loss_distribution(fr.impact_samples, "Loss distribution"),
                    use_container_width=True)

    # ----- Time dynamics -----
    times, p_active = reconcile_time_curve(fr.point_p, spec.typical_duration_months)
    st.markdown("### Time dynamics")
    st.plotly_chart(viz.hazard_curve(times, p_active), use_container_width=True)

    # ----- Cross-category contagion -----
    if fr.contagion_spillover:
        st.markdown("### Cross-category contagion")
        st.plotly_chart(
            viz.contagion_chart(fr.contagion_spillover, fr.tail_metrics["expected_loss"]),
            use_container_width=True,
        )

    # ----- OSINT details -----
    with st.expander("OSINT verification signals (details)"):
        if not fr.osint_bundle.signals:
            st.caption("No OSINT signals returned.")
        else:
            for s in fr.osint_bundle.signals:
                if s.error:
                    st.text(f"○ {s.source} ({s.label}): error — {s.error}")
                elif s.value is None:
                    st.text(f"○ {s.source} ({s.label}): no data")
                else:
                    st.text(f"✓ {s.source} ({s.label}): {s.interpretation}")

    # ----- Decision recommendation -----
    st.markdown("### Decision recommendation (Kelly, generic)")
    rec = decision_mod.kelly_recommendation(fr.point_p, fr.band_low, fr.band_high)
    col1, col2, col3 = st.columns(3)
    col1.metric("Full Kelly", f"{rec.kelly_fraction:+.1%}")
    col2.metric("Quarter Kelly", f"{rec.fractional_kelly:+.1%}")
    col3.metric("Label", rec.confidence_label)
    for note in rec.notes:
        st.caption("• " + note)

    st.success(f"✓ Forecast logged (id={pred_id}). Resolve it later on the History tab.")


def reconcile_time_curve(point_p, duration_mean):
    from risk_oracle import reconcile as r
    return r.time_hazard_surface(point_p, duration_mean, horizon_months=36)


# =============================================================================
# Watchlist tab
# =============================================================================

def render_watchlist_tab(secrets: Dict[str, str]):
    st.subheader("Watchlist")
    st.caption(
        "Ongoing triggers you want to track. Click Refresh to re-run the full pipeline "
        "for an item; the system records movement and flags alerts when reconciled "
        "probability moves more than your threshold."
    )

    items = wl_mod.list_items()
    if not items:
        st.info(
            "No watchlist items yet. Run a forecast and check 'Save to watchlist after running' "
            "to add one."
        )
        return

    # Summary table
    rows = []
    for it in items:
        movement = it.movement()
        rows.append({
            "id": it.id,
            "Category": get_category(it.category).name if it.category in TAXONOMY else it.category,
            "Trigger": it.trigger[:90],
            "Last P": f"{it.last_probability:.1%}" if it.last_probability is not None else "—",
            "Band": (f"[{it.last_band_low:.0%}, {it.last_band_high:.0%}]"
                     if it.last_band_low is not None else "—"),
            "Market": f"{it.last_market_prob:.1%}" if it.last_market_prob is not None else "—",
            "Δ vs prev": f"{movement:+.1%}" if movement is not None else "—",
            "Alert": "🔔" if it.has_alert() else "",
            "Last refresh": (it.last_refreshed_at.split("T")[0]
                             if it.last_refreshed_at else "never"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        choice = st.selectbox(
            "Pick a watchlist item",
            options=[f"#{it.id} — {it.trigger[:70]}" for it in items],
        )
        chosen = items[[f"#{it.id} — {it.trigger[:70]}" for it in items].index(choice)]
    with col2:
        if st.button("🔄 Refresh selected", use_container_width=True):
            _refresh_watchlist_item(chosen, secrets)
            st.rerun()
    with col3:
        if st.button("🗑 Remove selected", use_container_width=True):
            wl_mod.remove_item(chosen.id)
            st.rerun()

    if st.button("🔄 Refresh ALL watchlist items"):
        progress = st.progress(0.0, text="Refreshing…")
        for i, it in enumerate(items):
            _refresh_watchlist_item(it, secrets)
            progress.progress((i + 1) / len(items),
                              text=f"Refreshed #{it.id} ({i+1}/{len(items)})")
        st.success("All items refreshed.")
        st.rerun()

    # History chart for chosen item
    hist = wl_mod.history(chosen.id)
    if hist:
        st.markdown(f"### History for watchlist item #{chosen.id}")
        df = pd.DataFrame(hist)
        df["refreshed_at"] = pd.to_datetime(df["refreshed_at"])
        st.line_chart(df.set_index("refreshed_at")[["probability"]])
        with st.expander("Raw history rows"):
            st.dataframe(df, use_container_width=True, hide_index=True)


def _refresh_watchlist_item(it: wl_mod.WatchlistItem, secrets: Dict[str, str]):
    ev = [models_mod.EvidenceFactor(
        name=f["name"], likelihood_ratio=f["likelihood_ratio"], confidence=f["confidence"],
    ) for f in it.evidence]
    fr = pipe_mod.run_forecast(
        trigger=it.trigger, category=it.category, prior=it.prior,
        evidence=ev, secrets=secrets, n_sims=10_000,
    )
    wl_mod.record_refresh(it.id, fr.point_p, fr.band_low, fr.band_high, fr.market_prob)


# =============================================================================
# Polymarket tab
# =============================================================================

def render_polymarket_tab(secrets: Dict[str, str]):
    st.subheader("Polymarket — browse and use as triggers")
    st.caption(
        "Browse the most active prediction markets. Click 'Use as trigger' on "
        "any market to pre-fill the New Forecast tab. The system will run its "
        "model and compute your edge vs. the live market price."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        sort_by = st.selectbox("Sort by", ["volume", "liquidity"], index=0)
    with col2:
        limit = st.slider("How many markets", min_value=5, max_value=50, value=20)

    with st.spinner("Fetching live Polymarket markets…"):
        markets = pm_mod.top_markets(limit=limit, sort_by=sort_by)

    if not markets:
        st.warning("Couldn't reach Polymarket Gamma API. Try again in a moment.")
        return

    st.caption(f"Showing top {len(markets)} live markets sorted by {sort_by}.")

    for m in markets:
        if m.yes_price is None:
            continue
        with st.container(border=True):
            cols = st.columns([4, 1, 1, 1, 1])
            with cols[0]:
                st.markdown(f"**{m.question}**")
                meta_bits = [
                    f"YES ${m.yes_price:.2f}",
                    f"vol24h ${m.volume_24h_usd:,.0f}",
                    f"liq ${m.liquidity_usd:,.0f}",
                ]
                if m.days_until_close is not None:
                    meta_bits.append(f"closes in {m.days_until_close}d")
                if m.category:
                    meta_bits.append(f"#{m.category}")
                st.caption(" · ".join(meta_bits) + f"  ·  [polymarket.com →]({m.url})")
            cols[1].metric("YES", f"{m.yes_price:.0%}")
            cols[2].metric("Vol24h", f"${m.volume_24h_usd/1000:,.0f}k")
            cols[3].metric("Liq", f"${m.liquidity_usd/1000:,.0f}k")
            with cols[4]:
                if st.button("Use as trigger", key=f"pm_use_{m.id}"):
                    st.session_state.trigger_text = m.question
                    st.success("Loaded into New Forecast tab.")


# =============================================================================
# Bets tab
# =============================================================================

def render_bets_tab(secrets: Dict[str, str]):
    st.subheader("Bets ledger")
    st.caption(
        "All Polymarket bets you've logged from the forecast tab. Resolve them "
        "when the market closes to track P&L, win rate, and edge realization."
    )

    s = bet_mod.summary()
    if s["total_bets"] == 0:
        st.info(
            "No bets logged yet. Run a forecast on the New Forecast tab — when a "
            "matching Polymarket market exists with positive edge, you'll get a "
            "'Log bet' button that records it here."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open / Closed", f"{s['open_bets']} / {s['closed_bets']}")
    c2.metric("Realized P&L", f"${s['realized_pnl_usd']:+,.0f}",
              delta=f"{s['roi']*100:+.1f}% ROI" if s['roi'] is not None else None)
    c3.metric("Win rate", f"{s['win_rate']:.0%}" if s['win_rate'] is not None else "—",
              delta=f"vs expected {s['expected_win_rate']:.0%}"
                    if s['expected_win_rate'] is not None else None)
    if s["calibration_gap"] is not None:
        gap = s["calibration_gap"]
        c4.metric("Calibration gap", f"{gap:+.1%}",
                  help="Actual win rate minus expected win rate. Positive = under-confident; "
                       "negative = over-confident.")
    else:
        c4.metric("Calibration gap", "—")

    st.divider()
    st.markdown("### Open bets")
    open_b = bet_mod.list_bets(only_open=True)
    if not open_b:
        st.caption("No open bets.")
    else:
        rows = []
        for b in open_b:
            rows.append({
                "id": b["id"], "Market": b["market_question"][:80],
                "Side": b["side"], "Entry": f"${b['entry_price']:.2f}",
                "Size": f"${b['size_usd']:,.0f}",
                "Our P": f"{b['our_probability']:.0%}",
                "Edge at entry": f"{b['edge_at_entry']:.1%}",
                "Placed": b["placed_at"][:10],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Resolve a bet**")
        choices = {f'#{b["id"]} — {b["market_question"][:60]} ({b["side"]})': b["id"]
                   for b in open_b}
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            choice = st.selectbox("Pick a bet", list(choices.keys()))
        with c2:
            outcome = st.radio("Outcome", ["YES", "NO"], horizontal=True)
        with c3:
            if st.button("Resolve"):
                bet_mod.resolve_bet(choices[choice], outcome)
                st.success("Resolved and P&L computed.")
                st.rerun()

    st.divider()
    st.markdown("### Closed bets (recent)")
    closed = [b for b in bet_mod.list_bets() if b["resolved"] == 1][:50]
    if not closed:
        st.caption("No closed bets yet.")
    else:
        rows = []
        for b in closed:
            won = b["side"] == b["outcome"]
            rows.append({
                "id": b["id"], "Market": b["market_question"][:80],
                "Side / Outcome": f"{b['side']} → {b['outcome']}",
                "Result": "✓ win" if won else "✗ loss",
                "Size": f"${b['size_usd']:,.0f}",
                "P&L": f"${b['pnl_usd']:+,.0f}" if b['pnl_usd'] is not None else "—",
                "Edge at entry": f"{b['edge_at_entry']:.1%}",
                "Closed": (b['closed_at'] or "")[:10],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    csv = pd.DataFrame(bet_mod.list_bets()).to_csv(index=False)
    st.download_button("Download bet history as CSV", csv.encode("utf-8"),
                       "risk_oracle_bets.csv", mime="text/csv")


# =============================================================================
# Portfolio tab
# =============================================================================

def render_portfolio_tab():
    st.subheader("Portfolio")
    st.caption(
        "Tag each position with its sensitivity to each risk category (0 = unaffected, "
        "1 = fully exposed, negative = hedged). Forecasts then show the dollar exposure "
        "and estimated loss for your specific positions."
    )

    positions = pf_mod.list_positions()
    if positions:
        rows = []
        for p in positions:
            rows.append({
                "id": p.id, "Ticker": p.ticker, "Description": p.description,
                "Notional": f"${p.notional_usd:,.0f}", "Direction": p.direction,
                **{f"sens.{k[:6]}": f"{p.category_sensitivities.get(k, 0):+.0%}"
                   for k in CATEGORY_KEYS},
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            remove_id = st.number_input("Remove position by id", min_value=0, value=0, step=1)
            if st.button("Remove") and remove_id > 0:
                pf_mod.remove_position(int(remove_id))
                st.rerun()
    else:
        st.info("No positions yet. Add one below.")

    st.divider()
    st.markdown("### Add a position")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("Ticker / symbol", placeholder="XOM")
    with col2:
        notional = st.number_input("Notional USD", min_value=0.0, step=1000.0, value=10000.0)
    with col3:
        direction = st.selectbox("Direction", ["long", "short"])
    with col4:
        asset_class = st.selectbox(
            "Asset class hint (sets default sensitivities)",
            ["custom", "equity_broad", "equity_tech", "equity_energy",
             "equity_financial", "equity_health", "equity_consumer",
             "bond_treasury", "bond_corporate",
             "commodity_oil", "commodity_gold", "crypto", "fx_usd", "cash"],
        )
    description = st.text_input("Description (optional)", placeholder="Energy giant")

    default_sens = pf_mod.suggest_sensitivities(asset_class)
    st.markdown("**Category sensitivities** — how exposed is this position to each risk category?")
    sens_cols = st.columns(4)
    sensitivities: Dict[str, float] = {}
    for i, key in enumerate(CATEGORY_KEYS):
        with sens_cols[i % 4]:
            sensitivities[key] = st.slider(
                get_category(key).name,
                min_value=-1.0, max_value=1.0, step=0.05,
                value=float(default_sens.get(key, 0.0)),
                key=f"sens_{key}",
            )
    if st.button("Add position", type="primary", disabled=not ticker.strip()):
        pf_mod.add_position(pf_mod.Position(
            ticker=ticker.strip().upper(), description=description,
            notional_usd=notional, direction=direction,
            category_sensitivities={k: v for k, v in sensitivities.items() if abs(v) > 1e-6},
        ))
        st.success(f"Added {ticker}.")
        st.rerun()


# =============================================================================
# History tab
# =============================================================================

def render_history_tab():
    st.subheader("Prediction history")
    rows = cal_mod.list_predictions(limit=200)
    if not rows:
        st.info("No predictions yet.")
        return
    df = pd.DataFrame(rows)
    show = ["id", "created_at", "category", "reconciled_p",
            "band_low", "band_high", "expected_resolution", "resolved",
            "resolution_outcome", "brier_reconciled", "trigger"]
    d = df[show].copy()
    for c in ("reconciled_p", "band_low", "band_high"):
        d[c] = d[c].apply(lambda v: f"{v:.0%}")
    d["brier_reconciled"] = d["brier_reconciled"].apply(
        lambda v: f"{v:.3f}" if v is not None else "—"
    )
    st.dataframe(d, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Resolve a prediction")
    open_preds = [r for r in rows if r["resolved"] == 0]
    if not open_preds:
        st.caption("No open predictions.")
    else:
        choices = {f'#{r["id"]} ({r["category"]}) — {r["trigger"][:60]}': r["id"]
                   for r in open_preds}
        c = st.selectbox("Pick a prediction", list(choices.keys()))
        outcome = st.radio("Outcome?", ["Yes (event occurred)", "No (event did not occur)"],
                           horizontal=True)
        if st.button("Mark resolved"):
            cal_mod.resolve_prediction(choices[c], outcome.startswith("Yes"))
            st.success("Resolved. Brier scored. Calibration weights updated.")
            st.rerun()

    st.divider()
    csv = df.to_csv(index=False)
    st.download_button("Download predictions as CSV", csv.encode("utf-8"),
                       "risk_oracle_predictions.csv", mime="text/csv")


# =============================================================================
# Calibration tab
# =============================================================================

def render_calibration_tab():
    st.subheader("Calibration dashboard")
    st.caption(
        "Brier score = (forecast - actual)² averaged across resolved predictions. "
        "Lower is better. 0.25 = random; <0.20 = decent; <0.10 = excellent. "
        "Calibration weights update automatically based on these scores."
    )
    stats = cal_mod.category_brier_stats()
    if not stats:
        st.info(
            "No resolved predictions yet. Calibration weights stabilise after ~50 "
            "resolutions per category. Run forecasts, resolve them on the History tab."
        )
        return
    rows = []
    for cat, s in stats.items():
        rows.append({
            "Category": get_category(cat).name if cat in TAXONOMY else cat,
            "Resolved n": s["n"],
            "Primary Brier": f"{s['brier_primary']:.3f}",
            "Critic Brier": f"{s['brier_critic']:.3f}",
            "Reconciled Brier": f"{s['brier_reconciled']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# About tab
# =============================================================================

def render_about_tab():
    st.subheader("Risk Oracle — Architecture")
    st.markdown("""
**v1 core**: 8-category taxonomy · smart router (LLM + rule-based fallback) ·
parallel primary + critic models · reconciliation with disagreement-as-uncertainty ·
OSINT verification (GDELT, FRED, Polymarket, Manifold, Metaculus, USGS, World Bank, NOAA) ·
SQLite calibration store with Brier-scored feedback loop · cross-category contagion ·
time-hazard surface · tail-explicit output (VaR, ES) · black-swan detection ·
Kelly-based decision recommendation.

**v2 enhancements (this version)**:
- 💼 **Portfolio context** — tag positions by category sensitivity; forecasts show
  your specific dollar exposure and hedge ideas.
- 🪄 **Evidence assistant** — describe the situation in plain English; an LLM
  extracts evidence factors with suggested likelihood ratios and rationale.
- 📝 **Explanation layer** — every forecast gets a 2-3 paragraph natural-language
  explanation of why the probability landed where it did.
- 👁 **Watchlist** — persistent ongoing triggers with manual refresh and
  movement-based alerts. Background polling requires running locally + cron.
- ⚖ **Comparison view** — side-by-side with Polymarket, Manifold, Metaculus.
  Large disagreement with market consensus is flagged.

**Honest caveats**:
- Underlying primary/critic models are simplified illustrative variants. Swap in
  real GARCH / SEIR / catastrophe / FAIR / Bayesian-network implementations for
  production-grade accuracy in specific categories.
- Calibration weights only diverge from 50/50 after ~50 resolved predictions
  per category. Be patient.
- Portfolio loss estimates assume an illustrative $1B "reference unit"; for
  precise dollar conversions, override `expected_drawdown_pct` in
  `portfolio.estimate_loss_given_event`.
- Watchlist on Streamlit Community Cloud has no true background polling; click
  "Refresh all" or run locally with cron. Calibration DB is ephemeral on
  Community unless you wire `EXTERNAL_DB_URL` to a persistent Postgres.
- This is decision support, not an oracle. Tracking and resolving predictions
  is the single highest-leverage habit.
""")


if __name__ == "__main__":
    main()
