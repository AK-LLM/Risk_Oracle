"""
Risk Oracle — Streamlit entrypoint.

Run locally:  streamlit run app.py
Run on Streamlit Community Cloud:  push to GitHub, connect via share.streamlit.io
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st

from risk_oracle import router as router_mod
from risk_oracle import models as models_mod
from risk_oracle import reconcile as reconcile_mod
from risk_oracle import osint as osint_mod
from risk_oracle import calibration as cal_mod
from risk_oracle import decision as decision_mod
from risk_oracle import contagion as contagion_mod
from risk_oracle import visualization as viz
from risk_oracle.taxonomy import TAXONOMY, get_category, CATEGORY_KEYS


st.set_page_config(
    page_title="Risk Oracle",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_secrets() -> Dict[str, str]:
    """Read secrets from Streamlit secrets if available, else from env."""
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


# ---------- Sidebar ----------

def render_sidebar(secrets: Dict[str, str]):
    st.sidebar.title("Risk Oracle")
    st.sidebar.caption("Probabilistic risk modeling with calibration feedback")

    st.sidebar.subheader("Status")
    rows = [
        ("Smart router (LLM)",
         "✓ active" if (secrets["ANTHROPIC_API_KEY"] or secrets["OPENAI_API_KEY"])
         else "○ rule-based fallback"),
        ("FRED macro data", "✓ active" if secrets["FRED_API_KEY"] else "○ skipped"),
        ("ACLED conflict data", "✓ active" if secrets["ACLED_API_KEY"] else "○ skipped"),
        ("GDELT news", "✓ active (free)"),
        ("Polymarket / Manifold", "✓ active (free)"),
        ("World Bank / USGS / NOAA", "✓ active (free)"),
    ]
    for label, status in rows:
        st.sidebar.text(f"{status}   {label}")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**No API keys configured?** The system still works using only free sources. "
        "Add keys via `.streamlit/secrets.toml` (local) or Cloud secrets (Community) for more signals."
    )


# ---------- Main app ----------

def main():
    secrets = _get_secrets()
    render_sidebar(secrets)

    st.title("Risk Oracle")
    st.markdown(
        "Smart router → primary + critic models → reconciliation → OSINT verification → "
        "calibrated probability with decision support."
    )

    tab_forecast, tab_history, tab_calibration, tab_about = st.tabs(
        ["New forecast", "History", "Calibration", "About / architecture"]
    )

    with tab_forecast:
        render_forecast_tab(secrets)
    with tab_history:
        render_history_tab()
    with tab_calibration:
        render_calibration_tab()
    with tab_about:
        render_about_tab()


# ---------- Forecast tab ----------

def render_forecast_tab(secrets: Dict[str, str]):
    st.subheader("Run a new forecast")

    # Default examples
    example = st.selectbox(
        "Example trigger (or type your own below)",
        options=[
            "—",
            "Will direct Iran-Israel strikes still be occurring on October 1, 2026?",
            "Will the US enter a recession (two consecutive negative GDP quarters) by Q4 2026?",
            "Will a major H5N1 outbreak (>1000 human cases in one country) be declared by WHO before end of 2026?",
            "Will a Cat 4+ hurricane make US landfall during the 2026 Atlantic season?",
            "Will the S&P 500 close more than 20% below its current level at any point in 2026?",
        ],
        index=0,
    )
    default_trigger = "" if example == "—" else example
    trigger = st.text_area(
        "Trigger (precise, dated question)",
        value=default_trigger,
        height=80,
        placeholder="Will <specific event> have occurred by <specific date>?",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        run_button = st.button("Run forecast", type="primary", disabled=not trigger.strip())
    with col2:
        expected_resolution = st.date_input(
            "Expected resolution date (for calibration tracking)",
            value=datetime.utcnow().date() + timedelta(days=180),
        )

    if not run_button:
        return

    # === Step 1: Route ===
    with st.spinner("Routing trigger through smart router…"):
        decision = router_mod.route(trigger, secrets=secrets)

    st.divider()
    st.markdown("### Step 1 — Smart router")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Primary category", get_category(decision.primary_category).name)
    with col2:
        st.metric("Router confidence", f"{decision.confidence:.0%}")
    with col3:
        st.metric("Mode", "LLM" if decision.used_llm else "Rule-based")

    if decision.is_anomaly:
        st.warning(
            "**Black-swan / anomaly flag triggered.** This trigger doesn't fit cleanly into "
            "any taxonomy category. Routing to the closest match but maximum uncertainty "
            "should be assumed. Reasons: " + "; ".join(decision.anomaly_reasons)
        )

    if decision.secondary_categories:
        st.info(
            "Cross-category spillover detected. Secondary categories: "
            + ", ".join(get_category(c).name for c in decision.secondary_categories)
        )

    if decision.extracted_features and decision.used_llm:
        with st.expander("Extracted features"):
            st.json(decision.extracted_features)

    st.caption("Router reasoning: " + decision.reasoning)

    spec = decision.primary_spec
    st.caption(
        f"**Primary model**: `{spec.primary_model}`  ·  "
        f"**Critic model**: `{spec.critic_model}`"
    )

    # === Step 2: Build evidence ===
    st.divider()
    st.markdown("### Step 2 — Set prior and evidence")

    with st.expander("Prior probability and evidence factors", expanded=True):
        prior = st.slider(
            "Base rate / prior probability (your starting belief before specific evidence)",
            min_value=0.01, max_value=0.99, step=0.01,
            value=float(spec.base_rate_default),
        )
        st.caption("Reference class: " + spec.reference_class_note)

        st.markdown("**Evidence factors** — likelihood ratios for each piece of evidence.")
        st.caption("LR > 1 raises probability; LR < 1 lowers it. 1.0 = neutral.")

        ev_count = st.number_input("Number of evidence factors", min_value=0, max_value=8, value=3)
        evidence: list[models_mod.EvidenceFactor] = []
        defaults = [
            ("Active signal supporting event", 2.0),
            ("Counter-signal indicating resolution", 0.6),
            ("Domestic/institutional incentive structure", 1.5),
            ("Economic / capacity constraints", 0.8),
        ]
        for i in range(int(ev_count)):
            cols = st.columns([3, 2, 1])
            name = cols[0].text_input(
                f"Factor {i+1} name", value=defaults[i % 4][0], key=f"ev_name_{i}"
            )
            lr = cols[1].number_input(
                "Likelihood ratio",
                min_value=0.05, max_value=20.0, step=0.1,
                value=float(defaults[i % 4][1]), key=f"ev_lr_{i}",
            )
            conf = cols[2].slider(
                "Conf.", min_value=0.0, max_value=1.0, step=0.05, value=0.8,
                key=f"ev_conf_{i}",
            )
            evidence.append(models_mod.EvidenceFactor(name=name, likelihood_ratio=lr, confidence=conf))

    # === Step 3: Run models ===
    with st.spinner("Running primary + critic models…"):
        impacts = models_mod.get_default_impacts(decision.primary_category)
        primary = models_mod.run_primary(
            decision.primary_category, prior, evidence, impacts,
            duration_mean=spec.typical_duration_months,
        )
        critic = models_mod.run_critic(
            decision.primary_category, prior, evidence, impacts,
            duration_mean=spec.typical_duration_months,
        )

    # === Step 4: Reconcile ===
    w_primary, w_critic = cal_mod.get_model_weights(decision.primary_category)
    reconciled = reconcile_mod.reconcile(
        primary, critic,
        primary_calibration=w_primary, critic_calibration=w_critic,
    )

    st.divider()
    st.markdown("### Step 3 — Parallel models + reconciliation")

    col1, col2, col3 = st.columns(3)
    col1.metric("Primary P", f"{primary.posterior_probability:.1%}",
                help=primary.methodology)
    col2.metric("Critic P", f"{critic.posterior_probability:.1%}",
                help=critic.methodology)
    col3.metric("Disagreement", f"{reconciled.disagreement:.1%}",
                delta="material" if reconciled.disagreement_flag else "small",
                delta_color="inverse")

    # === Step 5: OSINT verification ===
    with st.spinner("Querying OSINT signals for verification…"):
        osint_bundle = osint_mod.gather_osint(
            query=trigger,
            signal_keys=spec.osint_signals,
            secrets=secrets,
        )

    market_prob = None
    for sig in osint_bundle.signals:
        if sig.source in ("polymarket", "manifold", "metaculus") and isinstance(sig.value, float):
            market_prob = sig.value
            break

    # === Step 6: Final output ===
    st.divider()
    st.markdown("### Step 4 — Output: probability, band, and decision")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability", f"{reconciled.point_probability:.1%}")
    col2.metric("Band low (P10-ish)", f"{reconciled.band_low:.1%}")
    col3.metric("Band high (P90-ish)", f"{reconciled.band_high:.1%}")
    if market_prob is not None:
        col4.metric("Prediction market", f"{market_prob:.1%}",
                    delta=f"{(market_prob - reconciled.point_probability):+.1%}",
                    help="Live aggregated probability from prediction markets.")
    else:
        col4.metric("Prediction market", "—", help="No matching market found.")

    st.plotly_chart(
        viz.probability_band(
            reconciled.point_probability,
            reconciled.band_low,
            reconciled.band_high,
            primary.posterior_probability,
            critic.posterior_probability,
            market_p=market_prob,
        ),
        use_container_width=True,
    )

    for note in reconciled.notes:
        st.info(note)

    # Tail risk
    tail = reconcile_mod.tail_risk(reconciled.combined_impact_samples)
    st.markdown("#### Tail-explicit risk")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected loss", f"${tail['expected_loss']:,.0f}")
    col2.metric("VaR 95%", f"${tail['VaR_95']:,.0f}")
    col3.metric("VaR 99%", f"${tail['VaR_99']:,.0f}")
    col4.metric("Worst 1% mean", f"${tail['worst_1pct_mean']:,.0f}")

    st.plotly_chart(
        viz.loss_distribution(
            reconciled.combined_impact_samples,
            title="Loss distribution (combined primary + critic, weighted)",
        ),
        use_container_width=True,
    )

    # Time dynamics
    st.markdown("#### Time dynamics (hazard surface)")
    times, p_active = reconcile_mod.time_hazard_surface(
        reconciled.point_probability,
        duration_mean_months=spec.typical_duration_months,
        horizon_months=36,
    )
    st.plotly_chart(
        viz.hazard_curve(times, p_active),
        use_container_width=True,
    )

    # Cross-category contagion
    st.markdown("#### Cross-category contagion")
    spillover = contagion_mod.cross_category_spillover(
        decision.primary_category,
        tail["expected_loss"],
        reconciled.point_probability,
    )
    if spillover:
        st.plotly_chart(
            viz.contagion_chart(spillover, tail["expected_loss"]),
            use_container_width=True,
        )

    # OSINT verification
    st.markdown("#### OSINT verification signals")
    if not osint_bundle.signals:
        st.caption("No OSINT signals returned for this trigger.")
    else:
        st.caption(
            f"Queried {len(osint_bundle.sources_queried)} sources, "
            f"{len(osint_bundle.sources_succeeded)} returned data."
        )
        for sig in osint_bundle.signals:
            if sig.error:
                st.text(f"○ {sig.source} ({sig.label}): error — {sig.error}")
            elif sig.value is None:
                st.text(f"○ {sig.source} ({sig.label}): no data")
            else:
                st.text(f"✓ {sig.source} ({sig.label}): {sig.interpretation}")

    # Decision recommendation
    st.markdown("#### Decision recommendation (Kelly-based)")
    rec = decision_mod.kelly_recommendation(
        reconciled.point_probability, reconciled.band_low, reconciled.band_high,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Full Kelly fraction", f"{rec.kelly_fraction:+.1%}")
    col2.metric("Fractional Kelly (¼)", f"{rec.fractional_kelly:+.1%}")
    col3.metric("Action label", rec.confidence_label)
    for note in rec.notes:
        st.caption("• " + note)

    sens = decision_mod.sensitivity_table(reconciled.point_probability)
    st.markdown("**Sensitivity:**")
    sens_df = pd.DataFrame(sens).T
    sens_df["probability"] = sens_df["probability"].apply(lambda v: f"{v:.0%}")
    sens_df["kelly_full"] = sens_df["kelly_full"].apply(lambda v: f"{v:+.1%}")
    sens_df["kelly_quarter"] = sens_df["kelly_quarter"].apply(lambda v: f"{v:+.1%}")
    sens_df["expected_value"] = sens_df["expected_value"].apply(lambda v: f"{v:+.2f}")
    st.dataframe(sens_df, use_container_width=True)

    # === Step 7: Log ===
    pred_id = cal_mod.log_prediction(
        cal_mod.Prediction(
            trigger=trigger,
            category=decision.primary_category,
            primary_p=primary.posterior_probability,
            critic_p=critic.posterior_probability,
            reconciled_p=reconciled.point_probability,
            band_low=reconciled.band_low,
            band_high=reconciled.band_high,
            expected_resolution=str(expected_resolution),
            metadata={
                "router_confidence": decision.confidence,
                "router_used_llm": decision.used_llm,
                "secondary_categories": decision.secondary_categories,
                "is_anomaly": decision.is_anomaly,
                "tail_risk": tail,
                "primary_weight": reconciled.primary_weight,
                "critic_weight": reconciled.critic_weight,
                "market_prob": market_prob,
            },
        ),
    )
    st.success(f"✓ Forecast logged (id={pred_id}). Resolve it on the History tab when the date arrives.")


# ---------- History tab ----------

def render_history_tab():
    st.subheader("Prediction history")
    rows = cal_mod.list_predictions(limit=200)
    if not rows:
        st.info("No predictions yet. Run a forecast on the New forecast tab.")
        return

    df = pd.DataFrame(rows)
    show_cols = ["id", "created_at", "category", "reconciled_p",
                 "band_low", "band_high", "expected_resolution", "resolved",
                 "resolution_outcome", "brier_reconciled", "trigger"]
    df_show = df[show_cols].copy()
    for col in ("reconciled_p", "band_low", "band_high"):
        df_show[col] = df_show[col].apply(lambda v: f"{v:.0%}")
    df_show["brier_reconciled"] = df_show["brier_reconciled"].apply(
        lambda v: f"{v:.3f}" if v is not None else "—"
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Resolve a prediction")
    open_preds = [r for r in rows if r["resolved"] == 0]
    if not open_preds:
        st.caption("No open predictions to resolve.")
    else:
        pred_choices = {f'#{r["id"]} ({r["category"]}) — {r["trigger"][:60]}': r["id"]
                        for r in open_preds}
        choice = st.selectbox("Pick a prediction to resolve", list(pred_choices.keys()))
        outcome = st.radio("Did the event occur?", ["Yes", "No"], horizontal=True)
        if st.button("Mark resolved"):
            cal_mod.resolve_prediction(pred_choices[choice], outcome == "Yes")
            st.success("Resolved. Brier scores computed. Calibration weights updated.")
            st.rerun()

    st.divider()
    st.markdown("### Export")
    csv = df.to_csv(index=False)
    st.download_button(
        "Download all predictions as CSV",
        csv.encode("utf-8"),
        file_name="risk_oracle_predictions.csv",
        mime="text/csv",
    )


# ---------- Calibration tab ----------

def render_calibration_tab():
    st.subheader("Calibration dashboard")
    st.caption(
        "Brier score: (forecast - actual)² averaged across resolved predictions. "
        "Lower is better. < 0.20 is decent, < 0.10 is excellent. "
        "Random guessing scores 0.25."
    )
    stats = cal_mod.category_brier_stats()
    if not stats:
        st.info(
            "No resolved predictions yet. Calibration emerges after ~50 resolved "
            "predictions per category. Run forecasts, set expected resolution dates, "
            "and come back to score them."
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

    st.divider()
    st.markdown(
        "**How weights update**: the reconciliation layer reads the rolling Brier "
        "from the last 50 resolved predictions per category and uses `weight = 1 - Brier`. "
        "Better-performing models get more weight automatically."
    )


# ---------- About tab ----------

def render_about_tab():
    st.subheader("Architecture")
    st.markdown("""
This system implements the architecture designed across the planning conversation:

**v1 core**:
- Smart router with LLM feature extraction + rule-based fallback
- 8-category trigger taxonomy with primary + critic model assignments
- Bayesian probability update + Monte Carlo impact simulation (20k sims)
- Reconciliation with disagreement-as-uncertainty (band widens with model disagreement)
- OSINT verification using free/open APIs (GDELT, FRED, Polymarket, Manifold, USGS, World Bank)
- Calibration loop with Brier scoring and rolling model weight updates

**v2 enhancements (active)**:
- Cross-category contagion (interaction matrix between categories)
- Time dynamics (exponential hazard surface over 36 months)
- Tail-explicit output (VaR 95%, VaR 99%, Expected Shortfall, worst 1%)
- Black swan detection (anomaly flag at router stage)
- Decision layer (Kelly criterion, fractional Kelly, sensitivity)

**v3 hooks (extensible interfaces)**:
- Prediction market integration (live for Polymarket and Manifold)
- Active learning / VOI calculation (basic)
- Causal DAGs, LLM wargaming, multilingual OSINT, satellite vision —
  interfaces in place, implementations to layer in

**Critical honest caveats**:
- Underlying primary/critic models are simplified illustrative versions of
  GARCH, SEIR, FAIR, catastrophe models, etc. The architecture is right;
  the underlying models should be swapped for production-grade equivalents
  for specific domains.
- Calibration requires ~50 resolved predictions per category before weights
  meaningfully diverge from 50/50.
- This is decision support, not an oracle. Tracking your predictions is the
  single highest-value habit you can build.
""")


if __name__ == "__main__":
    main()
