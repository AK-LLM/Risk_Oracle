"""
lifecycle.py — V2.2 stage tagging for watchlist forecasts.

Ports STP's SCOUT/STALKING/STRIKING/LATE lifecycle pattern into Risk Oracle.
A watchlist item's stage is computed from its refresh history:

  SCOUT     — fewer than 3 refreshes; signal is brand-new, low confidence
  STALKING  — ≥3 refreshes, evidence accumulating, band still wide
  STRIKING  — band has compressed materially (probability consolidating),
              market_prob is also moving toward our point_p (consensus forming)
  LATE      — probability has reached an extreme (≥0.90 or ≤0.10),
              OR many refreshes have happened (>20),
              OR band is very narrow (consensus fully formed)

The stage feeds two things:
  • Dispatch alert priority (STRIKING/LATE moves get higher-urgency alerts)
  • Sizing recommendations (LATE = chasing, downgrade Kelly fraction)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


STAGE_SCOUT = "SCOUT"
STAGE_STALKING = "STALKING"
STAGE_STRIKING = "STRIKING"
STAGE_LATE = "LATE"
STAGES = (STAGE_SCOUT, STAGE_STALKING, STAGE_STRIKING, STAGE_LATE)


def compute_stage(history: List[Dict[str, Any]],
                  current_probability: Optional[float] = None) -> str:
    """Return a stage label for a watchlist item given its refresh history.

    `history` is a list of dicts each with keys: probability, band_low,
    band_high, market_prob (from watchlist_history table). Newest last.
    """
    n = len(history)
    if n == 0:
        return STAGE_SCOUT
    if n < 3:
        return STAGE_SCOUT

    latest = history[-1]
    p = current_probability if current_probability is not None else latest.get("probability", 0.5)

    # Hard signals for LATE
    if p is not None and (p >= 0.90 or p <= 0.10):
        return STAGE_LATE
    if n > 20:
        return STAGE_LATE

    # Band-compression check
    initial = history[0]
    initial_bw = float(initial.get("band_high", 1.0)) - float(initial.get("band_low", 0.0))
    recent_bw = float(latest.get("band_high", 1.0)) - float(latest.get("band_low", 0.0))
    if initial_bw <= 0:
        compression = 1.0
    else:
        compression = recent_bw / initial_bw

    if compression < 0.3 and n >= 5:
        return STAGE_LATE
    if compression < 0.6 and n >= 4:
        return STAGE_STRIKING
    return STAGE_STALKING


def stage_priority(stage: str) -> int:
    """Lower number = earlier in lifecycle = act-now priority."""
    return {STAGE_SCOUT: 0, STAGE_STALKING: 1, STAGE_STRIKING: 2, STAGE_LATE: 3}.get(stage, 99)


def stage_sizing_multiplier(stage: str) -> float:
    """How much to scale Kelly recommendation by lifecycle stage.

    SCOUT     — 0.5  (very early; the signal might not even confirm)
    STALKING  — 0.8  (evidence building; size cautiously)
    STRIKING  — 1.0  (full confirmation; full Kelly)
    LATE      — 0.4  (consensus formed; chasing has poor expected value)
    """
    return {STAGE_SCOUT: 0.5, STAGE_STALKING: 0.8, STAGE_STRIKING: 1.0, STAGE_LATE: 0.4}.get(stage, 1.0)
