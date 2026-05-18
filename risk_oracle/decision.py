"""
Decision layer — translates probability + impact into action guidance.

Includes:
- Kelly criterion for position sizing under uncertainty
- Sensitivity analysis around the point estimate
- Value of information (VOI) calculation for next research step
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class DecisionRecommendation:
    kelly_fraction: float          # full Kelly fraction
    fractional_kelly: float         # 1/4 Kelly recommendation
    expected_value: float           # in $ given a unit bet
    max_loss_at_band: float         # worst case if probability at lower band
    confidence_label: str           # "speculative" / "actionable" / "high conviction"
    notes: List[str]


def kelly_recommendation(
    probability: float,
    band_low: float,
    band_high: float,
    win_payoff: float = 1.0,
    loss_payoff: float = 1.0,
    fractional_factor: float = 0.25,
) -> DecisionRecommendation:
    """Compute Kelly position sizing.

    Kelly fraction f* = (p*b - q) / b  where:
        p = probability of win
        q = 1 - p
        b = ratio of win payoff to loss payoff
    f* > 0 means bet in favour; f* < 0 means bet against (or pass).
    Fractional Kelly (1/4 or 1/2) is standard practice to reduce variance.
    """
    p = probability
    q = 1 - p
    b = win_payoff / loss_payoff if loss_payoff > 0 else 1.0
    f_full = (p * b - q) / b if b > 0 else 0.0
    f_frac = f_full * fractional_factor

    ev = p * win_payoff - q * loss_payoff
    max_loss = q * loss_payoff
    # Re-compute EV at lower band as worst case
    ev_low = band_low * win_payoff - (1 - band_low) * loss_payoff
    max_loss_at_band = -ev_low if ev_low < 0 else max_loss

    if abs(f_frac) < 0.01:
        label = "pass (no edge)"
    elif f_frac < 0:
        label = "bet against / hedge"
    elif (band_high - band_low) > 0.30:
        label = "speculative (wide uncertainty)"
    elif f_frac < 0.05:
        label = "small position"
    else:
        label = "actionable"

    notes = []
    if (band_high - band_low) > 0.25:
        notes.append(
            f"Wide uncertainty band ({band_low:.0%}-{band_high:.0%}) — "
            "size cautiously."
        )
    if abs(f_full) > 0.5:
        notes.append(
            "Full Kelly fraction is very large — fractional Kelly is strongly "
            "recommended to manage variance."
        )

    return DecisionRecommendation(
        kelly_fraction=float(f_full),
        fractional_kelly=float(f_frac),
        expected_value=float(ev),
        max_loss_at_band=float(max_loss_at_band),
        confidence_label=label,
        notes=notes,
    )


def sensitivity_table(probability: float, win_payoff: float = 1.0,
                      loss_payoff: float = 1.0) -> Dict[str, Dict[str, float]]:
    """How does the Kelly recommendation change if probability shifts ±10%?"""
    out = {}
    for delta_pct, label in [(-10, "p - 10%"), (0, "p"), (10, "p + 10%")]:
        p = max(0.01, min(0.99, probability + delta_pct/100))
        q = 1 - p
        b = win_payoff / loss_payoff if loss_payoff > 0 else 1.0
        f_full = (p * b - q) / b if b > 0 else 0.0
        ev = p * win_payoff - q * loss_payoff
        out[label] = {
            "probability": p,
            "kelly_full": f_full,
            "kelly_quarter": f_full * 0.25,
            "expected_value": ev,
        }
    return out


def value_of_information(
    current_band_width: float,
    expected_band_width_after: float,
    decision_value: float = 1.0,
) -> Dict[str, float]:
    """Estimate VOI from a research action that would tighten the band.

    Simple heuristic: VOI scales with band reduction × decision value.
    """
    if current_band_width <= 0:
        return {"voi_score": 0.0, "band_reduction": 0.0}
    reduction = max(0.0, current_band_width - expected_band_width_after)
    voi = reduction * decision_value / current_band_width
    return {
        "voi_score": float(voi),
        "band_reduction_pct_points": float(100 * reduction),
    }
