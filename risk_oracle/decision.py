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


# ============================================================================
# Polymarket-specific bet sizing
# ============================================================================

@dataclass
class PolymarketRecommendation:
    side: str                       # "YES", "NO", or "PASS"
    market_yes_price: float
    our_probability: float
    edge: float                     # absolute edge in probability points
    full_kelly_fraction: float       # fraction of bankroll (full Kelly)
    fractional_kelly_fraction: float # fraction of bankroll (typically quarter)
    expected_value_per_dollar: float # USD return per USD risked
    recommended_size_usd: float      # in USD, after liquidity & sanity clamps
    raw_kelly_size_usd: float        # before any clamps
    liquidity_warning: bool
    notes: List[str]


def polymarket_recommendation(
    our_probability: float,
    band_low: float,
    band_high: float,
    market_yes_price: float,
    bankroll_usd: float = 10_000,
    available_liquidity_usd: float = 1_000,
    fractional_factor: float = 0.25,   # quarter Kelly default
    min_edge: float = 0.02,
    max_position_pct_of_bankroll: float = 0.10,
) -> PolymarketRecommendation:
    """Recommend a Polymarket bet given our forecast vs. the live market price.

    Polymarket binary market mechanics:
      A YES share costs `market_yes_price` (0–1) and pays $1 if YES resolves.
      A NO share costs `1 - market_yes_price` and pays $1 if NO resolves.
      For YES at price p_m with true probability p_t:
          EV per dollar = (p_t - p_m) / p_m
          Kelly fraction = (p_t * b - (1 - p_t)) / b   where b = (1 - p_m) / p_m
      Symmetric for NO (flip signs).

    Returns a structured recommendation including liquidity-aware position size.
    """
    notes: List[str] = []
    p_t = max(min(our_probability, 0.999), 0.001)
    p_m = max(min(market_yes_price, 0.999), 0.001)

    edge_yes = p_t - p_m
    if abs(edge_yes) < min_edge:
        return PolymarketRecommendation(
            side="PASS",
            market_yes_price=p_m, our_probability=p_t,
            edge=abs(edge_yes),
            full_kelly_fraction=0.0, fractional_kelly_fraction=0.0,
            expected_value_per_dollar=0.0,
            recommended_size_usd=0.0, raw_kelly_size_usd=0.0,
            liquidity_warning=False,
            notes=[
                f"Edge ({edge_yes:+.1%}) is below the {min_edge:.0%} threshold — "
                f"likely doesn't beat transaction costs and fees. Pass."
            ],
        )

    if edge_yes > 0:
        side = "YES"
        b = (1 - p_m) / p_m
        f_full = (p_t * b - (1 - p_t)) / b
        ev_per_dollar = (p_t - p_m) / p_m
    else:
        side = "NO"
        p_no = 1 - p_m
        p_t_no = 1 - p_t
        b = (1 - p_no) / p_no
        f_full = (p_t_no * b - (1 - p_t_no)) / b
        ev_per_dollar = (p_t_no - p_no) / p_no

    f_full = max(0.0, f_full)
    f_frac = f_full * fractional_factor
    raw_size = f_frac * bankroll_usd

    # Clamps: never more than X% of bankroll, never more than 20% of liquidity
    cap_bankroll = max_position_pct_of_bankroll * bankroll_usd
    cap_liquidity = 0.20 * available_liquidity_usd

    final_size = raw_size
    liquidity_warning = False

    if final_size > cap_bankroll:
        notes.append(
            f"Raw Kelly size (${raw_size:.0f}) exceeds {max_position_pct_of_bankroll:.0%} "
            f"of bankroll cap (${cap_bankroll:.0f}). Capping at bankroll limit."
        )
        final_size = cap_bankroll

    if final_size > cap_liquidity:
        liquidity_warning = True
        notes.append(
            f"Size (${final_size:.0f}) is >20% of available liquidity "
            f"(${available_liquidity_usd:.0f}); slippage will be material. "
            f"Capping at ${cap_liquidity:.0f}."
        )
        final_size = cap_liquidity

    # Band sanity check
    band_width = band_high - band_low
    if abs(edge_yes) < 0.05 and band_width > 0.20:
        notes.append(
            f"Edge ({edge_yes:+.1%}) is small relative to confidence band "
            f"({band_width:.0%} wide). Consider waiting for higher conviction."
        )

    if f_full > 0.5:
        notes.append(
            f"Full Kelly is {f_full:.1%} of bankroll — that's extremely large. "
            f"Fractional Kelly strongly recommended; size shown is "
            f"{fractional_factor:.0%} of full."
        )

    return PolymarketRecommendation(
        side=side,
        market_yes_price=p_m,
        our_probability=p_t,
        edge=abs(edge_yes),
        full_kelly_fraction=float(f_full),
        fractional_kelly_fraction=float(f_frac),
        expected_value_per_dollar=float(ev_per_dollar),
        recommended_size_usd=float(final_size),
        raw_kelly_size_usd=float(raw_size),
        liquidity_warning=liquidity_warning,
        notes=notes,
    )
