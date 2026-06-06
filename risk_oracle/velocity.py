"""
velocity.py — V2.2 probability-movement velocity tracking for watchlist items.

Mirrors STP's velocity_tracker pattern. Each refresh moves the probability
by some delta; this module reports whether those deltas are accelerating,
stable, or decelerating, in either direction.

States:
  ACCELERATING_UP    — probability rising and the rise is getting faster
  ACCELERATING_DOWN  — probability falling and the fall is getting faster
  RISING             — probability rising at a steady pace
  FALLING            — probability falling at a steady pace
  STABLE             — probability flat / oscillating within noise
  DECELERATING       — directional move losing steam (could be exhaustion)
  INSUFFICIENT_DATA  — fewer than 4 refreshes; can't compute acceleration

Used by:
  • Dispatch: ACCELERATING_UP/DOWN events get bumped to higher-urgency alerts
  • Pipeline: feeds the regime + noise context for sizing
"""
from __future__ import annotations
from typing import Any, Dict, List


ACCEL_UP = "ACCELERATING_UP"
ACCEL_DOWN = "ACCELERATING_DOWN"
RISING = "RISING"
FALLING = "FALLING"
STABLE = "STABLE"
DECEL = "DECELERATING"
INSUFFICIENT = "INSUFFICIENT_DATA"


# Minimum probability change to register as a real move (filter noise)
MIN_MOVE = 0.005   # 0.5pp
ACCEL_RATIO = 1.5  # latest delta must exceed prior delta by this factor


def compute_velocity(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return {acceleration, velocity_ratio, recent_delta, prior_delta} from
    a watchlist item's refresh history (newest last)."""
    if len(history) < 4:
        return {
            "acceleration": INSUFFICIENT,
            "velocity_ratio": 1.0,
            "recent_delta": 0.0,
            "prior_delta": 0.0,
        }

    # Two most-recent deltas, smoothed by averaging adjacent refreshes
    p_now = float(history[-1].get("probability", 0.5))
    p_prev = float(history[-2].get("probability", 0.5))
    p_2ago = float(history[-3].get("probability", 0.5))
    p_3ago = float(history[-4].get("probability", 0.5))

    recent_delta = p_now - p_prev
    prior_delta = p_2ago - p_3ago

    # Filter sub-noise moves
    if abs(recent_delta) < MIN_MOVE and abs(prior_delta) < MIN_MOVE:
        return {
            "acceleration": STABLE,
            "velocity_ratio": 1.0,
            "recent_delta": recent_delta,
            "prior_delta": prior_delta,
        }

    if abs(prior_delta) < MIN_MOVE:
        # Was flat; now moving. Treat as new directional move.
        accel = RISING if recent_delta > 0 else FALLING
        return {
            "acceleration": accel,
            "velocity_ratio": float("inf"),
            "recent_delta": recent_delta,
            "prior_delta": prior_delta,
        }

    ratio = recent_delta / prior_delta

    # Same direction, getting faster
    if ratio > ACCEL_RATIO and recent_delta > 0:
        accel = ACCEL_UP
    elif ratio > ACCEL_RATIO and recent_delta < 0:
        accel = ACCEL_DOWN
    # Direction reversal
    elif ratio < 0:
        # Was moving one way, now the other — flag as either RISING or FALLING fresh
        accel = RISING if recent_delta > 0 else FALLING
    # Same direction, losing steam
    elif 0 < ratio < 0.5:
        accel = DECEL
    elif recent_delta > 0:
        accel = RISING
    else:
        accel = FALLING

    return {
        "acceleration": accel,
        "velocity_ratio": ratio,
        "recent_delta": recent_delta,
        "prior_delta": prior_delta,
    }


def is_significant(velocity: Dict[str, Any], threshold: float = 0.03) -> bool:
    """Is the current velocity worth dispatching an alert about?
    Default: a delta of 3pp in either direction over the last refresh."""
    return abs(velocity.get("recent_delta", 0.0)) >= threshold
