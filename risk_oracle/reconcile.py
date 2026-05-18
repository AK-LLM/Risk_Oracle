"""
Reconciliation — combine primary + critic outputs into a final probability and
impact distribution, with explicit treatment of disagreement.

Key rule: when primary and critic disagree, the answer is NOT just the average.
The output uncertainty band widens, because the disagreement is itself signal.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from .models import ModelOutput


@dataclass
class ReconciledOutput:
    point_probability: float
    band_low: float
    band_high: float
    primary: ModelOutput
    critic: ModelOutput
    disagreement: float
    disagreement_flag: bool
    primary_weight: float
    critic_weight: float
    combined_impact_samples: np.ndarray
    combined_duration_samples: np.ndarray
    notes: List[str] = field(default_factory=list)


def reconcile(
    primary: ModelOutput,
    critic: ModelOutput,
    primary_calibration: float = 0.5,    # historical Brier-based weight, 0-1
    critic_calibration: float = 0.5,
    base_band_width: float = 0.10,        # default ±10% uncertainty
    disagreement_threshold: float = 0.15,
) -> ReconciledOutput:
    p1 = primary.posterior_probability
    p2 = critic.posterior_probability

    # Calibration weights — normalised. Default 50/50 until track record exists.
    total = primary_calibration + critic_calibration
    if total <= 0:
        w1, w2 = 0.5, 0.5
    else:
        w1 = primary_calibration / total
        w2 = critic_calibration / total

    point = w1 * p1 + w2 * p2
    disagreement = abs(p1 - p2)

    # Disagreement-as-uncertainty: widen band proportional to disagreement.
    # base_band ± additional based on |p1 - p2|.
    band_width = base_band_width * (1.0 + 2.0 * disagreement)
    band_low = max(0.0, point - band_width)
    band_high = min(1.0, point + band_width)

    notes: List[str] = []
    flag = disagreement > disagreement_threshold
    if flag:
        notes.append(
            f"Material disagreement between models: |p_primary - p_critic| = "
            f"{disagreement:.2%}. Uncertainty band widened from "
            f"±{base_band_width:.0%} to ±{band_width:.0%}."
        )

    # Combine impact samples by weighted concatenation
    n_primary = int(round(w1 * len(primary.impact_samples)))
    n_critic = len(primary.impact_samples) - n_primary
    rng = np.random.default_rng(0)
    if n_primary > 0:
        idx_p = rng.choice(len(primary.impact_samples), size=n_primary, replace=False)
    else:
        idx_p = np.array([], dtype=int)
    if n_critic > 0:
        idx_c = rng.choice(len(critic.impact_samples), size=n_critic, replace=False)
    else:
        idx_c = np.array([], dtype=int)

    combined_impact = np.concatenate([
        primary.impact_samples[idx_p],
        critic.impact_samples[idx_c],
    ])
    combined_duration = np.concatenate([
        primary.duration_samples[idx_p],
        critic.duration_samples[idx_c],
    ])

    return ReconciledOutput(
        point_probability=float(point),
        band_low=float(band_low),
        band_high=float(band_high),
        primary=primary,
        critic=critic,
        disagreement=float(disagreement),
        disagreement_flag=flag,
        primary_weight=float(w1),
        critic_weight=float(w2),
        combined_impact_samples=combined_impact,
        combined_duration_samples=combined_duration,
        notes=notes,
    )


def time_hazard_surface(
    point_probability: float,
    duration_mean_months: float,
    horizon_months: int = 36,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a hazard-rate curve over time.

    Returns (times_months, cumulative_probability_resolved).
    Models the conditional probability that the event has resolved by month t
    using an exponential hazard with rate calibrated to the typical duration.
    """
    times = np.arange(1, horizon_months + 1)
    # Hazard rate: 1/duration_mean (events resolve at typical pace)
    # but conditional on the event having occurred (point_probability)
    rate = 1.0 / max(duration_mean_months, 0.5)
    # Probability of still being unresolved at time t given it occurred:
    survival = np.exp(-rate * times)
    # Probability event is still active at time t (unconditional):
    p_active_at_t = point_probability * survival
    return times, p_active_at_t


def tail_risk(
    impact_samples: np.ndarray,
    tail_quantile: float = 0.95,
) -> dict:
    """Compute tail-explicit risk metrics: VaR, Expected Shortfall, worst-case."""
    if len(impact_samples) == 0 or impact_samples.max() == 0:
        return {
            "VaR_95": 0.0, "VaR_99": 0.0,
            "ES_95": 0.0, "worst_1pct_mean": 0.0,
            "expected_loss": 0.0,
        }
    p_threshold_95 = np.quantile(impact_samples, 0.95)
    p_threshold_99 = np.quantile(impact_samples, 0.99)
    es_95 = impact_samples[impact_samples >= p_threshold_95].mean() if (impact_samples >= p_threshold_95).any() else 0.0
    worst_1pct = impact_samples[impact_samples >= p_threshold_99].mean() if (impact_samples >= p_threshold_99).any() else 0.0
    return {
        "VaR_95": float(p_threshold_95),
        "VaR_99": float(p_threshold_99),
        "ES_95": float(es_95),
        "worst_1pct_mean": float(worst_1pct),
        "expected_loss": float(impact_samples.mean()),
    }
