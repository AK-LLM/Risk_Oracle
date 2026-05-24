"""
Pipeline orchestrator — single entry point that runs the full forecast workflow.
Used by both the Streamlit forecast tab and the watchlist refresh.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from . import models as models_mod
from . import reconcile as reconcile_mod
from . import osint as osint_mod
from . import calibration as cal_mod
from . import comparison as cmp_mod
from . import contagion as contagion_mod
from .taxonomy import get_category


@dataclass
class ForecastResult:
    trigger: str
    category: str
    primary_p: float
    primary_method: str
    critic_p: float
    critic_method: str
    point_p: float
    band_low: float
    band_high: float
    disagreement: float
    disagreement_flag: bool
    impact_samples: np.ndarray
    duration_samples: np.ndarray
    tail_metrics: Dict[str, float]
    osint_bundle: Any                       # OSINTBundle
    market_prob: Optional[float]            # consensus across markets
    comparison: Any                         # ComparisonBundle
    contagion_spillover: Dict[str, Dict[str, float]]
    evidence_used: List[Dict]
    primary_weight: float
    critic_weight: float
    notes: List[str] = field(default_factory=list)


def run_forecast(
    trigger: str,
    category: str,
    prior: float,
    evidence: List[models_mod.EvidenceFactor],
    secrets: Dict[str, str],
    n_sims: int = 20_000,
    include_comparison: bool = True,
) -> ForecastResult:
    spec = get_category(category)
    impacts = models_mod.get_default_impacts(category)

    primary = models_mod.run_primary(
        category, prior, evidence, impacts, spec.typical_duration_months, n_sims=n_sims,
    )
    critic = models_mod.run_critic(
        category, prior, evidence, impacts, spec.typical_duration_months, n_sims=n_sims,
    )

    w_p, w_c = cal_mod.get_model_weights(category)
    rec = reconcile_mod.reconcile(
        primary, critic, primary_calibration=w_p, critic_calibration=w_c,
    )

    osint = osint_mod.gather_osint(trigger, spec.osint_signals, secrets=secrets)

    market_prob = None
    cmp_bundle = None
    if include_comparison:
        cmp_bundle = cmp_mod.compare(trigger, rec.point_probability)
        market_prob = cmp_bundle.consensus_market_prob()

    tail = reconcile_mod.tail_risk(rec.combined_impact_samples)
    spillover = contagion_mod.cross_category_spillover(
        category, tail["expected_loss"], rec.point_probability,
    )

    return ForecastResult(
        trigger=trigger,
        category=category,
        primary_p=primary.posterior_probability,
        primary_method=primary.methodology,
        critic_p=critic.posterior_probability,
        critic_method=critic.methodology,
        point_p=rec.point_probability,
        band_low=rec.band_low,
        band_high=rec.band_high,
        disagreement=rec.disagreement,
        disagreement_flag=rec.disagreement_flag,
        impact_samples=rec.combined_impact_samples,
        duration_samples=rec.combined_duration_samples,
        tail_metrics=tail,
        osint_bundle=osint,
        market_prob=market_prob,
        comparison=cmp_bundle,
        contagion_spillover=spillover,
        evidence_used=[
            {"name": e.name, "likelihood_ratio": e.likelihood_ratio, "confidence": e.confidence}
            for e in evidence
        ],
        primary_weight=rec.primary_weight,
        critic_weight=rec.critic_weight,
        notes=list(rec.notes),
    )


def forecast_to_explanation_state(fr: ForecastResult) -> Dict[str, Any]:
    """Pack a forecast result into the dict shape expected by explanation.explain()."""
    osint_signals = []
    for s in fr.osint_bundle.signals:
        osint_signals.append({
            "source": s.source, "value": s.value,
            "interpretation": s.interpretation, "error": s.error,
        })
    return {
        "trigger": fr.trigger,
        "category": fr.category,
        "point_p": fr.point_p,
        "band_low": fr.band_low,
        "band_high": fr.band_high,
        "primary_p": fr.primary_p,
        "primary_method": fr.primary_method,
        "critic_p": fr.critic_p,
        "critic_method": fr.critic_method,
        "disagreement": fr.disagreement,
        "evidence": fr.evidence_used,
        "osint_signals": osint_signals,
        "market_prob": fr.market_prob,
        "expected_loss": fr.tail_metrics.get("expected_loss", 0),
        "var_99": fr.tail_metrics.get("VaR_99", 0),
    }
