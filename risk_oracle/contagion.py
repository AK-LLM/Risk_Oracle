"""
Cross-category contagion — propagates an event's impact across the taxonomy.

When a trigger fires in category A, this layer estimates the secondary impact
in categories B, C, etc. using a fixed interaction matrix calibrated from
broad historical co-occurrence (e.g., wars affect markets; cyber breaches
affect corporate reputation; macro recessions affect everything).
"""
from __future__ import annotations
from typing import Dict, Tuple

# Interaction strengths: rows = source category, cols = target category.
# Values in 0-1 represent how much the source category propagates to the target.
# 1.0 = same as direct impact; 0 = no propagation.
# These are illustrative defaults; in production they would be calibrated from
# historical event-impact datasets.
INTERACTION_MATRIX: Dict[str, Dict[str, float]] = {
    "geopolitical": {
        "macro_financial": 0.55,
        "market_specific": 0.65,
        "epidemic": 0.10,
        "natural_hazard": 0.05,
        "cyber_tech": 0.30,
        "operational_corporate": 0.40,
        "political_regulatory": 0.50,
    },
    "macro_financial": {
        "geopolitical": 0.25,
        "market_specific": 0.85,
        "epidemic": 0.05,
        "natural_hazard": 0.05,
        "cyber_tech": 0.20,
        "operational_corporate": 0.55,
        "political_regulatory": 0.45,
    },
    "market_specific": {
        "geopolitical": 0.10,
        "macro_financial": 0.40,
        "epidemic": 0.05,
        "natural_hazard": 0.05,
        "cyber_tech": 0.15,
        "operational_corporate": 0.40,
        "political_regulatory": 0.20,
    },
    "epidemic": {
        "geopolitical": 0.25,
        "macro_financial": 0.65,
        "market_specific": 0.55,
        "natural_hazard": 0.05,
        "cyber_tech": 0.15,
        "operational_corporate": 0.50,
        "political_regulatory": 0.45,
    },
    "natural_hazard": {
        "geopolitical": 0.15,
        "macro_financial": 0.30,
        "market_specific": 0.40,
        "epidemic": 0.20,
        "cyber_tech": 0.10,
        "operational_corporate": 0.55,
        "political_regulatory": 0.30,
    },
    "cyber_tech": {
        "geopolitical": 0.25,
        "macro_financial": 0.15,
        "market_specific": 0.35,
        "epidemic": 0.05,
        "natural_hazard": 0.02,
        "operational_corporate": 0.65,
        "political_regulatory": 0.40,
    },
    "operational_corporate": {
        "geopolitical": 0.05,
        "macro_financial": 0.15,
        "market_specific": 0.50,
        "epidemic": 0.05,
        "natural_hazard": 0.05,
        "cyber_tech": 0.15,
        "political_regulatory": 0.35,
    },
    "political_regulatory": {
        "geopolitical": 0.30,
        "macro_financial": 0.35,
        "market_specific": 0.40,
        "epidemic": 0.05,
        "natural_hazard": 0.05,
        "cyber_tech": 0.20,
        "operational_corporate": 0.45,
    },
}


def cross_category_spillover(
    source_category: str,
    source_expected_loss: float,
    source_probability: float,
) -> Dict[str, Dict[str, float]]:
    """For a source event's expected loss and probability, estimate the
    spillover expected loss in each other category.

    Returns: { target_category: { "spillover_loss": $, "weight": 0-1 } }
    """
    if source_category not in INTERACTION_MATRIX:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for target, weight in INTERACTION_MATRIX[source_category].items():
        spillover_loss = source_expected_loss * weight * source_probability
        out[target] = {
            "spillover_loss": float(spillover_loss),
            "weight": float(weight),
        }
    return out


def total_contagion_loss(
    source_category: str,
    source_expected_loss: float,
    source_probability: float,
) -> float:
    spill = cross_category_spillover(source_category, source_expected_loss, source_probability)
    return sum(v["spillover_loss"] for v in spill.values())
