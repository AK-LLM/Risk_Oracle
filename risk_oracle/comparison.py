"""
Comparison view — aggregate external prediction-market probabilities for
side-by-side display against our reconciled forecast.

Sources: Polymarket, Manifold, Metaculus. All free, no auth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import requests

from . import osint as osint_mod


REQUEST_TIMEOUT = 12


@dataclass
class MarketComparison:
    source: str
    probability: Optional[float]   # 0-1 if available
    question_matched: str = ""
    n_markets: int = 0
    note: str = ""


@dataclass
class ComparisonBundle:
    our_probability: float
    items: List[MarketComparison] = field(default_factory=list)

    def consensus_market_prob(self) -> Optional[float]:
        probs = [i.probability for i in self.items if i.probability is not None]
        if not probs:
            return None
        return sum(probs) / len(probs)

    def max_disagreement(self) -> Optional[float]:
        consensus = self.consensus_market_prob()
        if consensus is None:
            return None
        return abs(self.our_probability - consensus)


# ---------- Metaculus ----------

def fetch_metaculus(query: str, max_questions: int = 5) -> MarketComparison:
    try:
        url = "https://www.metaculus.com/api2/questions/"
        params = {"search": query, "limit": max_questions, "status": "open"}
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        probs = []
        matched_titles = []
        for q in results:
            cp = q.get("community_prediction", {})
            # Prefer full distribution mean, fall back to median
            full = cp.get("full", {}) if isinstance(cp, dict) else {}
            p = full.get("q2") if isinstance(full, dict) else None
            if p is None:
                p = full.get("mean") if isinstance(full, dict) else None
            if isinstance(p, (int, float)) and 0 <= p <= 1:
                probs.append(float(p))
                matched_titles.append(q.get("title", "")[:80])
        if not probs:
            return MarketComparison(
                source="metaculus", probability=None,
                n_markets=0,
                note="No matching open questions on Metaculus.",
            )
        avg = sum(probs) / len(probs)
        return MarketComparison(
            source="metaculus",
            probability=avg,
            question_matched="; ".join(matched_titles[:3]),
            n_markets=len(probs),
            note=f"Average community prediction across {len(probs)} matching questions.",
        )
    except Exception as e:
        return MarketComparison(source="metaculus", probability=None,
                                note=f"error: {e}")


# ---------- Polymarket wrapper ----------

def fetch_polymarket(query: str) -> MarketComparison:
    sig = osint_mod.fetch_polymarket_search(query, max_markets=5)
    if sig is None or sig.error or sig.value is None:
        return MarketComparison(
            source="polymarket", probability=None,
            note=sig.error if sig and sig.error else "no matching markets",
        )
    raw = sig.raw if isinstance(sig.raw, list) else []
    return MarketComparison(
        source="polymarket",
        probability=float(sig.value),
        question_matched="; ".join(str(q)[:80] for q in raw[:3]),
        n_markets=len(raw),
        note=sig.interpretation,
    )


# ---------- Manifold wrapper ----------

def fetch_manifold(query: str) -> MarketComparison:
    sig = osint_mod.fetch_manifold_search(query, max_markets=5)
    if sig is None or sig.error or sig.value is None:
        return MarketComparison(
            source="manifold", probability=None,
            note=sig.error if sig and sig.error else "no matching markets",
        )
    raw = sig.raw if isinstance(sig.raw, list) else []
    return MarketComparison(
        source="manifold",
        probability=float(sig.value),
        question_matched="; ".join(str(q)[:80] for q in raw[:3]),
        n_markets=len(raw),
        note=sig.interpretation,
    )


def compare(query: str, our_probability: float) -> ComparisonBundle:
    """Run all comparison sources in sequence; return a bundle."""
    bundle = ComparisonBundle(our_probability=our_probability)
    for fn in (fetch_polymarket, fetch_manifold, fetch_metaculus):
        try:
            bundle.items.append(fn(query))
        except Exception as e:
            bundle.items.append(MarketComparison(
                source=fn.__name__, probability=None, note=f"error: {e}"
            ))
    return bundle
