"""
Smart router — classifies a natural-language trigger into the taxonomy
and assigns primary + critic models.

Two modes:
  1. LLM-powered (if ANTHROPIC_API_KEY or OPENAI_API_KEY is configured) —
     extracts structured features and reasons about category fit.
  2. Rule-based fallback — keyword matching against the taxonomy.

The router also runs the black-swan / out-of-taxonomy detector.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .taxonomy import TAXONOMY, CATEGORY_KEYS, CategorySpec, get_category


@dataclass
class RoutingDecision:
    primary_category: str
    secondary_categories: List[str] = field(default_factory=list)
    confidence: float = 0.5
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    is_anomaly: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)
    reasoning: str = ""
    used_llm: bool = False

    @property
    def primary_spec(self) -> CategorySpec:
        return get_category(self.primary_category)

    @property
    def secondary_specs(self) -> List[CategorySpec]:
        return [get_category(c) for c in self.secondary_categories]


# ---------- Rule-based router ----------

def _score_keywords(text: str) -> Dict[str, float]:
    """Keyword-overlap score per category."""
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    for key, spec in TAXONOMY.items():
        hits = sum(1 for kw in spec.keywords if kw in text_lower)
        scores[key] = hits / max(len(spec.keywords), 1)
    return scores


def _rule_based_route(trigger: str) -> RoutingDecision:
    scores = _score_keywords(trigger)
    sorted_cats = sorted(scores.items(), key=lambda x: -x[1])
    primary, primary_score = sorted_cats[0]
    secondary = [c for c, s in sorted_cats[1:3] if s > 0.02]

    # No keyword overlap -> probable anomaly
    is_anomaly = primary_score < 0.01
    anomaly_reasons: List[str] = []
    if is_anomaly:
        anomaly_reasons.append(
            "Trigger contains no recognised keywords from the taxonomy — "
            "out-of-distribution candidate."
        )
        # Default to political/regulatory which has broadest applicability
        primary = "political_regulatory"

    # Confidence proportional to keyword density and gap to next category
    next_score = sorted_cats[1][1] if len(sorted_cats) > 1 else 0
    gap = primary_score - next_score
    confidence = min(0.95, 0.3 + 5 * primary_score + 2 * gap)

    return RoutingDecision(
        primary_category=primary,
        secondary_categories=secondary,
        confidence=float(confidence),
        extracted_features={"keyword_scores": scores},
        is_anomaly=is_anomaly,
        anomaly_reasons=anomaly_reasons,
        reasoning="Rule-based: keyword overlap with taxonomy entries.",
        used_llm=False,
    )


# ---------- LLM-based router ----------

_LLM_SYSTEM_PROMPT = """You classify risk-forecast trigger questions into a fixed taxonomy.

Categories (use exact keys):
- geopolitical: wars, escalations, sanctions, regime change, alliances
- macro_financial: recessions, currency crises, rate cycles, sovereign debt
- market_specific: asset prices, sectors, single-name equities, IPOs, M&A
- epidemic: disease outbreaks, pandemics, biosecurity
- natural_hazard: earthquakes, hurricanes, floods, wildfires, climate events
- cyber_tech: breaches, attacks, AI incidents, platform outages
- operational_corporate: fraud, scandals, supply-chain, key-person, ESG
- political_regulatory: elections, legislation, court rulings, regulatory action

You also detect ANOMALY triggers that fit no category (novel risks outside the taxonomy).

Return ONLY a JSON object with these exact keys, no other text:
{
  "primary_category": "<one of the keys above>",
  "secondary_categories": ["<other relevant keys, can be empty>"],
  "confidence": <0.0 to 1.0>,
  "is_anomaly": <true if the trigger doesn't fit any category well>,
  "anomaly_reasons": ["<reasons>" if anomaly else empty list],
  "extracted": {
    "geography": "<country/region or null>",
    "actors": ["<list of named actors>"],
    "time_horizon_months": <integer or null>,
    "quantifiable_outcome": "<what is being predicted, terse>"
  },
  "reasoning": "<one short sentence>"
}"""


def _llm_route_anthropic(trigger: str, api_key: str) -> Optional[RoutingDecision]:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Trigger: {trigger}"}],
        )
        content = msg.content[0].text
        return _parse_llm_response(content)
    except Exception as e:
        return None


def _llm_route_openai(trigger: str, api_key: str) -> Optional[RoutingDecision]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Trigger: {trigger}"},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        return _parse_llm_response(content)
    except Exception:
        return None


def _parse_llm_response(content: str) -> Optional[RoutingDecision]:
    # Strip code fences if present
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try extracting first JSON object found
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    primary = data.get("primary_category", "")
    if primary not in CATEGORY_KEYS:
        return None

    secondary = [c for c in data.get("secondary_categories", []) if c in CATEGORY_KEYS]

    return RoutingDecision(
        primary_category=primary,
        secondary_categories=secondary,
        confidence=float(data.get("confidence", 0.5)),
        extracted_features=data.get("extracted", {}),
        is_anomaly=bool(data.get("is_anomaly", False)),
        anomaly_reasons=data.get("anomaly_reasons", []),
        reasoning=data.get("reasoning", ""),
        used_llm=True,
    )


# ---------- Main entry point ----------

def route(trigger: str, secrets: Optional[Dict[str, str]] = None) -> RoutingDecision:
    """Route a trigger to a primary+secondary category.

    Tries LLM first if API key is available; falls back to rule-based.
    """
    secrets = secrets or {}

    anthropic_key = secrets.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        decision = _llm_route_anthropic(trigger, anthropic_key)
        if decision is not None:
            return decision

    openai_key = secrets.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        decision = _llm_route_openai(trigger, openai_key)
        if decision is not None:
            return decision

    return _rule_based_route(trigger)
