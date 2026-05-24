"""
Evidence elicitation assistant.

User describes the situation in plain English; LLM extracts structured
evidence factors (name, likelihood ratio, confidence, rationale) that drop
straight into the Bayesian update.

Falls back to None if no LLM key is configured — the UI then routes the user
to manual evidence entry.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ExtractedEvidence:
    name: str
    likelihood_ratio: float
    confidence: float
    rationale: str


_SYSTEM_PROMPT = """You are an evidence-elicitation assistant for a Bayesian risk forecasting system.

The user has a TRIGGER question (e.g., "Will X happen by date Y?") and will describe what they currently know about the situation. Your job is to extract individual evidence factors and estimate each one's likelihood ratio (LR).

Likelihood ratio definition:
- LR = P(evidence | event happens) / P(evidence | event doesn't happen)
- LR > 1 raises the posterior probability
- LR < 1 lowers it
- LR = 1 is neutral (no information)
- Typical strong signal: LR ~ 3-5 (or 0.2-0.33 in the other direction)
- Very strong signal: LR ~ 8-10 (or 0.1)
- Weak signal: LR ~ 1.3-2 (or 0.5-0.7)

Confidence (0 to 1) represents how sure YOU are of the LR estimate.

Extract 2-6 distinct evidence factors. Return ONLY a JSON array, nothing else:
[
  {
    "name": "<short label, <= 60 chars>",
    "likelihood_ratio": <float, e.g. 2.5 or 0.4>,
    "confidence": <float 0-1>,
    "rationale": "<one sentence justification, <= 150 chars>"
  },
  ...
]

Be calibrated — don't claim LR=10 for weak evidence. Most real evidence is in the 1.3-3 range. If the user describes mixed signals, return factors in both directions."""


def _strip_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)


def _parse_llm_response(content: str) -> Optional[List[ExtractedEvidence]]:
    content = _strip_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list):
        return None
    out: List[ExtractedEvidence] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ExtractedEvidence(
                name=str(item.get("name", "")).strip()[:80],
                likelihood_ratio=float(item.get("likelihood_ratio", 1.0)),
                confidence=float(item.get("confidence", 0.7)),
                rationale=str(item.get("rationale", "")).strip()[:200],
            ))
        except (TypeError, ValueError):
            continue
    return out or None


def extract_evidence(
    trigger: str,
    situation: str,
    category: str,
    secrets: Dict[str, str],
) -> Optional[List[ExtractedEvidence]]:
    """Use LLM to extract evidence factors. Returns None if no LLM is available."""
    if not trigger.strip() or not situation.strip():
        return None

    user_msg = (
        f"CATEGORY: {category}\n"
        f"TRIGGER: {trigger}\n\n"
        f"WHAT I KNOW ABOUT THE CURRENT SITUATION:\n{situation}\n\n"
        f"Extract evidence factors."
    )

    anthropic_key = secrets.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return _parse_llm_response(msg.content[0].text)
        except Exception:
            pass

    openai_key = secrets.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1200,
                temperature=0.0,
            )
            return _parse_llm_response(resp.choices[0].message.content or "")
        except Exception:
            pass

    return None
