"""
Natural language explanation layer.

Takes the full forecast state (trigger, models, OSINT, comparison) and
produces a 2-3 paragraph explanation. Uses LLM if available; falls back to
a structured template otherwise.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any


_SYSTEM_PROMPT = """You explain probabilistic risk forecasts to sophisticated readers.

You are given the full state of a forecast: the trigger question, the reconciled probability and band, primary and critic model outputs, the evidence used, OSINT signals, and external market comparison.

Write a 2-3 paragraph explanation that:
1. States the probability and confidence band clearly, and what they mean in plain English
2. Names the 1-3 evidence factors doing the heaviest work in the forecast
3. Explains where primary and critic agree or disagree, and what's behind the disagreement
4. Notes the OSINT signal direction (does the live data support or challenge the forecast?)
5. Comments on the external markets if present (are we above or below them, and why might that be?)
6. Is honest about uncertainty — if the band is wide, say so; if external sources strongly disagree, flag it

Style: direct, no hedging language like "it is important to note", no recap of the inputs, no bullet points. Just the explanation in flowing prose. Total length 200-350 words."""


def _build_user_message(state: Dict[str, Any]) -> str:
    ev_lines = []
    for ev in state.get("evidence", []):
        ev_lines.append(
            f"  - {ev['name']}: LR={ev['likelihood_ratio']}, confidence={ev['confidence']}"
        )
    osint_lines = []
    for sig in state.get("osint_signals", []):
        if sig.get("error"):
            continue
        if sig.get("value") is None:
            continue
        osint_lines.append(f"  - {sig['source']}: {sig.get('interpretation','')}")

    market_section = ""
    if state.get("market_prob") is not None:
        market_section = (
            f"EXTERNAL PREDICTION MARKETS: average probability {state['market_prob']:.1%}\n"
            f"DELTA from our forecast: {(state['point_p'] - state['market_prob']):+.1%}\n"
        )

    return (
        f"TRIGGER: {state['trigger']}\n"
        f"CATEGORY: {state['category']}\n"
        f"RECONCILED PROBABILITY: {state['point_p']:.1%}  "
        f"(band {state['band_low']:.1%} – {state['band_high']:.1%})\n"
        f"PRIMARY MODEL ({state['primary_method']}): {state['primary_p']:.1%}\n"
        f"CRITIC MODEL ({state['critic_method']}): {state['critic_p']:.1%}\n"
        f"DISAGREEMENT: {state['disagreement']:.1%}\n\n"
        f"EVIDENCE FACTORS:\n"
        + "\n".join(ev_lines) + "\n\n"
        + f"OSINT SIGNALS:\n"
        + ("\n".join(osint_lines) if osint_lines else "  (no signals available)") + "\n\n"
        + market_section
        + f"EXPECTED LOSS (Monte Carlo mean): ${state.get('expected_loss', 0):,.0f}\n"
        + f"VaR 99%: ${state.get('var_99', 0):,.0f}\n"
    )


def explain(state: Dict[str, Any], secrets: Dict[str, str]) -> str:
    """Return a 2-3 paragraph explanation. LLM-powered if key present, else template."""
    anthropic_key = secrets.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=900,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_message(state)}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return _template_explanation(state) + f"\n\n*(LLM explanation unavailable: {e}; using template fallback.)*"

    openai_key = secrets.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(state)},
                ],
                max_tokens=900,
                temperature=0.4,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass

    return _template_explanation(state)


def _template_explanation(state: Dict[str, Any]) -> str:
    """Programmatic fallback when no LLM is available."""
    direction = "above" if state.get("market_prob") is not None and state["point_p"] > state["market_prob"] else "below"
    band_width = state["band_high"] - state["band_low"]
    confidence_phrase = (
        "wide uncertainty band, reflecting genuine model disagreement"
        if band_width > 0.25
        else "moderate uncertainty band"
        if band_width > 0.12
        else "narrow uncertainty band"
    )

    osint_summary = ""
    osint_signals = [s for s in state.get("osint_signals", [])
                     if not s.get("error") and s.get("value") is not None]
    if osint_signals:
        osint_summary = f"OSINT verification queried {len(osint_signals)} live signal(s): " + \
                        "; ".join(s.get("interpretation", s["source"]) for s in osint_signals[:3])
    else:
        osint_summary = "OSINT verification returned no actionable signals for this trigger."

    market_part = ""
    if state.get("market_prob") is not None:
        market_part = (
            f" External prediction markets average {state['market_prob']:.1%}, "
            f"placing our forecast {direction} the market consensus."
        )

    top_evidence = sorted(
        state.get("evidence", []),
        key=lambda e: abs(e["likelihood_ratio"] - 1.0) * e.get("confidence", 1),
        reverse=True,
    )[:3]
    ev_part = ""
    if top_evidence:
        ev_part = " The strongest evidence factors driving this estimate are: " + \
                  "; ".join(
                      f"{e['name']} (LR={e['likelihood_ratio']:.1f})"
                      for e in top_evidence
                  ) + "."

    disagreement_part = (
        f"Primary ({state['primary_p']:.1%}) and critic ({state['critic_p']:.1%}) "
        f"differ by {state['disagreement']:.1%}, "
        + ("which is material and the band has been widened to reflect this."
           if state['disagreement'] > 0.15
           else "which is small — the models broadly agree.")
    )

    return (
        f"The reconciled probability is **{state['point_p']:.1%}** with a "
        f"{confidence_phrase} of [{state['band_low']:.1%}, {state['band_high']:.1%}]. "
        f"{ev_part} {disagreement_part}{market_part}\n\n"
        f"{osint_summary}\n\n"
        f"For decision-making: the band, not the point, is what matters. The point estimate "
        f"is a summary; the band reflects what the system honestly doesn't know."
    )
