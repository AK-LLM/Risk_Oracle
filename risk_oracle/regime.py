"""
regime.py — V2.2 regime detection for Risk Oracle.

Ports STP's regime-context awareness (VIX-driven panic/elevated/normal/
complacent classification) so Risk Oracle's reconcile and dispatch can
modulate behaviour based on the broader market state.

Single function: `detect_regime(secrets)` returns a dict with the regime
label, the underlying VIX value, and a short explanatory note.

Used by:
  • pipeline.reconcile  — when regime == "panic", widen the disagreement
                          threshold (don't penalise primary/critic disagreement
                          as harshly; uncertainty is *expected*)
  • dispatch            — when regime == "panic", lower the alert threshold
                          (so smaller moves get flagged because they matter more)
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import requests


REGIME_PANIC = "panic"
REGIME_ELEVATED = "elevated"
REGIME_NORMAL = "normal"
REGIME_COMPLACENT = "complacent"
REGIME_UNKNOWN = "unknown"


VIX_PANIC = 30.0
VIX_ELEVATED = 20.0
VIX_COMPLACENT = 13.0


def _fetch_vix_from_fred(api_key: str) -> Optional[float]:
    """Pull latest VIX close from FRED (series VIXCLS). Free with key."""
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "VIXCLS",
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=10,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        for o in obs:
            val = o.get("value", ".")
            if val and val != ".":
                try:
                    return float(val)
                except ValueError:
                    continue
        return None
    except Exception:
        return None


def _fetch_vix_from_yahoo() -> Optional[float]:
    """Fallback: Yahoo Finance public quote for ^VIX. No key, but rate-limited.
    This is a best-effort fallback — if it fails we return None gracefully."""
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            params={"symbols": "^VIX"},
            headers={"User-Agent": "Mozilla/5.0 (risk_oracle/2.2)"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        quotes = (data.get("quoteResponse", {}) or {}).get("result", []) or []
        if not quotes:
            return None
        price = quotes[0].get("regularMarketPrice")
        return float(price) if price is not None else None
    except Exception:
        return None


def detect_regime(secrets: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Classify the current market regime from VIX.

    Tries FRED first (cleanest), falls back to Yahoo. If both fail, returns
    REGIME_UNKNOWN so callers can degrade gracefully.
    """
    secrets = secrets or {}
    vix = _fetch_vix_from_fred(secrets.get("FRED_API_KEY", ""))
    if vix is None:
        vix = _fetch_vix_from_yahoo()

    if vix is None:
        return {
            "regime": REGIME_UNKNOWN,
            "vix": None,
            "note": "Could not fetch VIX from FRED or Yahoo. Regime context unavailable.",
        }

    if vix >= VIX_PANIC:
        label = REGIME_PANIC
        note = f"VIX {vix:.1f} — panic regime; expect wide reconciliation bands and frequent moves."
    elif vix >= VIX_ELEVATED:
        label = REGIME_ELEVATED
        note = f"VIX {vix:.1f} — elevated regime; volatility above norm."
    elif vix <= VIX_COMPLACENT:
        label = REGIME_COMPLACENT
        note = f"VIX {vix:.1f} — complacent regime; tail risks may be underpriced."
    else:
        label = REGIME_NORMAL
        note = f"VIX {vix:.1f} — normal regime."

    return {"regime": label, "vix": float(vix), "note": note}


def regime_disagreement_threshold(regime: str, default: float = 0.15) -> float:
    """Higher threshold in panic = don't penalise primary/critic disagreement
    as harshly because uncertainty is expected."""
    return {
        REGIME_PANIC: default * 1.8,
        REGIME_ELEVATED: default * 1.3,
        REGIME_NORMAL: default,
        REGIME_COMPLACENT: default * 0.8,
        REGIME_UNKNOWN: default,
    }.get(regime, default)


def regime_alert_threshold(regime: str, default: float = 0.05) -> float:
    """In panic, smaller moves matter more — lower alert threshold so they fire."""
    return {
        REGIME_PANIC: default * 0.6,        # 3pp instead of 5pp
        REGIME_ELEVATED: default * 0.8,
        REGIME_NORMAL: default,
        REGIME_COMPLACENT: default * 1.2,
        REGIME_UNKNOWN: default,
    }.get(regime, default)
