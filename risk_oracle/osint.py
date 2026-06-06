"""
OSINT layer — fetches real-world signals for verification.

All functions are designed to degrade gracefully: if the API is unreachable
or the key is missing, they return None and the verification layer reports
which signals were available.

Most sources here are FREE and require no key. FRED and ACLED require free keys.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import urllib.parse
import requests


REQUEST_TIMEOUT = 12  # seconds


@dataclass
class OSINTSignal:
    source: str
    label: str
    value: Any
    interpretation: str = ""
    raw: Any = None
    error: Optional[str] = None
    noise_level: str = "medium"  # V2.2: "low" / "medium" / "high"


@dataclass
class OSINTBundle:
    signals: List[OSINTSignal] = field(default_factory=list)
    sources_queried: List[str] = field(default_factory=list)
    sources_succeeded: List[str] = field(default_factory=list)

    def concordance(self, expected_direction: int) -> float:
        """Compute concordance score: fraction of signals consistent with the
        expected direction (1 = event likely, -1 = event unlikely).

        Each signal must have a numeric `value` and interpretation hint.
        Naive default: signals with positive value agree with positive direction.
        """
        if not self.signals:
            return 0.5  # no information
        hits = 0
        scored = 0
        for s in self.signals:
            if s.error or not isinstance(s.value, (int, float)):
                continue
            scored += 1
            if (s.value > 0 and expected_direction > 0) or (s.value < 0 and expected_direction < 0):
                hits += 1
        return hits / scored if scored else 0.5

    def weighted_concordance(self, expected_direction: int) -> float:
        """V2.2: Noise-aware concordance. Low-noise sources (regulatory,
        on-chain, official macro) carry more weight than high-noise (social,
        manipulable prediction markets in early stages)."""
        weights = {"low": 1.0, "medium": 0.6, "high": 0.3}
        if not self.signals:
            return 0.5
        weighted_hits = 0.0
        weighted_total = 0.0
        for s in self.signals:
            if s.error or not isinstance(s.value, (int, float)):
                continue
            w = weights.get(s.noise_level, 0.6)
            weighted_total += w
            if (s.value > 0 and expected_direction > 0) or (s.value < 0 and expected_direction < 0):
                weighted_hits += w
        return weighted_hits / weighted_total if weighted_total > 0 else 0.5


# ---------- GDELT (free, no auth) ----------

def fetch_gdelt(query: str, lookback_days: int = 30) -> Optional[OSINTSignal]:
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "timelinevolinfo",
            "format": "json",
            "timespan": f"{lookback_days}d",
        }
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return OSINTSignal(
                source="gdelt",
                label="news_volume_30d",
                value=0,
                interpretation="No GDELT articles matched query.",
            )
        # Sum volume across the period
        points = timeline[0].get("data", [])
        total_volume = sum(p.get("value", 0) for p in points)
        return OSINTSignal(
            source="gdelt",
            label="news_volume_30d",
            value=float(total_volume),
            interpretation=(
                "High news volume suggests the event is active in global media. "
                f"Total mentions over {lookback_days}d: {int(total_volume)}."
            ),
            raw=timeline,
        )
    except Exception as e:
        return OSINTSignal(source="gdelt", label="news_volume_30d", value=None, error=str(e))


# ---------- FRED (free, requires key) ----------

def fetch_fred_series(series_id: str, api_key: str) -> Optional[OSINTSignal]:
    if not api_key:
        return OSINTSignal(source="fred", label=series_id, value=None, error="No FRED API key")
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 12,
        }
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        obs = data.get("observations", [])
        if not obs:
            return OSINTSignal(source="fred", label=series_id, value=None,
                               error="No observations returned")
        recent = [float(o["value"]) for o in obs[:3] if o["value"] != "."]
        older = [float(o["value"]) for o in obs[6:12] if o["value"] != "."]
        if not recent or not older:
            return OSINTSignal(source="fred", label=series_id, value=None,
                               error="Insufficient data")
        change = (sum(recent) / len(recent)) - (sum(older) / len(older))
        return OSINTSignal(
            source="fred",
            label=series_id,
            value=change,
            interpretation=(
                f"{series_id} recent 3-period avg vs prior 6-period avg: "
                f"{change:+.4f}. Direction matters for macro signals."
            ),
            raw=obs,
        )
    except Exception as e:
        return OSINTSignal(source="fred", label=series_id, value=None, error=str(e))


# ---------- World Bank (free, no auth) ----------

def fetch_world_bank_indicator(country: str, indicator: str) -> Optional[OSINTSignal]:
    try:
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        params = {"format": "json", "per_page": 5}
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            return OSINTSignal(source="world_bank", label=indicator, value=None,
                               error="No data")
        recent = next((d for d in data[1] if d.get("value") is not None), None)
        if not recent:
            return OSINTSignal(source="world_bank", label=indicator, value=None,
                               error="No recent value")
        return OSINTSignal(
            source="world_bank",
            label=f"{country}/{indicator}",
            value=float(recent["value"]),
            interpretation=f"{indicator} for {country} ({recent.get('date','')}): {recent['value']}",
            raw=recent,
        )
    except Exception as e:
        return OSINTSignal(source="world_bank", label=indicator, value=None, error=str(e))


# ---------- USGS earthquakes (free, no auth) ----------

def fetch_usgs_recent_quakes(min_magnitude: float = 5.0, lookback_days: int = 30) -> Optional[OSINTSignal]:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%d"),
            "endtime": end.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
        }
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        count = data.get("metadata", {}).get("count", 0)
        return OSINTSignal(
            source="usgs",
            label="quakes_mag5_30d",
            value=float(count),
            interpretation=f"USGS reports {count} earthquakes >M{min_magnitude} in last {lookback_days} days.",
        )
    except Exception as e:
        return OSINTSignal(source="usgs", label="quakes_mag5_30d", value=None, error=str(e))


# ---------- Polymarket (free, public) ----------

def fetch_polymarket_search(query: str, max_markets: int = 5) -> Optional[OSINTSignal]:
    try:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"limit": max_markets, "active": "true", "closed": "false"}
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        markets = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
        q_lower = query.lower()
        matches = [
            m for m in markets
            if any(w in str(m.get("question", "")).lower() for w in q_lower.split() if len(w) > 3)
        ][:max_markets]
        if not matches:
            return OSINTSignal(
                source="polymarket", label="market_implied_prob",
                value=None, interpretation="No matching Polymarket markets found."
            )
        prices = []
        for m in matches:
            try:
                outcome_prices = m.get("outcomePrices", "[]")
                if isinstance(outcome_prices, str):
                    import json as _json
                    outcome_prices = _json.loads(outcome_prices)
                if outcome_prices:
                    prices.append(float(outcome_prices[0]))
            except Exception:
                continue
        if not prices:
            return OSINTSignal(source="polymarket", label="market_implied_prob",
                               value=None, error="No prices parseable")
        avg = sum(prices) / len(prices)
        return OSINTSignal(
            source="polymarket",
            label="market_implied_prob",
            value=avg,
            interpretation=(
                f"Polymarket average implied probability across {len(prices)} "
                f"matching markets: {avg:.1%}."
            ),
            raw=[m.get("question", "") for m in matches],
        )
    except Exception as e:
        return OSINTSignal(source="polymarket", label="market_implied_prob",
                           value=None, error=str(e))


# ---------- Manifold (free, public) ----------

def fetch_manifold_search(query: str, max_markets: int = 5) -> Optional[OSINTSignal]:
    try:
        url = "https://api.manifold.markets/v0/search-markets"
        params = {"term": query, "limit": max_markets}
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        markets = r.json()
        if not markets:
            return OSINTSignal(source="manifold", label="manifold_prob",
                               value=None, interpretation="No matching Manifold markets.")
        probs = [m.get("probability") for m in markets if m.get("probability") is not None]
        if not probs:
            return OSINTSignal(source="manifold", label="manifold_prob",
                               value=None, error="No probability data")
        avg = sum(probs) / len(probs)
        return OSINTSignal(
            source="manifold",
            label="manifold_prob",
            value=avg,
            interpretation=(
                f"Manifold average probability across {len(probs)} matching markets: "
                f"{avg:.1%}."
            ),
            raw=[m.get("question") for m in markets[:5]],
        )
    except Exception as e:
        return OSINTSignal(source="manifold", label="manifold_prob",
                           value=None, error=str(e))


# ---------- V2.1: Fed Speech NLP (macro_financial enrichment) ----------

_FED_HAWKISH = {
    "inflation persistent", "above target", "additional tightening", "restrictive",
    "higher for longer", "vigilance", "wage pressure", "tight labor market",
    "overheating", "hike", "raise rates", "tighten",
}
_FED_DOVISH = {
    "moderating", "easing", "rate cuts", "accommodation", "below target",
    "softening labor", "disinflation", "weakening demand", "downside risks",
    "recession risk", "supportive policy", "lower rates", "weakening economy",
}


def fetch_fed_speeches(query: str = "", lookback_days: int = 14) -> Optional[OSINTSignal]:
    """Net hawkish/dovish tilt of recent Fed speeches.

    Returns OSINTSignal with value in [-1, +1]:
      -1 → strongly dovish (rate-cut bias; risk-on for risk assets)
      +1 → strongly hawkish (rate-hold/hike bias; risk-off)
      0  → balanced / no signal

    The `query` arg is unused but kept for SIGNAL_FETCHERS dispatch parity.
    """
    import xml.etree.ElementTree as ET
    try:
        r = requests.get(
            "https://www.federalreserve.gov/feeds/speeches.xml",
            headers={"User-Agent": "risk_oracle/2.1 contact:local@example.com"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        # Tally hawkish/dovish terms across recent items
        hawkish = 0
        dovish = 0
        items_read = 0
        for item in root.iter("item"):
            text = " ".join([
                (item.findtext("title") or ""),
                (item.findtext("description") or ""),
            ]).lower()
            if not text.strip():
                continue
            items_read += 1
            for t in _FED_HAWKISH:
                if t in text:
                    hawkish += 1
            for t in _FED_DOVISH:
                if t in text:
                    dovish += 1
        if items_read == 0:
            return OSINTSignal(
                source="fed_speeches", label="net_tilt",
                value=0, interpretation="No recent Fed speeches found.",
            )
        total = hawkish + dovish
        if total == 0:
            tilt = 0.0
            label = "balanced"
        else:
            tilt = (hawkish - dovish) / total  # range [-1, +1]
            label = "hawkish" if tilt > 0.15 else ("dovish" if tilt < -0.15 else "balanced")
        return OSINTSignal(
            source="fed_speeches",
            label="net_tilt",
            value=float(tilt),
            interpretation=(
                f"{items_read} recent Fed speeches read; net tilt {label} "
                f"({hawkish} hawkish vs {dovish} dovish term hits)."
            ),
            raw={"items_read": items_read, "hawkish_hits": hawkish, "dovish_hits": dovish},
        )
    except Exception as e:
        return OSINTSignal(
            source="fed_speeches", label="net_tilt", value=None, error=str(e),
        )


# ---------- V2.1: Politician trades (political_regulatory enrichment) ----------

def fetch_politician_trades_volume(query: str = "", lookback_days: int = 14) -> Optional[OSINTSignal]:
    """Volume of recent Senate/House periodic transaction reports as an
    OSINT signal for political_regulatory category.

    Currently a stub. The Senate (efdsearch.senate.gov) and House
    (disclosures-clerk.house.gov) provide PDFs and search but no clean
    public JSON. Aggregators like capitoltrades.com offer JSON APIs but
    require keys. Returns None when no source is configured so the
    OSINT bundle reports it as 'queried but unavailable'.

    To wire a real source: set CAPITOLTRADES_API_KEY in secrets and
    implement the GET against their /v1/trades endpoint.
    """
    return OSINTSignal(
        source="politician_trades",
        label="ptr_volume_14d",
        value=None,
        interpretation=(
            "Politician trades source unconfigured. Set CAPITOLTRADES_API_KEY "
            "or implement a senate/house PTR scraper to enable."
        ),
    )


# V2.2: on-chain whale flow as an OSINT signal. Mirrors STP's lewis_feeds
# pattern but exposes only the net direction (BUY/SELL) and count so the
# Bayesian update treats it as a single evidence factor.

_WHALE_KNOWN_CEX_TAGS = {
    "binance", "coinbase", "kraken", "okx", "bitfinex", "bybit", "gate.io",
    "kucoin", "bitstamp", "huobi", "gemini", "ftx",
}
_WHALE_USD_THRESHOLD = 5_000_000


def _classify_whale_flow(tx: Dict[str, Any]) -> Optional[str]:
    """CEX→private = accumulation (BUY). private→CEX = distribution (SELL)."""
    from_owner = ((tx.get("from") or {}).get("owner") or "").lower()
    to_owner = ((tx.get("to") or {}).get("owner") or "").lower()
    from_type = ((tx.get("from") or {}).get("owner_type") or "").lower()
    to_type = ((tx.get("to") or {}).get("owner_type") or "").lower()

    from_is_cex = from_type == "exchange" or any(t in from_owner for t in _WHALE_KNOWN_CEX_TAGS)
    to_is_cex = to_type == "exchange" or any(t in to_owner for t in _WHALE_KNOWN_CEX_TAGS)
    if from_is_cex and not to_is_cex:
        return "BUY"
    if to_is_cex and not from_is_cex:
        return "SELL"
    return None


def fetch_whale_transfers(query: str = "",
                          secrets: Optional[Dict[str, str]] = None,
                          lookback_hours: int = 6) -> Optional[OSINTSignal]:
    """Net direction of on-chain whale flow over the last N hours.

    Returns a numeric value in [-1, +1]:
      +1.0 = all classifiable transfers were CEX→private (accumulation)
      -1.0 = all classifiable transfers were private→CEX (distribution)
       0.0 = balanced or no classifiable flow

    Uses Whale Alert API when WHALE_ALERT_API_KEY is set; degrades to a
    null signal otherwise (no free public substitute exists with the same
    quality). Wired into market_specific and cyber_tech categories.
    """
    secrets = secrets or {}
    api_key = (secrets.get("WHALE_ALERT_API_KEY") or "").strip()
    if not api_key:
        return OSINTSignal(
            source="whale_transfers",
            label="whale_flow_unconfigured",
            value=None,
            interpretation=(
                "Whale Alert source unconfigured. Set WHALE_ALERT_API_KEY to "
                "enable on-chain whale-flow signals for market_specific and cyber_tech."
            ),
            noise_level="low",
        )

    try:
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        start = int((_dt.now(_tz.utc) - _td(hours=lookback_hours)).timestamp())
        r = requests.get(
            "https://api.whale-alert.io/v1/transactions",
            params={"api_key": api_key, "min_value": _WHALE_USD_THRESHOLD,
                    "start": start, "limit": 100},
            timeout=12,
        )
        r.raise_for_status()
        txs = r.json().get("transactions", []) or []
    except Exception as e:
        return OSINTSignal(source="whale_transfers", label="whale_flow",
                           value=None, error=str(e), noise_level="low")

    buys = 0
    sells = 0
    for tx in txs:
        flow = _classify_whale_flow(tx)
        if flow == "BUY":
            buys += 1
        elif flow == "SELL":
            sells += 1

    total = buys + sells
    if total == 0:
        flow_score = 0.0
        interp = f"{len(txs)} whale transfers seen in last {lookback_hours}h but none classifiable as CEX↔private."
    else:
        flow_score = (buys - sells) / total
        direction = "accumulation" if flow_score > 0.1 else ("distribution" if flow_score < -0.1 else "balanced")
        interp = (
            f"{total} classifiable whale transfers in last {lookback_hours}h "
            f"({buys} accumulation, {sells} distribution). Net flow score "
            f"{flow_score:+.2f} → {direction}."
        )

    return OSINTSignal(
        source="whale_transfers",
        label="whale_flow_score",
        value=flow_score,
        interpretation=interp,
        raw={"buys": buys, "sells": sells, "total_seen": len(txs)},
        noise_level="low",  # on-chain data is unmanipulable
    )


# ---------- Orchestrator ----------

SIGNAL_FETCHERS = {
    "gdelt": lambda q, _: fetch_gdelt(q),
    "polymarket": lambda q, _: fetch_polymarket_search(q),
    "metaculus": lambda q, _: fetch_manifold_search(q),  # Manifold is a free substitute
    "manifold": lambda q, _: fetch_manifold_search(q),
    "fred": lambda q, sec: fetch_fred_series("DFF", sec.get("FRED_API_KEY", "")),
    "world_bank": lambda q, _: fetch_world_bank_indicator("WLD", "NY.GDP.MKTP.KD.ZG"),
    "usgs": lambda q, _: fetch_usgs_recent_quakes(),
    "noaa": lambda q, _: None,
    "healthmap": lambda q, _: None,
    "promed": lambda q, _: None,
    "acled": lambda q, _: None,
    "cisa_kev": lambda q, _: None,
    "have_i_been_pwned": lambda q, _: None,
    "edgar": lambda q, _: None,
    "market_data": lambda q, _: None,
    # V2.1 additions
    "fed_speeches": lambda q, _: fetch_fed_speeches(q),
    "politician_trades": lambda q, _: fetch_politician_trades_volume(q),
    # V2.2 additions
    "whale_transfers": lambda q, sec: fetch_whale_transfers(q, sec),
}


# V2.2: per-source noise classification. Low-noise = regulatory / official /
# on-chain. Medium = aggregated news / data. High = social / early
# prediction-market consensus.
SOURCE_NOISE = {
    # Low noise (authoritative / regulatory / official / on-chain)
    "fred": "low", "world_bank": "low", "usgs": "low", "noaa": "low",
    "edgar": "low", "fed_speeches": "low", "cisa_kev": "low",
    "whale_transfers": "low",
    # Medium noise (aggregated news, market data)
    "gdelt": "medium", "acled": "medium", "market_data": "medium",
    "healthmap": "medium", "promed": "medium", "have_i_been_pwned": "medium",
    # High noise (prediction markets in early stages can be manipulated;
    # politician trades is filtered insider info but inconsistent reporting)
    "polymarket": "high", "metaculus": "medium", "manifold": "high",
    "politician_trades": "medium",
}


def gather_osint(query: str, signal_keys: List[str],
                 secrets: Optional[Dict[str, str]] = None) -> OSINTBundle:
    secrets = secrets or {}
    bundle = OSINTBundle()
    for key in signal_keys:
        bundle.sources_queried.append(key)
        fetcher = SIGNAL_FETCHERS.get(key)
        if fetcher is None:
            continue
        try:
            sig = fetcher(query, secrets)
        except Exception as e:
            sig = OSINTSignal(source=key, label=key, value=None, error=str(e))
        if sig is None:
            continue
        # V2.2: tag noise level if the fetcher didn't already
        if sig.noise_level == "medium":
            sig.noise_level = SOURCE_NOISE.get(sig.source, "medium")
        bundle.signals.append(sig)
        if sig.error is None and sig.value is not None:
            bundle.sources_succeeded.append(key)
    return bundle
