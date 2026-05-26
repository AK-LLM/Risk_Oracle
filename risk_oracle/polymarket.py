"""
Polymarket integration — deeper than the comparison-only OSINT signal.

Provides:
- Browse most-active / most-liquid markets
- Search markets matching a trigger question
- Per-market: price, liquidity, volume, end date, URL
- All via the free public Gamma API (no auth required)

Used by the forecast tab's "edge analysis" panel and the Bets tab.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import requests


REQUEST_TIMEOUT = 12
GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass
class PolymarketMarket:
    id: str
    question: str
    slug: str
    yes_price: Optional[float]      # 0–1; None if no live price
    no_price: Optional[float]       # 0–1
    volume_24h_usd: float
    volume_total_usd: float
    liquidity_usd: float
    end_date: Optional[str]
    category: Optional[str]
    description: str = ""
    closed: bool = False
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}" if self.slug else "https://polymarket.com"

    @property
    def days_until_close(self) -> Optional[int]:
        if not self.end_date:
            return None
        try:
            end_dt = datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
            return max(0, (end_dt - datetime.utcnow().replace(tzinfo=end_dt.tzinfo)).days)
        except Exception:
            return None


def _parse_market(raw: Dict[str, Any]) -> Optional[PolymarketMarket]:
    if not isinstance(raw, dict):
        return None

    def _to_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return []

    outcome_prices = _to_list(raw.get("outcomePrices", []))
    clob_token_ids = _to_list(raw.get("clobTokenIds", []))

    yes_price = None
    no_price = None
    if outcome_prices:
        try:
            yes_price = float(outcome_prices[0])
            if len(outcome_prices) > 1:
                no_price = float(outcome_prices[1])
            else:
                no_price = 1.0 - yes_price
        except (TypeError, ValueError):
            pass

    yes_tok = str(clob_token_ids[0]) if len(clob_token_ids) > 0 else None
    no_tok = str(clob_token_ids[1]) if len(clob_token_ids) > 1 else None

    def _f(key, default=0.0):
        v = raw.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return PolymarketMarket(
        id=str(raw.get("id", "")),
        question=str(raw.get("question", "")).strip(),
        slug=str(raw.get("slug", "")),
        yes_price=yes_price,
        no_price=no_price,
        volume_24h_usd=_f("volume24hr"),
        volume_total_usd=_f("volume"),
        liquidity_usd=_f("liquidity"),
        end_date=raw.get("endDate"),
        category=raw.get("category") or (raw.get("tags", [{}])[0] or {}).get("label"),
        description=str(raw.get("description", "")).strip()[:500],
        closed=bool(raw.get("closed", False)),
        yes_token_id=yes_tok,
        no_token_id=no_tok,
    )


def _fetch_markets_raw(limit: int = 50,
                       sort_by: str = "volume",
                       offset: int = 0) -> List[Dict[str, Any]]:
    """Fetch active, non-closed markets from Gamma."""
    order_key = {
        "volume": "volume24hr",
        "liquidity": "liquidity",
        "newest": "createdAt",
    }.get(sort_by, "volume24hr")
    params = {
        "limit": limit,
        "offset": offset,
        "active": "true",
        "closed": "false",
        "order": order_key,
        "ascending": "false",
    }
    r = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    raw = r.json()
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("data", []) or []
    return []


def top_markets(limit: int = 20, sort_by: str = "volume") -> List[PolymarketMarket]:
    """Return top live markets sorted by 24h volume or liquidity."""
    try:
        raw = _fetch_markets_raw(limit=limit, sort_by=sort_by)
        out = [_parse_market(r) for r in raw]
        return [m for m in out if m and m.question]
    except Exception:
        return []


def search_markets(query: str, limit: int = 10,
                   search_pool: int = 200) -> List[PolymarketMarket]:
    """Find live markets whose question/description matches the query.

    Pulls a pool of top markets (by volume) and scores by keyword overlap.
    Polymarket's Gamma API doesn't have a great native text search, so this
    matches the standard approach used elsewhere in the system.
    """
    try:
        pool_raw = _fetch_markets_raw(limit=search_pool, sort_by="volume")
        pool = [m for m in (_parse_market(r) for r in pool_raw) if m and m.question]
    except Exception:
        return []

    q_words = [w.lower().strip(".,?!") for w in query.split() if len(w) > 2]
    if not q_words:
        return pool[:limit]

    scored: List[tuple] = []
    for m in pool:
        text = (m.question + " " + m.description + " " + (m.category or "")).lower()
        score = sum(1 for w in q_words if w in text)
        if score > 0:
            scored.append((score, m.volume_24h_usd, m))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [m for _, _, m in scored[:limit]]


def get_market(market_id: str) -> Optional[PolymarketMarket]:
    try:
        r = requests.get(f"{GAMMA_BASE}/markets/{market_id}", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return _parse_market(r.json())
    except Exception:
        return None


CATEGORY_HINT_KEYWORDS = {
    "geopolitical": ["war", "conflict", "ukraine", "iran", "israel", "russia", "china", "nato"],
    "macro_financial": ["recession", "inflation", "fed", "rate", "gdp", "unemployment"],
    "market_specific": ["price", "stock", "btc", "bitcoin", "ethereum", "tesla", "spx"],
    "epidemic": ["outbreak", "virus", "pandemic", "ebola", "h5n1"],
    "natural_hazard": ["hurricane", "earthquake", "wildfire"],
    "cyber_tech": ["hack", "breach", "ai", "agi"],
    "political_regulatory": ["election", "vote", "approval", "ban", "ruling"],
}


def guess_category(market: PolymarketMarket) -> str:
    text = (market.question + " " + market.description + " " + (market.category or "")).lower()
    scores = {cat: sum(1 for kw in kws if kw in text)
              for cat, kws in CATEGORY_HINT_KEYWORDS.items()}
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "political_regulatory"
