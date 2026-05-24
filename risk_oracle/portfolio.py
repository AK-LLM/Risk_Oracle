"""
Portfolio context layer — track positions, compute exposure to risk categories,
suggest directional hedges.

Each position has category sensitivities: a dict mapping risk-category key to a
sensitivity factor in [0, 1] representing how much a 1-unit move in that
category's risk affects this position. Defaults are provided per common ticker
pattern; user can override.
"""
from __future__ import annotations
import sqlite3
import json
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def _default_db_path() -> str:
    folder = Path.home() / ".risk_oracle"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "portfolio.db")


@contextmanager
def _conn(db_path: str):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db(db_path: Optional[str] = None) -> str:
    db_path = db_path or _default_db_path()
    with _conn(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                description TEXT,
                notional_usd REAL NOT NULL,
                direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
                category_sensitivities TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return db_path


@dataclass
class Position:
    ticker: str
    description: str
    notional_usd: float
    direction: str   # "long" or "short"
    category_sensitivities: Dict[str, float] = field(default_factory=dict)
    id: Optional[int] = None

    def signed_notional(self) -> float:
        return self.notional_usd if self.direction == "long" else -self.notional_usd


# Sensible default sensitivities by asset-class hint.
# 0 = no sensitivity to this risk category; 1 = full sensitivity.
DEFAULT_SENSITIVITIES: Dict[str, Dict[str, float]] = {
    "equity_broad":     {"macro_financial": 0.8, "market_specific": 1.0, "geopolitical": 0.4, "political_regulatory": 0.3},
    "equity_tech":      {"market_specific": 1.0, "cyber_tech": 0.6, "macro_financial": 0.7, "political_regulatory": 0.4},
    "equity_energy":    {"market_specific": 1.0, "geopolitical": 0.85, "macro_financial": 0.6, "natural_hazard": 0.4},
    "equity_financial": {"market_specific": 1.0, "macro_financial": 0.9, "cyber_tech": 0.5, "political_regulatory": 0.6},
    "equity_health":    {"market_specific": 1.0, "epidemic": 0.7, "political_regulatory": 0.5},
    "equity_consumer":  {"market_specific": 1.0, "macro_financial": 0.7, "operational_corporate": 0.4},
    "bond_treasury":    {"macro_financial": 0.9, "political_regulatory": 0.3},
    "bond_corporate":   {"macro_financial": 0.8, "market_specific": 0.5, "operational_corporate": 0.4},
    "commodity_oil":    {"geopolitical": 0.9, "macro_financial": 0.6, "market_specific": 0.7, "natural_hazard": 0.3},
    "commodity_gold":   {"geopolitical": 0.7, "macro_financial": 0.5, "market_specific": -0.3},  # gold often hedges
    "crypto":           {"market_specific": 1.0, "cyber_tech": 0.5, "political_regulatory": 0.6, "macro_financial": 0.6},
    "fx_usd":           {"macro_financial": 0.7, "geopolitical": 0.4},
    "cash":             {},
    "custom":           {},
}


def suggest_sensitivities(asset_class: str) -> Dict[str, float]:
    return DEFAULT_SENSITIVITIES.get(asset_class, {}).copy()


def add_position(pos: Position, db_path: Optional[str] = None) -> int:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        cur = c.execute("""
            INSERT INTO positions
            (ticker, description, notional_usd, direction, category_sensitivities)
            VALUES (?, ?, ?, ?, ?)
        """, (
            pos.ticker, pos.description, pos.notional_usd, pos.direction,
            json.dumps(pos.category_sensitivities),
        ))
        return cur.lastrowid


def remove_position(pos_id: int, db_path: Optional[str] = None):
    db_path = db_path or _default_db_path()
    with _conn(db_path) as c:
        c.execute("DELETE FROM positions WHERE id=?", (pos_id,))


def list_positions(db_path: Optional[str] = None) -> List[Position]:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM positions ORDER BY id DESC").fetchall()
    out: List[Position] = []
    for r in rows:
        try:
            sens = json.loads(r["category_sensitivities"])
        except Exception:
            sens = {}
        out.append(Position(
            id=r["id"], ticker=r["ticker"], description=r["description"] or "",
            notional_usd=r["notional_usd"], direction=r["direction"],
            category_sensitivities=sens,
        ))
    return out


def exposure_by_category(
    category_key: str,
    db_path: Optional[str] = None,
) -> Tuple[float, List[Position]]:
    """Return (signed_exposure_to_category, contributing_positions).

    Positive = net long the category risk (will lose if risk fires).
    Negative = net short / hedged (will gain if risk fires).
    """
    positions = list_positions(db_path=db_path)
    contributing: List[Position] = []
    exposure = 0.0
    for p in positions:
        sens = p.category_sensitivities.get(category_key, 0.0)
        if abs(sens) < 1e-6:
            continue
        exposure += p.signed_notional() * sens
        contributing.append(p)
    return exposure, contributing


# Generic hedge suggestions by category (these are *directional ideas*, not
# specific recommendations and not investment advice).
HEDGE_IDEAS: Dict[str, List[str]] = {
    "geopolitical": [
        "Long crude oil exposure (calls or futures) — geopolitical disruption typically spikes oil",
        "Long gold — classic geopolitical hedge",
        "Long defense-sector ETFs",
        "Long VIX or VIX call spreads — volatility expansion",
        "Short EM currencies most exposed to the conflict region",
    ],
    "macro_financial": [
        "Long long-duration Treasuries — flight to quality in recession",
        "Long USD index — risk-off",
        "Long gold",
        "Short cyclicals, long defensives",
        "Buy put spreads on broad equity indices",
    ],
    "market_specific": [
        "Buy put options on the specific name or sector",
        "Short sector ETF if available",
        "Reduce position size as primary hedge",
    ],
    "epidemic": [
        "Long pharma / vaccine names",
        "Long staples (toilet paper economy)",
        "Short travel, leisure, restaurant exposure",
        "Long VIX",
    ],
    "natural_hazard": [
        "Long catastrophe-bond ETFs (inverse) — but illiquid",
        "Short property-heavy insurers in the affected region",
        "Long commodities affected by the hazard (e.g., natural gas if cold snap)",
    ],
    "cyber_tech": [
        "Long cybersecurity ETFs (HACK, CIBR)",
        "Short the specific firm if you have conviction on the breach impact",
        "Long platform alternatives if the trigger is platform-collapse",
    ],
    "operational_corporate": [
        "Reduce single-name exposure",
        "Buy puts on the affected name",
        "Long competitors who would gain market share",
    ],
    "political_regulatory": [
        "Long sectors that benefit from the regulatory direction",
        "Short most-exposed names",
        "Buy puts dated past the regulatory event",
    ],
}


def hedge_ideas(category_key: str) -> List[str]:
    return HEDGE_IDEAS.get(category_key, ["Reduce exposure; consider broad index put spreads."])


def estimate_loss_given_event(
    exposure_usd: float,
    expected_drawdown_pct: float,
    p99_drawdown_pct: float,
) -> Dict[str, float]:
    """Convert exposure + drawdown estimates into dollar loss bands.

    Expected drawdown is typically ~5-15% for moderate events, 20-40% for severe.
    These are approximations; real loss depends on which specific instruments
    you hold, the implied vol environment, and exit liquidity.
    """
    if exposure_usd <= 0:
        return {"expected_loss": 0.0, "tail_loss": 0.0, "exposure": exposure_usd}
    expected = exposure_usd * (expected_drawdown_pct / 100.0)
    tail = exposure_usd * (p99_drawdown_pct / 100.0)
    return {
        "expected_loss": float(expected),
        "tail_loss": float(tail),
        "exposure": float(exposure_usd),
    }
