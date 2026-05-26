"""
Bet tracker — log Polymarket bets, resolve them on outcome, track P&L
and edge realization.

Bets are stored locally in SQLite at ~/.risk_oracle/bets.db.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


def _default_db_path() -> str:
    folder = Path.home() / ".risk_oracle"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "bets.db")


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
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                placed_at TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_question TEXT NOT NULL,
                market_url TEXT,
                side TEXT NOT NULL CHECK (side IN ('YES','NO')),
                entry_price REAL NOT NULL,
                size_usd REAL NOT NULL,
                our_probability REAL NOT NULL,
                edge_at_entry REAL NOT NULL,
                expected_value_usd REAL,
                forecast_id INTEGER,
                resolved INTEGER NOT NULL DEFAULT 0,
                outcome TEXT,
                closed_at TEXT,
                pnl_usd REAL,
                notes TEXT
            )
        """)
    return db_path


@dataclass
class Bet:
    placed_at: str
    market_id: str
    market_question: str
    side: str
    entry_price: float
    size_usd: float
    our_probability: float
    edge_at_entry: float
    expected_value_usd: float = 0.0
    market_url: str = ""
    forecast_id: Optional[int] = None
    notes: str = ""
    id: Optional[int] = None
    resolved: bool = False
    outcome: Optional[str] = None
    closed_at: Optional[str] = None
    pnl_usd: Optional[float] = None


def log_bet(bet: Bet, db_path: Optional[str] = None) -> int:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        cur = c.execute("""
            INSERT INTO bets
            (placed_at, market_id, market_question, market_url, side,
             entry_price, size_usd, our_probability, edge_at_entry,
             expected_value_usd, forecast_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bet.placed_at or datetime.utcnow().isoformat(),
            bet.market_id, bet.market_question, bet.market_url, bet.side,
            bet.entry_price, bet.size_usd, bet.our_probability,
            bet.edge_at_entry, bet.expected_value_usd,
            bet.forecast_id, bet.notes,
        ))
        return cur.lastrowid


def resolve_bet(bet_id: int, outcome: str, db_path: Optional[str] = None):
    """Mark bet resolved and compute P&L.

    Polymarket binary mechanics: each share costs entry_price, pays $1 if the
    bet outcome occurred, $0 otherwise. So P&L per share = (1 if win else 0) - entry_price.
    Number of shares = size_usd / entry_price. Net P&L = size_usd * (win/entry_price - 1).
    """
    db_path = db_path or _default_db_path()
    outcome = outcome.upper()
    if outcome not in ("YES", "NO"):
        raise ValueError("outcome must be 'YES' or 'NO'")
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()
        if not row:
            return
        won = (row["side"] == outcome)
        shares = row["size_usd"] / row["entry_price"] if row["entry_price"] > 0 else 0
        pnl = (shares * 1.0 - row["size_usd"]) if won else (-row["size_usd"])
        c.execute("""
            UPDATE bets
            SET resolved=1, outcome=?, closed_at=?, pnl_usd=?
            WHERE id=?
        """, (outcome, datetime.utcnow().isoformat(), pnl, bet_id))


def remove_bet(bet_id: int, db_path: Optional[str] = None):
    db_path = db_path or _default_db_path()
    with _conn(db_path) as c:
        c.execute("DELETE FROM bets WHERE id=?", (bet_id,))


def list_bets(db_path: Optional[str] = None,
              only_open: bool = False,
              limit: int = 500) -> List[Dict]:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    q = "SELECT * FROM bets"
    if only_open:
        q += " WHERE resolved=0"
    q += " ORDER BY id DESC LIMIT ?"
    with _conn(db_path) as c:
        rows = c.execute(q, (limit,)).fetchall()
    return [dict(r) for r in rows]


def summary(db_path: Optional[str] = None) -> Dict:
    """Aggregate stats: total bets, win rate, ROI, edge realization."""
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute("""
            SELECT side, entry_price, size_usd, our_probability,
                   edge_at_entry, resolved, outcome, pnl_usd
            FROM bets
        """).fetchall()
    total = len(rows)
    open_bets = [r for r in rows if r["resolved"] == 0]
    closed = [r for r in rows if r["resolved"] == 1]
    open_size = sum(r["size_usd"] for r in open_bets)
    closed_size = sum(r["size_usd"] for r in closed)
    wins = [r for r in closed if r["side"] == r["outcome"]]
    losses = [r for r in closed if r["side"] != r["outcome"]]
    realized_pnl = sum(r["pnl_usd"] or 0 for r in closed)
    win_rate = len(wins) / len(closed) if closed else None
    roi = realized_pnl / closed_size if closed_size > 0 else None

    # Expected win rate weighted by our_probability at entry, by side
    expected_wins = 0.0
    for r in closed:
        p_our = r["our_probability"]
        if r["side"] == "YES":
            expected_wins += p_our
        else:
            expected_wins += (1 - p_our)
    expected_win_rate = expected_wins / len(closed) if closed else None

    return {
        "total_bets": total,
        "open_bets": len(open_bets),
        "closed_bets": len(closed),
        "open_size_usd": open_size,
        "closed_size_usd": closed_size,
        "realized_pnl_usd": realized_pnl,
        "win_rate": win_rate,
        "expected_win_rate": expected_win_rate,
        "calibration_gap": (win_rate - expected_win_rate)
            if (win_rate is not None and expected_win_rate is not None) else None,
        "roi": roi,
    }
