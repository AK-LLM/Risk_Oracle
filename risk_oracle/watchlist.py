"""
Watchlist — persistent set of ongoing triggers that the user wants to track
over time. Each item stores the trigger plus its evidence so it can be re-run
on demand. The system records each refresh and flags items whose reconciled
probability has moved more than `alert_threshold` since the last check.

Streamlit Community Cloud caveat: there's no real background task runner.
A "Refresh all" button in the UI re-runs every watchlist item sequentially.
For continuous polling, run the app locally and use a cron job that invokes
`refresh_all_due()` periodically.
"""
from __future__ import annotations
import sqlite3
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


def _default_db_path() -> str:
    folder = Path.home() / ".risk_oracle"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "watchlist.db")


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
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger TEXT NOT NULL,
                category TEXT NOT NULL,
                prior REAL NOT NULL,
                evidence_json TEXT NOT NULL,
                alert_threshold REAL NOT NULL DEFAULT 0.05,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_refreshed_at TEXT,
                last_probability REAL,
                last_band_low REAL,
                last_band_high REAL,
                previous_probability REAL,
                last_market_prob REAL,
                notes TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watchlist_id INTEGER NOT NULL,
                refreshed_at TEXT NOT NULL,
                probability REAL NOT NULL,
                band_low REAL NOT NULL,
                band_high REAL NOT NULL,
                market_prob REAL
            )
        """)
        # V2.2: lifecycle + velocity columns on the main row. Use ALTER TABLE
        # with try/except so existing v2.1 databases migrate transparently.
        for ddl in [
            "ALTER TABLE watchlist ADD COLUMN stage TEXT",
            "ALTER TABLE watchlist ADD COLUMN velocity_acceleration TEXT",
            "ALTER TABLE watchlist ADD COLUMN velocity_recent_delta REAL",
            "ALTER TABLE watchlist ADD COLUMN regime_at_refresh TEXT",
        ]:
            try:
                c.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already exists
    return db_path


@dataclass
class WatchlistItem:
    trigger: str
    category: str
    prior: float
    evidence: List[Dict]
    alert_threshold: float = 0.05
    notes: str = ""
    id: Optional[int] = None
    last_refreshed_at: Optional[str] = None
    last_probability: Optional[float] = None
    last_band_low: Optional[float] = None
    last_band_high: Optional[float] = None
    previous_probability: Optional[float] = None
    last_market_prob: Optional[float] = None

    def movement(self) -> Optional[float]:
        if self.last_probability is None or self.previous_probability is None:
            return None
        return self.last_probability - self.previous_probability

    def has_alert(self) -> bool:
        m = self.movement()
        return m is not None and abs(m) >= self.alert_threshold


def add_item(item: WatchlistItem, db_path: Optional[str] = None) -> int:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        cur = c.execute("""
            INSERT INTO watchlist
            (trigger, category, prior, evidence_json, alert_threshold, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            item.trigger, item.category, item.prior,
            json.dumps(item.evidence),
            item.alert_threshold, item.notes,
        ))
        return cur.lastrowid


def remove_item(item_id: int, db_path: Optional[str] = None):
    db_path = db_path or _default_db_path()
    with _conn(db_path) as c:
        c.execute("DELETE FROM watchlist WHERE id=?", (item_id,))
        c.execute("DELETE FROM watchlist_history WHERE watchlist_id=?", (item_id,))


def list_items(db_path: Optional[str] = None) -> List[WatchlistItem]:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM watchlist ORDER BY id DESC").fetchall()
    out: List[WatchlistItem] = []
    for r in rows:
        try:
            ev = json.loads(r["evidence_json"])
        except Exception:
            ev = []
        out.append(WatchlistItem(
            id=r["id"], trigger=r["trigger"], category=r["category"],
            prior=r["prior"], evidence=ev, alert_threshold=r["alert_threshold"],
            notes=r["notes"] or "",
            last_refreshed_at=r["last_refreshed_at"],
            last_probability=r["last_probability"],
            last_band_low=r["last_band_low"],
            last_band_high=r["last_band_high"],
            previous_probability=r["previous_probability"],
            last_market_prob=r["last_market_prob"],
        ))
    return out


def record_refresh(
    item_id: int,
    probability: float,
    band_low: float,
    band_high: float,
    market_prob: Optional[float] = None,
    db_path: Optional[str] = None,
    stage: Optional[str] = None,
    velocity_acceleration: Optional[str] = None,
    velocity_recent_delta: Optional[float] = None,
    regime_at_refresh: Optional[str] = None,
):
    db_path = db_path or _default_db_path()
    init_db(db_path)
    now = datetime.utcnow().isoformat()
    with _conn(db_path) as c:
        # Pull previous to move into previous_probability slot
        row = c.execute("SELECT last_probability FROM watchlist WHERE id=?",
                        (item_id,)).fetchone()
        prev = row["last_probability"] if row and row["last_probability"] is not None else None
        c.execute("""
            UPDATE watchlist
            SET previous_probability = ?,
                last_probability = ?,
                last_band_low = ?,
                last_band_high = ?,
                last_market_prob = ?,
                last_refreshed_at = ?,
                stage = COALESCE(?, stage),
                velocity_acceleration = COALESCE(?, velocity_acceleration),
                velocity_recent_delta = COALESCE(?, velocity_recent_delta),
                regime_at_refresh = COALESCE(?, regime_at_refresh)
            WHERE id = ?
        """, (prev, probability, band_low, band_high, market_prob, now,
              stage, velocity_acceleration, velocity_recent_delta, regime_at_refresh,
              item_id))
        c.execute("""
            INSERT INTO watchlist_history
            (watchlist_id, refreshed_at, probability, band_low, band_high, market_prob)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item_id, now, probability, band_low, band_high, market_prob))


def history(item_id: int, db_path: Optional[str] = None) -> List[Dict]:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute("""
            SELECT * FROM watchlist_history
            WHERE watchlist_id = ?
            ORDER BY id ASC
        """, (item_id,)).fetchall()
    return [dict(r) for r in rows]
