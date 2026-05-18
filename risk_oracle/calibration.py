"""
Calibration store — SQLite-backed log of every prediction with resolution
tracking and Brier scoring. The output feeds back into model weights via
get_model_weights().
"""
from __future__ import annotations
import sqlite3
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple


def _default_db_path() -> str:
    home = Path.home()
    folder = home / ".risk_oracle"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "calibration.db")


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
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                trigger TEXT NOT NULL,
                category TEXT NOT NULL,
                primary_p REAL NOT NULL,
                critic_p REAL NOT NULL,
                reconciled_p REAL NOT NULL,
                band_low REAL NOT NULL,
                band_high REAL NOT NULL,
                expected_resolution TEXT,
                resolved INTEGER NOT NULL DEFAULT 0,
                resolution_outcome INTEGER,
                resolution_date TEXT,
                brier_primary REAL,
                brier_critic REAL,
                brier_reconciled REAL,
                metadata TEXT
            )
        """)
    return db_path


@dataclass
class Prediction:
    trigger: str
    category: str
    primary_p: float
    critic_p: float
    reconciled_p: float
    band_low: float
    band_high: float
    expected_resolution: Optional[str] = None
    metadata: Optional[Dict] = None


def log_prediction(p: Prediction, db_path: Optional[str] = None) -> int:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        cur = c.execute("""
            INSERT INTO predictions
            (created_at, trigger, category, primary_p, critic_p, reconciled_p,
             band_low, band_high, expected_resolution, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            p.trigger, p.category,
            p.primary_p, p.critic_p, p.reconciled_p,
            p.band_low, p.band_high,
            p.expected_resolution,
            json.dumps(p.metadata or {}),
        ))
        return cur.lastrowid


def resolve_prediction(pred_id: int, outcome: bool, db_path: Optional[str] = None):
    """Mark a prediction resolved. outcome=True if event occurred, False if not.

    Brier score: (forecast - actual)^2, lower is better. Range 0-1.
    """
    db_path = db_path or _default_db_path()
    o = 1 if outcome else 0
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM predictions WHERE id=?", (pred_id,)).fetchone()
        if not row:
            return
        b_p = (row["primary_p"] - o) ** 2
        b_c = (row["critic_p"] - o) ** 2
        b_r = (row["reconciled_p"] - o) ** 2
        c.execute("""
            UPDATE predictions
            SET resolved=1, resolution_outcome=?, resolution_date=?,
                brier_primary=?, brier_critic=?, brier_reconciled=?
            WHERE id=?
        """, (o, datetime.utcnow().isoformat(), b_p, b_c, b_r, pred_id))


def list_predictions(db_path: Optional[str] = None,
                     limit: int = 200) -> List[Dict]:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_weights(category: str,
                      db_path: Optional[str] = None,
                      lookback: int = 50,
                      min_resolved: int = 5) -> Tuple[float, float]:
    """Compute calibration-weighted importance of primary vs critic for this category.

    Returns (primary_weight, critic_weight) in [0, 1] each, not normalised.
    Uses rolling mean Brier score from last `lookback` resolved predictions in
    the category. Lower Brier = higher weight.

    If fewer than `min_resolved` resolved predictions exist, returns (0.5, 0.5).
    """
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute("""
            SELECT brier_primary, brier_critic
            FROM predictions
            WHERE category=? AND resolved=1
            ORDER BY id DESC
            LIMIT ?
        """, (category, lookback)).fetchall()
    if len(rows) < min_resolved:
        return (0.5, 0.5)
    avg_b_p = sum(r["brier_primary"] for r in rows) / len(rows)
    avg_b_c = sum(r["brier_critic"] for r in rows) / len(rows)
    # Lower Brier is better. Weight = (1 - Brier).
    w_p = max(0.05, 1.0 - avg_b_p)
    w_c = max(0.05, 1.0 - avg_b_c)
    return (w_p, w_c)


def category_brier_stats(db_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    db_path = db_path or _default_db_path()
    init_db(db_path)
    with _conn(db_path) as c:
        rows = c.execute("""
            SELECT category,
                   AVG(brier_primary) AS b_p,
                   AVG(brier_critic) AS b_c,
                   AVG(brier_reconciled) AS b_r,
                   COUNT(*) AS n
            FROM predictions
            WHERE resolved=1
            GROUP BY category
        """).fetchall()
    return {
        r["category"]: {
            "n": int(r["n"]),
            "brier_primary": r["b_p"] or 0.0,
            "brier_critic": r["b_c"] or 0.0,
            "brier_reconciled": r["b_r"] or 0.0,
        }
        for r in rows
    }
