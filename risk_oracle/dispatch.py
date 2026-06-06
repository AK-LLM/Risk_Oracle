"""
dispatch.py — V2.1 email + Telegram delivery for Risk Oracle.

Fires when a watchlist forecast moves more than `alert_threshold` since the
last refresh. The threshold is per-watchlist-item (default 0.05 = 5 percentage
points), set when the item was created.

Idempotent: each (item, refresh_ts) pair is dispatched at most once. State is
kept in a small SQLite alongside the watchlist DB.

Required env (in Streamlit secrets OR shell env):
  GMAIL_USER             — sending Gmail address
  GMAIL_APP_PASSWORD     — 16-char app password from myaccount.google.com/apppasswords
  GMAIL_TO               — recipient (defaults to GMAIL_USER)
  TELEGRAM_BOT_TOKEN     — optional
  TELEGRAM_CHAT_ID       — optional

Headless usage (cron-driven): call dispatch_movement_alerts() after each
watchlist refresh. The CLI runner in `cli.py` does this automatically.
"""
from __future__ import annotations

import os
import smtplib
import sqlite3
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _dispatch_db_path() -> str:
    folder = Path.home() / ".risk_oracle"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "dispatch.db")


@contextmanager
def _conn(db_path: Optional[str] = None):
    c = sqlite3.connect(db_path or _dispatch_db_path())
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _init_db(db_path: Optional[str] = None) -> None:
    with _conn(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS dispatched (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watchlist_id INTEGER NOT NULL,
                refresh_ts TEXT NOT NULL,
                channel TEXT NOT NULL,
                ok INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(watchlist_id, refresh_ts, channel)
            )
        """)


def _already_dispatched(watchlist_id: int, refresh_ts: str, channel: str) -> bool:
    _init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM dispatched WHERE watchlist_id=? AND refresh_ts=? AND channel=? AND ok=1",
            (watchlist_id, refresh_ts, channel),
        ).fetchone()
    return row is not None


def _record_dispatch(watchlist_id: int, refresh_ts: str, channel: str, ok: bool, note: str) -> None:
    _init_db()
    with _conn() as c:
        try:
            c.execute(
                "INSERT INTO dispatched (watchlist_id, refresh_ts, channel, ok, note) VALUES (?,?,?,?,?)",
                (watchlist_id, refresh_ts, channel, 1 if ok else 0, note[:200]),
            )
        except sqlite3.IntegrityError:
            # Already recorded — fine
            pass


def _secrets() -> Dict[str, str]:
    """Read env + Streamlit secrets (if available) in that priority order."""
    out: Dict[str, str] = {}
    keys = ["GMAIL_USER", "GMAIL_APP_PASSWORD", "GMAIL_TO",
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            out[k] = v
    try:
        import streamlit as st  # type: ignore
        for k in keys:
            if k not in out and k in st.secrets:
                out[k] = str(st.secrets[k])
    except Exception:
        pass
    return out


def _build_subject(item: Dict[str, Any]) -> str:
    prev = item.get("previous_probability")
    curr = item.get("last_probability")
    direction = ""
    if prev is not None and curr is not None:
        delta = curr - prev
        direction = " ↑" if delta > 0 else " ↓"
    return f"[Risk Oracle]{direction} {item.get('category', '?')}: {item.get('trigger', '')[:60]}"


def _build_body(item: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Trigger: {item.get('trigger', '')}")
    lines.append(f"Category: {item.get('category', '?')}")
    lines.append("")
    curr = item.get("last_probability") or 0
    prev = item.get("previous_probability")
    bl = item.get("last_band_low") or 0
    bh = item.get("last_band_high") or 1
    lines.append(f"Current probability: {curr:.1%}")
    if prev is not None:
        lines.append(f"Previous probability: {prev:.1%}")
        lines.append(f"Movement: {(curr - prev) * 100:+.1f} percentage points")
    lines.append(f"Band: [{bl:.1%}, {bh:.1%}] (width {(bh - bl) * 100:.1f}pp)")
    last = item.get("last_refreshed_at")
    if last:
        lines.append("")
        lines.append(f"Refreshed: {last}")
    market_prob = item.get("last_market_prob")
    if market_prob is not None:
        lines.append(f"Market (Polymarket/Manifold/Metaculus consensus): {market_prob:.1%}")
        edge = curr - market_prob
        lines.append(f"Edge vs market: {edge * 100:+.1f}pp")
    lines.append("")
    lines.append("─" * 60)
    lines.append("This is a movement alert, not a trade recommendation.")
    return "\n".join(lines)


def _send_email(secrets: Dict[str, str], subject: str, body: str) -> Tuple[bool, str]:
    user = secrets.get("GMAIL_USER", "").strip()
    pw = secrets.get("GMAIL_APP_PASSWORD", "").strip().replace(" ", "")
    to = secrets.get("GMAIL_TO", "").strip() or user
    if not user or not pw:
        return False, "GMAIL_USER / GMAIL_APP_PASSWORD not configured"
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        msg.set_content(body)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
            s.login(user, pw)
            s.send_message(msg)
        return True, "ok"
    except Exception as exc:
        return False, f"email error: {exc}"


def _send_telegram(secrets: Dict[str, str], subject: str, body: str) -> Tuple[bool, str]:
    token = secrets.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = secrets.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return False, "telegram not configured"
    try:
        text = f"*{subject}*\n\n```\n{body}\n```"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat, "text": text, "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return (200 <= resp.status < 300, f"http {resp.status}")
    except Exception as exc:
        return False, f"telegram error: {exc}"


def dispatch_movement_alert(item: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a single watchlist item if its movement is alert-worthy and
    we haven't already dispatched this refresh cycle."""
    threshold = float(item.get("alert_threshold", 0.05) or 0.05)
    curr = item.get("last_probability")
    prev = item.get("previous_probability")
    wid = int(item.get("id", 0))
    refresh_ts = str(item.get("last_refreshed_at") or "")

    if curr is None or prev is None or wid == 0:
        return {"watchlist_id": wid, "skipped": "missing fields"}

    movement = abs(curr - prev)
    if movement < threshold:
        return {"watchlist_id": wid, "skipped": f"movement {movement:.3f} < threshold {threshold:.3f}"}

    subject = _build_subject(item)
    body = _build_body(item)
    secrets = _secrets()

    result = {"watchlist_id": wid, "movement": movement}

    if not _already_dispatched(wid, refresh_ts, "email"):
        ok, note = _send_email(secrets, subject, body)
        result["email_sent"] = ok
        result["email_note"] = note
        _record_dispatch(wid, refresh_ts, "email", ok, note)
    else:
        result["email_sent"] = "already_dispatched"

    if not _already_dispatched(wid, refresh_ts, "telegram"):
        ok, note = _send_telegram(secrets, subject, body)
        result["telegram_sent"] = ok
        result["telegram_note"] = note
        _record_dispatch(wid, refresh_ts, "telegram", ok, note)
    else:
        result["telegram_sent"] = "already_dispatched"

    return result


def dispatch_movement_alerts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run dispatch over every item in a watchlist refresh batch."""
    return [dispatch_movement_alert(item) for item in items]


def dispatch_status() -> Dict[str, Any]:
    s = _secrets()
    return {
        "email_configured": bool(s.get("GMAIL_USER") and s.get("GMAIL_APP_PASSWORD")),
        "telegram_configured": bool(s.get("TELEGRAM_BOT_TOKEN") and s.get("TELEGRAM_CHAT_ID")),
    }
