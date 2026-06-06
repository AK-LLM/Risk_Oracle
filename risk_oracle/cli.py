"""
cli.py — Headless command-line entry points for Risk Oracle.

Used by the OS-native schedulers (cron / launchd / Task Scheduler) so the
watchlist refreshes and dispatches alerts without a human opening Streamlit.

Subcommands:
  refresh-watchlist     Re-run every watchlist item; persist updated probabilities;
                        fire dispatch alerts for items that moved past threshold.
  status                Print bridge + dispatch + watchlist counts.

Usage:
  python -m risk_oracle.cli refresh-watchlist
  python -m risk_oracle.cli status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List


def _load_secrets() -> Dict[str, str]:
    """Pull secrets from env. Streamlit secrets aren't accessible headlessly."""
    keys = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "FRED_API_KEY", "ACLED_API_KEY", "ACLED_EMAIL",
        "GMAIL_USER", "GMAIL_APP_PASSWORD", "GMAIL_TO",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    ]
    return {k: os.environ.get(k, "") for k in keys if os.environ.get(k)}


def cmd_refresh_watchlist(args: argparse.Namespace) -> int:
    """Re-run watchlist items; fire alerts on movement."""
    from dataclasses import asdict
    from . import watchlist as wl
    from . import dispatch as dp
    from .pipeline import run_forecast
    from .models import EvidenceFactor

    secrets = _load_secrets()
    items = wl.list_items()
    if not items:
        print("(watchlist is empty)")
        return 0

    refreshed_dicts: List[Dict[str, Any]] = []
    failures = []
    for item in items:
        try:
            evidence = [
                EvidenceFactor(
                    name=e.get("name", ""),
                    likelihood_ratio=float(e.get("likelihood_ratio", 1.0)),
                    confidence=float(e.get("confidence", 1.0)),
                )
                for e in (item.evidence or [])
                if isinstance(e, dict)
            ]
            fr = run_forecast(
                trigger=item.trigger,
                category=item.category,
                prior=float(item.prior),
                evidence=evidence,
                secrets=secrets,
                include_comparison=True,
            )
            wl.record_refresh(
                item_id=item.id,
                probability=fr.point_p,
                band_low=fr.band_low,
                band_high=fr.band_high,
                market_prob=fr.market_prob,
            )
            # Re-read to get the updated row (with previous_probability set)
            updated_items = wl.list_items()
            updated = next((u for u in updated_items if u.id == item.id), None)
            if updated is not None:
                refreshed_dicts.append(asdict(updated))
        except Exception as exc:
            failures.append({"id": item.id, "error": str(exc)})

    if args.verbose:
        print(f"Refreshed {len(refreshed_dicts)} items, {len(failures)} failures.")
        for f in failures:
            print(f"  ! item {f['id']}: {f['error']}")

    # Dispatch on movement
    dispatched = dp.dispatch_movement_alerts(refreshed_dicts)
    sent = sum(1 for d in dispatched if d.get("email_sent") is True or d.get("telegram_sent") is True)
    skipped = sum(1 for d in dispatched if "skipped" in d)
    if args.verbose or sent > 0:
        print(f"Dispatched {sent} alerts; {skipped} items had no significant movement.")
        for d in dispatched:
            if "skipped" not in d:
                print(f"  watchlist_id={d['watchlist_id']} movement={d.get('movement', 0):.3f}")

    return 0 if not failures else 1


def cmd_status(args: argparse.Namespace) -> int:
    from . import watchlist as wl
    from . import dispatch as dp
    items = wl.list_items()
    dstat = dp.dispatch_status()
    print(json.dumps({
        "watchlist_count": len(items),
        "watchlist_categories": sorted({i.category for i in items}),
        "dispatch": dstat,
        "now": datetime.utcnow().isoformat(),
    }, indent=2))
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="risk_oracle.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refresh = sub.add_parser("refresh-watchlist", help="Re-run watchlist items and dispatch movement alerts")
    p_refresh.add_argument("--verbose", action="store_true", help="Show per-item refresh + dispatch detail")
    p_refresh.set_defaults(func=cmd_refresh_watchlist)

    p_status = sub.add_parser("status", help="Show watchlist + dispatch status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
