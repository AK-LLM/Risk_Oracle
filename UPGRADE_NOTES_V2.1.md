# Risk Oracle V2.1 — Headless + Dispatch

This release makes Risk Oracle usable as a background service: the watchlist
can now refresh itself on a schedule and email/Telegram you when a forecast
moves materially. Nothing in the core probabilistic pipeline changed.

## What's New

### `dispatch.py` — email + Telegram on watchlist movement

When the watchlist refresh produces a probability movement larger than the
item's `alert_threshold` (default 5pp), an email goes out — plus a Telegram
message if you've configured a bot. Idempotent: each (item, refresh_ts) pair
is dispatched at most once, tracked in a small SQLite at
`~/.risk_oracle/dispatch.db`.

Configured via secrets (`.streamlit/secrets.toml`) or environment variables:
- `GMAIL_USER` + `GMAIL_APP_PASSWORD` (app password, not your real Gmail pw)
- `GMAIL_TO` — optional, defaults to `GMAIL_USER`
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — optional

### `cli.py` — headless command-line runner

Two subcommands:
```bash
python -m risk_oracle.cli refresh-watchlist [--verbose]
python -m risk_oracle.cli status
```

`refresh-watchlist` re-runs every watchlist item through `run_forecast`,
records the new probabilities, and calls `dispatch_movement_alerts` on items
that moved past their threshold. This is what the OS schedulers call.

`status` prints a JSON summary of watchlist counts and whether dispatch is
configured.

### OS-native schedulers (`install/`)

Three pairs of scripts that register the hourly watchlist refresh as a
background service:
- `schedule_mac.sh` / `uninstall_mac.sh` — launchd
- `schedule_linux.sh` / `uninstall_linux.sh` — systemd-user timer preferred,
  crontab fallback
- `schedule_windows.ps1` / `uninstall_windows.ps1` — Task Scheduler

Default cadence: every hour. Edit the unit/plist/task to change.

### Two new OSINT sources (`osint.py` + `taxonomy.py`)

- **`fed_speeches`** — parses federalreserve.gov speeches RSS, returns a
  hawkish/dovish tilt in [-1, +1]. Wired into the `macro_financial` category's
  `osint_signals` list, so every macro forecast now gets this signal in its
  OSINT bundle. No API key required.
- **`politician_trades`** — placeholder hook for Senate/House Periodic
  Transaction Reports (PTRs). Currently a stub returning an "unconfigured"
  signal; the orchestrator includes it in `political_regulatory` queries so
  you can plug in capitoltrades.com or housestockwatcher.com later without
  touching the taxonomy.

## Files Changed

**New**
- `risk_oracle/dispatch.py`
- `risk_oracle/cli.py`
- `install/schedule_{mac,linux,windows}.{sh,ps1}` and matching uninstallers
- `UPGRADE_NOTES_V2.1.md`

**Modified**
- `risk_oracle/osint.py` — adds `fetch_fed_speeches`, `fetch_politician_trades_volume`,
  registers both in `SIGNAL_FETCHERS`
- `risk_oracle/taxonomy.py` — adds `fed_speeches` to `macro_financial.osint_signals`
  and `politician_trades` to `political_regulatory.osint_signals`
- `.streamlit/secrets.toml.example` — adds dispatch keys

## What Was Not Changed

- Probabilistic pipeline (`models.py`, `reconcile.py`, `pipeline.py`) — untouched
- Calibration loop (`calibration.py`) — untouched
- Decision layer (`decision.py`, Kelly sizing, Polymarket recommendation) — untouched
- Portfolio context (`portfolio.py`) — untouched
- Watchlist schema (`watchlist.py`) — untouched
- Streamlit app (`app.py`) — untouched

This means everything you've been doing in the UI still works exactly the same.
The new pieces are additive and live next to the existing modules.

## Headless usage

Once the OS scheduler is installed and your secrets are populated:

1. Add a few items to the watchlist via the Streamlit UI (give each item an
   `alert_threshold` — 0.05 is a reasonable default).
2. The scheduler triggers `refresh-watchlist` hourly. Each run forecasts every
   item, records the new probability + band, and dispatches alerts for items
   that moved past their threshold since the previous refresh.
3. Watch the logs (`~/Library/Logs/risk-oracle/` on Mac;
   `~/.local/state/risk-oracle/` on Linux; `%LOCALAPPDATA%\RiskOracle\Logs\`
   on Windows). The CLI prints one line per dispatched alert.

## Streamlit Community Cloud note

The OS schedulers are for local installs. On Streamlit Community Cloud, the
app process restarts on its own schedule and there's no cron-equivalent. If
you need headless refresh in the cloud, the cleanest path is either:
- Use Streamlit's own [scheduled rerun](https://docs.streamlit.io/) feature
  (calling `cli.cmd_refresh_watchlist` from a button or background thread), or
- Run the headless CLI on a small VPS / always-on machine pointing at the same
  `EXTERNAL_DB_URL`.

The dispatch module reads from env first, then Streamlit secrets — so it
works in both contexts.

## Migration

Drop-in compatible. No schema changes to existing databases (`calibration.db`,
`watchlist.db`, `portfolio.db`, `bets.db`). The new `dispatch.db` is created
on first use.
