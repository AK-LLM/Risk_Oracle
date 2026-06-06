# Risk Oracle V2.2 — Lifecycle, Velocity, Regime, Noise

This release ports STP's algorithmic disciplines into Risk Oracle so both
suites share the same epistemological patterns. Nothing in V2.1 was removed;
everything is additive.

## What's New

### `lifecycle.py` — SCOUT / STALKING / STRIKING / LATE

Every watchlist refresh now computes a stage from refresh history:

- **SCOUT** — fewer than 3 refreshes; signal is fresh
- **STALKING** — ≥3 refreshes, evidence accumulating, band still wide
- **STRIKING** — band has compressed materially; consensus forming
- **LATE** — probability at extreme (≥0.90 or ≤0.10), or >20 refreshes, or band very narrow

Two helpers:
- `stage_priority(stage)` — for dispatch ordering
- `stage_sizing_multiplier(stage)` — Kelly fraction modifier (SCOUT 0.5 / STALKING 0.8 / STRIKING 1.0 / LATE 0.4)

### `velocity.py` — probability movement acceleration

Tracks whether the probability is **ACCELERATING_UP**, **ACCELERATING_DOWN**, **RISING**, **FALLING**, **STABLE**, or **DECELERATING**. Used by dispatch to bump alert urgency when acceleration changes.

### `regime.py` — VIX-driven regime classification

Pulls latest VIX from FRED (or Yahoo as fallback) and classifies:
- **panic** (VIX ≥ 30) — widens disagreement threshold; lowers alert threshold
- **elevated** (20 ≤ VIX < 30)
- **normal** (13 < VIX < 20)
- **complacent** (VIX ≤ 13) — tail risks may be underpriced

### Noise-aware OSINT (`osint.py`)

`OSINTSignal` now has a `noise_level` field. Every fetcher gets tagged (low / medium / high) via the `SOURCE_NOISE` map. New `OSINTBundle.weighted_concordance()` weights signals by inverse noise: regulatory + on-chain (low) get full weight, social + early prediction markets (high) get one-third.

### Schema additions in `watchlist.py`

The watchlist table gains four columns via idempotent ALTER TABLE statements (existing v2.1 databases migrate automatically):
- `stage` — current lifecycle stage
- `velocity_acceleration` — current velocity state
- `velocity_recent_delta` — most-recent probability delta (signed)
- `regime_at_refresh` — regime label at the time of the refresh

### CLI integration (`cli.py`)

`refresh-watchlist` now:
1. Detects regime once per batch (single FRED call, not per-item)
2. Computes lifecycle stage from refresh history before each refresh
3. Computes velocity from the synthetic-with-new-refresh history
4. Records all three in `watchlist`
5. Applies regime-aware alert thresholds (panic regime → 3pp threshold instead of 5pp)

### Dispatch body upgrades (`dispatch.py`)

Movement-alert emails/Telegram now include stage, velocity, and regime in the body so you see context, not just the new probability.

## Files Changed

**New**
- `risk_oracle/lifecycle.py`
- `risk_oracle/velocity.py`
- `risk_oracle/regime.py`
- `UPGRADE_NOTES_V2.2.md`

**Modified**
- `risk_oracle/osint.py` — `OSINTSignal.noise_level`, `OSINTBundle.weighted_concordance()`, `SOURCE_NOISE` map, gather_osint tags signals
- `risk_oracle/watchlist.py` — schema additions + extended record_refresh signature
- `risk_oracle/cli.py` — refresh-watchlist computes and persists stage/velocity/regime
- `risk_oracle/dispatch.py` — alert body shows stage/velocity/regime

## What Was Not Changed

- Probabilistic pipeline (`models.py`, `reconcile.py`, `pipeline.py`) — untouched
- Primary + Critic adversarial layer — already in place from v1; no change needed
- Calibration loop (`calibration.py`) — untouched
- Decision layer (`decision.py`) — untouched
- Streamlit UI (`app.py`) — untouched (new fields are visible if you query them, but no new tabs)

## Migration

Drop-in compatible. The watchlist schema migration is idempotent (try/except on ALTER TABLE). On first run, items that haven't been refreshed yet will show `stage = None`; this resolves on the next refresh.

## Cross-suite note

STP V6.1 (built in parallel with this release) ports Risk Oracle's primary+critic + probability-band + calibration-writeback disciplines into the trading suite. The two releases are designed to ship together — together they close the algorithmic-parity gap.
