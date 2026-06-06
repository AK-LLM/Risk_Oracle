#!/usr/bin/env bash
# schedule_mac.sh — install hourly watchlist refresh as a launchd agent.
# Calls `python -m risk_oracle.cli refresh-watchlist` once per hour. Each run
# re-forecasts every watchlist item and dispatches email/Telegram on movement.
#
# Idempotent: re-running unloads + reloads cleanly.
# Logs land in ~/Library/Logs/risk-oracle/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
if [[ -z "$PY" ]]; then
  echo "python3 not found on PATH. Install Python 3.10+ first." >&2
  exit 1
fi

LABEL="ventures.local.risk-oracle.watchlist-refresh"
LA_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/risk-oracle"
PLIST="$LA_DIR/${LABEL}.plist"

mkdir -p "$LA_DIR" "$LOG_DIR"

# Hourly refresh (every 3600s). Edit StartInterval if you want a different cadence.
cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PY}</string>
    <string>-m</string>
    <string>risk_oracle.cli</string>
    <string>refresh-watchlist</string>
  </array>
  <key>WorkingDirectory</key><string>${PROJECT_ROOT}</string>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>${LOG_DIR}/refresh.out.log</string>
  <key>StandardErrorPath</key><string>${LOG_DIR}/refresh.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "✓ Risk Oracle watchlist refresh installed at ${LABEL}"
echo "  Cadence: every hour"
echo "  Logs:    ${LOG_DIR}/"
echo "  Notes:   set ANTHROPIC_API_KEY (and GMAIL_USER/GMAIL_APP_PASSWORD if you want email"
echo "           dispatch) in your shell environment — launchd inherits them on next load."
echo "  Uninstall with: bash ${PROJECT_ROOT}/install/uninstall_mac.sh"
