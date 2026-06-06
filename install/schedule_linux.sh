#!/usr/bin/env bash
# schedule_linux.sh — register hourly watchlist refresh via systemd-user
# timer (preferred) or crontab (fallback).
#
# Logs land in ~/.local/state/risk-oracle/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
if [[ -z "$PY" ]]; then
  echo "python3 not found on PATH. Install Python 3.10+ first." >&2
  exit 1
fi

LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/risk-oracle"
mkdir -p "$LOG_DIR"

if command -v systemctl >/dev/null 2>&1 && systemctl --user >/dev/null 2>&1; then
  # systemd-user path
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$UNIT_DIR"

  cat >"$UNIT_DIR/risk-oracle-refresh.service" <<EOF
[Unit]
Description=Risk Oracle watchlist refresh
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${PY} -m risk_oracle.cli refresh-watchlist
StandardOutput=append:${LOG_DIR}/refresh.out.log
StandardError=append:${LOG_DIR}/refresh.err.log
EOF

  cat >"$UNIT_DIR/risk-oracle-refresh.timer" <<EOF
[Unit]
Description=Hourly Risk Oracle watchlist refresh

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable risk-oracle-refresh.timer
  systemctl --user start risk-oracle-refresh.timer

  echo "✓ Risk Oracle watchlist refresh installed as systemd-user timer (hourly)"
  echo "  Status: systemctl --user list-timers | grep risk-oracle"
  echo "  Logs:   ${LOG_DIR}/"
  echo "  (To survive logout: 'loginctl enable-linger \$USER')"

else
  # crontab fallback
  MARK_START="# >>> risk-oracle (managed) >>>"
  MARK_END="# <<< risk-oracle (managed) <<<"
  current="$(crontab -l 2>/dev/null || true)"
  stripped="$(printf '%s\n' "$current" | awk -v s="$MARK_START" -v e="$MARK_END" '
    $0==s {skip=1; next}
    $0==e {skip=0; next}
    !skip {print}
  ')"
  block="$(cat <<EOF
${MARK_START}
0 * * * * cd ${PROJECT_ROOT} && ${PY} -m risk_oracle.cli refresh-watchlist >> ${LOG_DIR}/refresh.cron.log 2>&1
${MARK_END}
EOF
)"
  printf '%s\n\n%s\n' "$stripped" "$block" | crontab -
  echo "✓ Risk Oracle watchlist refresh scheduled via crontab (hourly)"
  echo "  Inspect: crontab -l"
  echo "  Logs:    ${LOG_DIR}/"
fi
