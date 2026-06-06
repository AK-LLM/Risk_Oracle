#!/usr/bin/env bash
# uninstall_linux.sh — remove Risk Oracle's systemd-user timer or crontab block.
set -euo pipefail

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TIMER="$UNIT_DIR/risk-oracle-refresh.timer"
SERVICE="$UNIT_DIR/risk-oracle-refresh.service"

if [[ -f "$TIMER" ]]; then
  systemctl --user stop  risk-oracle-refresh.timer   2>/dev/null || true
  systemctl --user disable risk-oracle-refresh.timer 2>/dev/null || true
  rm -f "$TIMER" "$SERVICE"
  systemctl --user daemon-reload 2>/dev/null || true
  echo "✓ Removed systemd-user timer"
fi

MARK_START="# >>> risk-oracle (managed) >>>"
MARK_END="# <<< risk-oracle (managed) <<<"
current="$(crontab -l 2>/dev/null || true)"
if printf '%s' "$current" | grep -q "$MARK_START"; then
  stripped="$(printf '%s\n' "$current" | awk -v s="$MARK_START" -v e="$MARK_END" '
    $0==s {skip=1; next}
    $0==e {skip=0; next}
    !skip {print}
  ')"
  printf '%s\n' "$stripped" | crontab -
  echo "✓ Removed crontab block"
fi
echo "Risk Oracle scripts + state are untouched at ~/.risk_oracle/"
