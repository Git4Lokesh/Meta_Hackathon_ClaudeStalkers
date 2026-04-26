#!/usr/bin/env bash
# Poll v7 HF job every 5 minutes: refresh v7_raw.log, re-parse with existing v6_raw.log, commit+push.
# v6 log is left unchanged (v6 run finished) — only v7 updates each cycle.
# Stop with: pkill -f poll_v7_loop.sh
set -euo pipefail

V7=69edb1bdd70108f37acdfbb1
INTERVAL_SEC="${POLL_INTERVAL_SEC:-300}"
cd "$(git rev-parse --show-toplevel)"
DEST=outputs/v6_vs_v7_comparison
LOGERR="$DEST/poll_v7_stderr.log"

poll_once() {
  local ts
  ts=$(date -u +"%Y-%m-%dT%H:%MZ")
  echo "[$ts] Polling v7 $V7 ..." | tee -a "$LOGERR"
  if ! hf jobs logs "$V7" > "$DEST/v7_raw.log" 2>>"$LOGERR"; then
    echo "[$ts] hf jobs logs failed (rate limit?); keeping previous v7_raw.log" | tee -a "$LOGERR"
    return 0
  fi
  if python "$DEST/parse_logs.py" "$DEST/v6_raw.log" "$DEST/v7_raw.log" 2>>"$LOGERR"; then
    :
  else
    echo "[$ts] parse_logs.py failed" | tee -a "$LOGERR"
  fi
  if git add "$DEST/" 2>/dev/null && ! git diff --cached --quiet; then
    git commit -m "poll: v7 snapshot at $ts" >/dev/null
    git push 2>>"$LOGERR" | tail -2
    echo "[$ts] Pushed." | tee -a "$LOGERR"
  else
    echo "[$ts] No v7 data changes; skip commit." | tee -a "$LOGERR"
  fi
}

echo "Starting v7-only poll every ${INTERVAL_SEC}s. Job $V7. Log: $LOGERR"
while true; do
  poll_once || true
  sleep "$INTERVAL_SEC"
done
