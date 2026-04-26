#!/usr/bin/env bash
# Poll v6 + v7 HF jobs, refresh the data files, commit + push.
# No docs, no readmes, just data.
set -euo pipefail

V6=69ed9454d70108f37acdf848
V7=69edb1bdd70108f37acdfbb1

cd "$(git rev-parse --show-toplevel)"
DEST=outputs/v6_vs_v7_comparison

echo "[1/4] Pulling v6 logs"
hf jobs logs "$V6" > "$DEST/v6_raw.log" 2>&1 || echo "(v6 fetch failed; keeping previous snapshot)"

echo "[2/4] Pulling v7 logs"
hf jobs logs "$V7" > "$DEST/v7_raw.log" 2>&1 || echo "(v7 fetch failed; keeping previous snapshot)"

echo "[3/4] Re-running parser"
python "$DEST/parse_logs.py" "$DEST/v6_raw.log" "$DEST/v7_raw.log"

echo "[4/4] Commit + push (data only)"
TS=$(date -u +"%Y-%m-%dT%H:%MZ")
git add "$DEST/" 2>/dev/null || true
if git diff --cached --quiet; then
    echo "No data changes since last poll. Done."
    exit 0
fi
git commit -m "poll: v6/v7 snapshot at $TS" >/dev/null
git push 2>&1 | tail -3
echo "Pushed."
