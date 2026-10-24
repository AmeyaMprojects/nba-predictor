#!/usr/bin/env bash
# Cut-over (Task 14): the cloud run now does the live work, so unload and
# remove the Mac's four live jobs. The news archiver and the weekly history
# refresh stay installed (reinstall them alone with
# bash scripts/install_schedule.sh --after-cutover).
# Rollback: bash scripts/install_schedule.sh
set -euo pipefail

LIVE_JOBS=(
    com.predictor.predict
    com.predictor.results
    com.predictor.odds
    com.predictor.schedule
)

for label in "${LIVE_JOBS[@]}"; do
    target="$HOME/Library/LaunchAgents/$label.plist"
    if [ -f "$target" ]; then
        launchctl unload "$target" 2>/dev/null || true
        rm -f "$target"
        echo "Unloaded and removed: $target"
    else
        echo "Not installed: $label"
    fi
done
