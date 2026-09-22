#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.predictor.daily.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/data/logs"
sed "s|PROJECT_DIR|$PROJECT_DIR|g" \
    "$PROJECT_DIR/scripts/com.predictor.daily.plist" > "$TARGET"

launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"

echo "Installed and loaded: $TARGET"
echo "Verify with: launchctl list | grep predictor"
