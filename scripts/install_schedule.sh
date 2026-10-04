#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/data/logs"

install_job() {
    local label="$1"
    local source="$PROJECT_DIR/scripts/$label.plist"
    local target="$HOME/Library/LaunchAgents/$label.plist"

    # Use Python for safe string replacement with proper XML escaping.
    # Handles special characters (&, |, <, >, etc) that break sed or are
    # invalid in XML. Paths are passed as argv, not interpolated into the
    # Python source text: a path containing a quote, backslash, or
    # triple-quote sequence could otherwise corrupt or inject into the
    # program.
    python3 - "$PROJECT_DIR" "$source" "$target" << 'PYTHON_SCRIPT'
import sys

def escape_xml(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s

project_dir, source, target = sys.argv[1], sys.argv[2], sys.argv[3]

with open(source, "r") as f:
    plist_content = f.read()

with open(target, "w") as f:
    f.write(plist_content.replace("PROJECT_DIR", escape_xml(project_dir)))
PYTHON_SCRIPT

    launchctl unload "$target" 2>/dev/null || true
    launchctl load "$target"

    # Verify the job actually registered (launchctl load can exit 0 without loading)
    if launchctl list | grep -q "$label"; then
        echo "Successfully installed and loaded: $target"
    else
        echo "ERROR: the scheduled job $label failed to load."
        echo "Check system logs with: log stream --predicate 'eventMessage contains[c] predictor' --level debug"
        exit 1
    fi
}

install_job com.predictor.daily
install_job com.predictor.schedule
install_job com.predictor.results
install_job com.predictor.predict

echo "The news archiver will run at 09:00, 14:00, and 19:00 daily."
echo "The schedule archiver will run at 10:30 daily."
echo "Live results will be captured at 11:00 daily."
echo "Today's predictions will be published at 18:00 daily."
