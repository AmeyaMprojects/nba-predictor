#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/com.predictor.daily.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/data/logs"

# Use Python for safe string replacement with proper XML escaping
# Handles special characters (&, |, <, >, etc) that break sed or are invalid in XML
python3 << 'PYTHON_SCRIPT'
import sys

# XML-escape the path: & -> &amp;, < -> &lt;, > -> &gt;
def escape_xml(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s

PROJECT_DIR = """$PROJECT_DIR"""
ESCAPED_PATH = escape_xml(PROJECT_DIR)

with open("""$PROJECT_DIR""" + "/scripts/com.predictor.daily.plist", 'r') as f:
    plist_content = f.read()

# Replace PROJECT_DIR placeholder with properly escaped path
output = plist_content.replace("PROJECT_DIR", ESCAPED_PATH)

with open("""$TARGET""", 'w') as f:
    f.write(output)
PYTHON_SCRIPT

launchctl unload "$TARGET" 2>/dev/null || true
launchctl load "$TARGET"

# Verify the job actually registered (launchctl load can exit 0 without loading)
if launchctl list | grep -q "com.predictor.daily"; then
    echo "Successfully installed and loaded: $TARGET"
    echo "The news archiver will run at 09:00, 14:00, and 19:00 daily."
else
    echo "ERROR: The scheduled job failed to load."
    echo "Check system logs with: log stream --predicate 'eventMessage contains[c] predictor' --level debug"
    exit 1
fi
