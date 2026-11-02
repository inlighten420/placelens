#!/bin/sh
# PlaceLens Studio on macOS or Linux: `sh start.sh`. The first time it sets itself up (a few minutes), then opens
# your browser.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/placelens ]; then
  echo "Setting up PlaceLens (first time only, a few minutes)..."
  if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "PlaceLens needs Python 3.10 or newer: https://www.python.org/downloads/"
    exit 1
  fi
  python3 -m venv .venv && .venv/bin/pip install --quiet --upgrade pip && .venv/bin/pip install --quiet -e . || {
    echo "Setting up didn't work (see above). Check your internet connection and try again."
    rm -rf .venv
    exit 1
  }
fi
echo "Starting PlaceLens Studio. Ctrl+C to stop."
exec .venv/bin/placelens studio
