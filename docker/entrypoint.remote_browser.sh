#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export XVFB_WHD="${XVFB_WHD:-1920x1080x24}"
export REMOTE_BROWSER_HOST="${REMOTE_BROWSER_HOST:-0.0.0.0}"
export REMOTE_BROWSER_PORT="${REMOTE_BROWSER_PORT:-8060}"

if [ -z "${REMOTE_BROWSER_API_KEY:-}" ]; then
  echo "REMOTE_BROWSER_API_KEY is required" >&2
  exit 1
fi

rm -f /tmp/.X99-lock
Xvfb "${DISPLAY}" -screen 0 "${XVFB_WHD}" -ac +extension RANDR >/tmp/xvfb-remote-browser.log 2>&1 &
XVFB_PID=$!

cleanup() {
  if ps -p "${XVFB_PID}" >/dev/null 2>&1; then
    kill "${XVFB_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

exec python -m uvicorn src.remote_browser_service.app:app \
  --host "${REMOTE_BROWSER_HOST}" \
  --port "${REMOTE_BROWSER_PORT}"

