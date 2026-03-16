#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export ALLOW_DOCKER_HEADED_CAPTCHA="${ALLOW_DOCKER_HEADED_CAPTCHA:-true}"
export XVFB_WHD="${XVFB_WHD:-1920x1080x24}"
export FLOW2API_AUTO_SET_DISPLAY="${FLOW2API_AUTO_SET_DISPLAY:-true}"
export FLOW2API_AUTO_START_XVFB="${FLOW2API_AUTO_START_XVFB:-true}"

if ! command -v Xvfb >/dev/null 2>&1; then
  echo "[start.linux.headed] Xvfb not found. Please install xvfb first." >&2
  exit 1
fi

if ! command -v fluxbox >/dev/null 2>&1; then
  echo "[start.linux.headed] fluxbox not found. Please install fluxbox first." >&2
  exit 1
fi

LOCK_FILE="/tmp/.X${DISPLAY#:}-lock"
if [ -f "${LOCK_FILE}" ]; then
  rm -f "${LOCK_FILE}" || true
fi

if ! pgrep -f "Xvfb ${DISPLAY}" >/dev/null 2>&1; then
  echo "[start.linux.headed] starting Xvfb on ${DISPLAY} (${XVFB_WHD})"
  Xvfb "${DISPLAY}" -screen 0 "${XVFB_WHD}" -ac -nolisten tcp +extension RANDR >/tmp/xvfb-flow2api.log 2>&1 &
  XVFB_PID=$!
else
  echo "[start.linux.headed] Xvfb already running on ${DISPLAY}"
  XVFB_PID=""
fi

if ! pgrep -x fluxbox >/dev/null 2>&1; then
  echo "[start.linux.headed] starting Fluxbox"
  fluxbox >/tmp/fluxbox-flow2api.log 2>&1 &
  FLUXBOX_PID=$!
else
  echo "[start.linux.headed] Fluxbox already running"
  FLUXBOX_PID=""
fi

cleanup() {
  if [ -n "${FLUXBOX_PID:-}" ] && ps -p "${FLUXBOX_PID}" >/dev/null 2>&1; then
    kill "${FLUXBOX_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${XVFB_PID:-}" ] && ps -p "${XVFB_PID}" >/dev/null 2>&1; then
    kill "${XVFB_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sleep 1

echo "[start.linux.headed] starting uvicorn src.main:app --host 0.0.0.0 --port 8000"
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
