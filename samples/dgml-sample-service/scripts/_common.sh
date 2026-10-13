# Shared settings and helpers for the sample service scripts. Sourced, not run.
# shellcheck shell=bash

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SERVICE_DIR/../.." && pwd)"
RUN_DIR="$SERVICE_DIR/.run"
FRONTEND_DIR="$SERVICE_DIR/frontend"
COMPOSE_FILE="$SERVICE_DIR/docker-compose.yml"

# Homebrew and ~/.local/bin (uv) may not be on a non-login shell's PATH.
for d in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin"; do
  [[ -d "$d" && ":$PATH:" != *":$d:"* ]] && PATH="$d:$PATH"
done
export PATH

export DGML_SAMPLE_PG_PORT="${DGML_SAMPLE_PG_PORT:-55432}"
export DGML_SAMPLE_S3_PORT="${DGML_SAMPLE_S3_PORT:-18333}"
export DGML_SAMPLE_FILER_PORT="${DGML_SAMPLE_FILER_PORT:-18888}"
API_PORT="${DGML_SAMPLE_API_PORT:-8700}"
WEB_PORT="${DGML_SAMPLE_WEB_PORT:-5180}"

export DGML_SAMPLE_DATABASE_URL="${DGML_SAMPLE_DATABASE_URL:-postgresql+pg8000://dgml:dgml@localhost:${DGML_SAMPLE_PG_PORT}/dgml_sample}"
export DGML_SAMPLE_S3_ENDPOINT="${DGML_SAMPLE_S3_ENDPOINT:-http://localhost:${DGML_SAMPLE_S3_PORT}}"
export DGML_SAMPLE_S3_BUCKET="${DGML_SAMPLE_S3_BUCKET:-dgml-sample}"
export DGML_SAMPLE_CORS_ORIGINS="${DGML_SAMPLE_CORS_ORIGINS:-http://localhost:${WEB_PORT},http://127.0.0.1:${WEB_PORT}}"

PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ -t 1 ]]; then
  _B=$'\033[1m' _G=$'\033[32m' _Y=$'\033[33m' _R=$'\033[31m' _D=$'\033[2m' _N=$'\033[0m'
else
  _B="" _G="" _Y="" _R="" _D="" _N=""
fi

say() { printf '%s==>%s %s\n' "$_B" "$_N" "$*"; }
ok() { printf '  %s✓%s %s\n' "$_G" "$_N" "$*"; }
warn() { printf '  %s!%s %s\n' "$_Y" "$_N" "$*" >&2; }
die() {
  printf '%serror:%s %s\n' "$_R" "$_N" "$*" >&2
  exit 1
}

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

# The PID recorded for a component, if that process is still alive.
live_pid() {
  local file="$RUN_DIR/$1.pid" pid
  [[ -f "$file" ]] || return 1
  pid="$(cat "$file")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

# Whatever is listening on a TCP port (to catch processes started some other way).
port_pids() { lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true; }

docker_up() { docker info >/dev/null 2>&1; }

# Wait until a URL answers, or give up after $2 seconds.
wait_http() {
  local url="$1" secs="${2:-30}" i
  for ((i = 0; i < secs * 4; i++)); do
    curl -sf -o /dev/null "$url" && return 0
    sleep 0.25
  done
  return 1
}
