#!/usr/bin/env bash
# Show what is running for the DGML sample service.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

row() { printf '  %-10s %-9s %s\n' "$1" "$2" "$3"; }
up() { printf '%sup%s' "$_G" "$_N"; }
down() { printf '%sdown%s' "$_D" "$_N"; }

say "DGML sample service"
if command -v docker >/dev/null && docker_up; then
  for svc in postgres seaweedfs; do
    state="$(compose ps --format '{{.Service}} {{.State}} {{.Health}}' 2>/dev/null | awk -v s="$svc" '$1==s {print $2" "$3}')"
    if [[ "$state" == running* ]]; then
      detail="localhost:$DGML_SAMPLE_PG_PORT"
      [[ "$svc" == seaweedfs ]] && detail="$DGML_SAMPLE_S3_ENDPOINT  filer http://localhost:$DGML_SAMPLE_FILER_PORT"
      row "$svc" "$(up)" "$detail  (${state#running })"
    else
      row "$svc" "$(down)" ""
    fi
  done
else
  row docker "$(down)" "daemon not reachable"
fi

if pid="$(live_pid backend)"; then
  health="$(curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null && echo healthy || echo 'not answering')"
  row backend "$(up)" "http://127.0.0.1:$API_PORT  pid $pid, $health"
else
  row backend "$(down)" ""
fi
if pid="$(live_pid frontend)"; then
  row frontend "$(up)" "http://localhost:$WEB_PORT  pid $pid"
else
  row frontend "$(down)" ""
fi
