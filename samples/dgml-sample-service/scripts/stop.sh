#!/usr/bin/env bash
# Stop the DGML sample service.
#
#   scripts/stop.sh               # stop frontend, backend, Postgres and SeaweedFS (data kept)
#   scripts/stop.sh --apps-only   # stop frontend + backend, leave the containers running
#   scripts/stop.sh --purge       # stop everything AND delete the Postgres/SeaweedFS volumes
#   scripts/stop.sh --colima      # also stop the colima VM afterwards
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

APPS_ONLY=0 PURGE=0 COLIMA=0
for arg in "$@"; do
  case "$arg" in
    --apps-only) APPS_ONLY=1 ;;
    --purge) PURGE=1 ;;
    --colima) COLIMA=1 ;;
    -h | --help)
      sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

stop_component() {
  local name="$1" port="$2" pid pids
  if pid="$(live_pid "$name")"; then
    kill "$pid" 2>/dev/null || true
    for _ in {1..40}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    ok "$name stopped (pid $pid)"
  else
    ok "$name not running"
  fi
  rm -f "$RUN_DIR/$name.pid"
  # A reloader child or a process started by hand may still hold the port.
  pids="$(port_pids "$port")"
  if [[ -n "$pids" ]]; then
    warn "port $port still held by pid(s) $(echo "$pids" | tr '\n' ' ')— stopping them"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
  fi
}

say "Apps"
stop_component frontend "$WEB_PORT"
stop_component backend "$API_PORT"

((APPS_ONLY)) && exit 0

say "Postgres + SeaweedFS"
if ! command -v docker >/dev/null || ! docker_up; then
  ok "Docker is not running — nothing to stop"
elif ((PURGE)); then
  compose down -v
  ok "containers removed and data volumes deleted"
else
  compose stop
  ok "containers stopped (data kept; --purge deletes it)"
fi

if ((COLIMA)) && command -v colima >/dev/null; then
  say "colima"
  colima stop
fi
