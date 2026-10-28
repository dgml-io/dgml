#!/usr/bin/env bash
# Follow a component's log.
#
#   scripts/logs.sh              # backend
#   scripts/logs.sh frontend
#   scripts/logs.sh postgres     # or seaweedfs (container logs)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

case "${1:-backend}" in
  backend | frontend)
    file="$RUN_DIR/${1:-backend}.log"
    [[ -f "$file" ]] || die "no log yet at $file — run scripts/start.sh"
    exec tail -n 200 -f "$file"
    ;;
  postgres | seaweedfs) exec docker compose -f "$COMPOSE_FILE" logs -f --tail 200 "$1" ;;
  *) die "unknown component '$1' (backend | frontend | postgres | seaweedfs)" ;;
esac
