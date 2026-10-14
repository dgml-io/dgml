#!/usr/bin/env bash
# Start the DGML sample service: Postgres + SeaweedFS (Docker), the FastAPI backend,
# and the React frontend (Vite dev server). Idempotent: anything already running is
# left alone.
#
#   scripts/start.sh               # everything, frontend with hot reload on :5180
#   scripts/start.sh --build       # build the frontend and serve it from the API (:8700)
#   scripts/start.sh --reload      # also restart the backend on Python changes
#   scripts/start.sh --infra-only  # just Postgres + SeaweedFS (run the apps yourself)
#   scripts/start.sh --no-web      # infrastructure + backend, no frontend
#
# Ports (override with env): DGML_SAMPLE_API_PORT=8700, DGML_SAMPLE_WEB_PORT=5180,
# DGML_SAMPLE_PG_PORT=55432, DGML_SAMPLE_S3_PORT=18333, DGML_SAMPLE_FILER_PORT=18888.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INFRA_ONLY=0 WEB=1 BUILD=0 RELOAD=0
for arg in "$@"; do
  case "$arg" in
    --infra-only) INFRA_ONLY=1 ;;
    --no-web) WEB=0 ;;
    --build) BUILD=1 ;;
    --reload) RELOAD=1 ;;
    -h | --help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

mkdir -p "$RUN_DIR"

# ---- Docker ----------------------------------------------------------------
say "Docker"
command -v docker >/dev/null || die "docker is not installed (brew install colima docker docker-compose, or Docker Desktop)"
if ! docker_up; then
  if command -v colima >/dev/null; then
    ok "starting colima (first start downloads a VM image and can take a few minutes)"
    colima start >/dev/null
  else
    die "the Docker daemon is not running — start Docker Desktop / OrbStack / colima"
  fi
fi
docker_up || die "the Docker daemon is still not reachable"
ok "daemon running"

# ---- Postgres + SeaweedFS ----------------------------------------------------
say "Postgres + SeaweedFS"
compose up -d --wait
ok "postgres   localhost:$DGML_SAMPLE_PG_PORT"
ok "seaweedfs  $DGML_SAMPLE_S3_ENDPOINT  (filer UI http://localhost:$DGML_SAMPLE_FILER_PORT)"

# ---- Python environment + default bucket -------------------------------------
say "Python environment"
if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c "import dgml_sample_service" 2>/dev/null; then
  ok "syncing the uv workspace"
  (cd "$REPO_ROOT" && uv sync --all-extras --dev >/dev/null)
fi
ok "$("$PYTHON" --version)"

"$PYTHON" - "$DGML_SAMPLE_S3_ENDPOINT" "$DGML_SAMPLE_S3_BUCKET" <<'PY'
import sys, boto3
from botocore.exceptions import ClientError
endpoint, bucket = sys.argv[1], sys.argv[2]
s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id="dgml",
                  aws_secret_access_key="dgml-secret", region_name="us-east-1")
try:
    s3.head_bucket(Bucket=bucket)
except ClientError:
    s3.create_bucket(Bucket=bucket)
PY
ok "bucket s3://$DGML_SAMPLE_S3_BUCKET"

if ((INFRA_ONLY)); then
  say "Infrastructure is up. Database URL: $DGML_SAMPLE_DATABASE_URL"
  exit 0
fi

# ---- Frontend build (--build) ------------------------------------------------
ensure_node_modules() {
  command -v npm >/dev/null || die "node/npm is not installed (brew install node)"
  if [[ ! -d "$FRONTEND_DIR/node_modules" || "$FRONTEND_DIR/package.json" -nt "$FRONTEND_DIR/node_modules" ]]; then
    ok "installing frontend dependencies"
    (cd "$FRONTEND_DIR" && npm install --no-audit --no-fund >/dev/null)
    touch "$FRONTEND_DIR/node_modules"
  fi
}

if ((BUILD && WEB)); then
  say "Frontend build"
  ensure_node_modules
  (cd "$FRONTEND_DIR" && npm run build >"$RUN_DIR/frontend-build.log" 2>&1) ||
    die "frontend build failed — see $RUN_DIR/frontend-build.log"
  export DGML_SAMPLE_FRONTEND_DIST="$FRONTEND_DIR/dist"
  ok "built into frontend/dist"
fi

# ---- Backend -----------------------------------------------------------------
say "Backend (FastAPI)"
if pid="$(live_pid backend)"; then
  ok "already running (pid $pid) — scripts/stop.sh first to restart it"
elif [[ -n "$(port_pids "$API_PORT")" ]]; then
  die "port $API_PORT is in use by another process (set DGML_SAMPLE_API_PORT)"
else
  args=(-m dgml_sample_service --port "$API_PORT")
  ((RELOAD)) && args+=(--reload)
  # `exec` so the recorded PID is the server itself, not a wrapper subshell.
  (cd "$SERVICE_DIR" && exec "$PYTHON" "${args[@]}") </dev/null >"$RUN_DIR/backend.log" 2>&1 &
  echo $! >"$RUN_DIR/backend.pid"
  if wait_http "http://127.0.0.1:$API_PORT/api/health" 60; then
    ok "http://127.0.0.1:$API_PORT  (API docs: /docs, log: .run/backend.log)"
  else
    tail -20 "$RUN_DIR/backend.log" >&2 || true
    die "the backend did not come up — see $RUN_DIR/backend.log"
  fi
fi

# ---- Frontend dev server -----------------------------------------------------
if ((WEB && !BUILD)); then
  say "Frontend (Vite)"
  if pid="$(live_pid frontend)"; then
    ok "already running (pid $pid)"
  elif [[ -n "$(port_pids "$WEB_PORT")" ]]; then
    die "port $WEB_PORT is in use by another process (set DGML_SAMPLE_WEB_PORT)"
  else
    ensure_node_modules
    (cd "$FRONTEND_DIR" && DGML_SAMPLE_API_PORT="$API_PORT" exec ./node_modules/.bin/vite \
      --port "$WEB_PORT" --strictPort) </dev/null >"$RUN_DIR/frontend.log" 2>&1 &
    echo $! >"$RUN_DIR/frontend.pid"
    if wait_http "http://localhost:$WEB_PORT/" 60; then
      ok "http://localhost:$WEB_PORT  (log: .run/frontend.log)"
    else
      tail -20 "$RUN_DIR/frontend.log" >&2 || true
      die "the frontend did not come up — see $RUN_DIR/frontend.log"
    fi
  fi
fi

echo
if ((WEB)); then
  url="http://localhost:$WEB_PORT"
  ((BUILD)) && url="http://127.0.0.1:$API_PORT"
  say "Ready: ${_B}$url${_N}   (scripts/status.sh · scripts/logs.sh · scripts/stop.sh)"
else
  say "Ready: API on http://127.0.0.1:$API_PORT   (scripts/stop.sh to stop)"
fi
