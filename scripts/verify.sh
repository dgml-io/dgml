#!/usr/bin/env bash
# Run the same checks CI does (.github/workflows/ci.yml), locally, in one
# shot. Stops on the first failure. Run before `git push` to avoid the
# round-trip of pushing, watching CI fail, fixing, and pushing again.
#
# Usage:
#   scripts/verify.sh              # run everything CI runs
#   scripts/verify.sh --no-sync    # skip `uv sync` (faster repeats, but
#                                  # only safe if you haven't touched
#                                  # pyproject.toml or uv.lock)
#   scripts/verify.sh --fast       # skip pytest and license-audit
#                                  # (fast feedback loop on lint/format/types)
set -euo pipefail

cd "$(dirname "$0")/.."

do_sync=1
fast=0
for arg in "$@"; do
  case "$arg" in
    --no-sync) do_sync=0 ;;
    --fast) fast=1 ;;
    -h | --help)
      sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "verify.sh: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

run() {
  printf '\n→ %s\n' "$*"
  "$@"
}

if [[ $do_sync -eq 1 ]]; then
  # Same install line CI uses — keeps the local venv in lockstep with
  # what CI exercises (--all-extras --dev --locked).
  run uv sync --all-extras --dev --locked
fi

run uv run ruff check .
run uv run ruff format --check .
# Cheap enough to run unconditionally, including under --fast: the two agent
# skill mirrors are hand-kept copies, and a stale one teaches a CLI surface that
# no longer exists. `--fix` on the same script resyncs them.
run scripts/check-agent-skills.sh
run uv run mypy packages

if [[ $fast -eq 0 ]]; then
  run uv run pytest
  # The sample-store suite runs above against in-process fakes. If the local
  # MinIO/Mongo stack happens to be up, re-run it against the real services —
  # that is what CI's storage-parity job does, and where wire behaviour the
  # fakes only approximate (pagination, real error codes) actually gets tested.
  if curl -sf http://localhost:9000/minio/health/live >/dev/null 2>&1; then
    # boto3 refuses to sign anonymously, so the suite needs credentials even
    # though MinIO is local. Default to the docker-compose root user; a real
    # AWS_* already in the environment wins, so this never overrides a
    # developer pointed at a different endpoint.
    DGML_TEST_S3_ENDPOINT=http://localhost:9000 \
    DGML_TEST_MONGO_URI=mongodb://localhost:27017 \
    AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-dgmltest}" \
    AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-dgmltest123}" \
    AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
      run uv run pytest packages/dgml-storage-s3
  else
    printf '\n· skipping real-service storage parity (MinIO not reachable);\n'
    printf '  start it with: docker compose -f packages/dgml-storage-s3/docker-compose.yml up -d\n'
  fi
  # Same deny-list as the license-audit CI job; strong copyleft deps
  # must not land in a runtime dependency of an Apache-2.0-licensed wheel.
  # `--partial-match` is required so deny tokens match real license
  # strings (without it pip-licenses uses exact match and silently
  # passes everything). MPL is intentionally absent — see CLAUDE.md
  # "License compatibility" for the policy.
  run uv run pip-licenses --from=mixed --partial-match \
    --fail-on='GPL;LGPL;AGPL;SSPL;EUPL;CC-BY-SA'
fi

echo
echo "OK — all CI gates passed locally."
