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
run uv run mypy packages

if [[ $fast -eq 0 ]]; then
  run uv run pytest
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
