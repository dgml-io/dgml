#!/usr/bin/env bash
# Measure test coverage for a single module (default: the code-based
# grounding matcher, dgml.matching) and print the lines still uncovered.
#
# The code-based grounder is the part of grounding we can test fully
# offline — no LLM, no network — so it's where line coverage is a
# meaningful, enforceable signal. The real-document regression tests in
# packages/dgml/tests/test_grounding_real_docs.py exist to drive this
# number up and hold it there.
#
# Usage:
#   scripts/coverage.sh                       # cover dgml.matching
#   scripts/coverage.sh dgml.grounded         # cover a different module
#   scripts/coverage.sh dgml.matching --fail-under=94
set -euo pipefail

cd "$(dirname "$0")/.."

module="${1:-dgml.matching}"
shift || true

run() {
  printf '\n→ %s\n' "$*"
  "$@"
}

run uv run pytest \
  --cov="${module}" \
  --cov-report=term-missing \
  packages/dgml/tests "$@"
