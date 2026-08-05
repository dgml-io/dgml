#!/usr/bin/env bash
# Fail if an agent skill file and its mirror have drifted apart.
#
# `.claude/skills/<name>/SKILL.md` is the canonical copy; `.gemini/skills/<name>/`
# holds a byte-identical mirror. The two agents use the same SKILL.md frontmatter
# format, so there is no structural reason for two files — but a CLI change that
# updates one and forgets the other leaves the second agent teaching a stale
# interface, silently and indefinitely.
#
# A symlink would make drift structurally impossible and is tempting, but this
# project supports Windows at runtime (`gswin64c`, Windows Credential Manager,
# cross-drive source paths). A Windows checkout without `core.symlinks=true`
# materializes a symlink as a *regular file containing the target path* — so the
# Gemini agent would read `../../.claude/skills/dgml/SKILL.md` as its entire
# instruction set. Two real files that a check keeps in sync degrade worse-case to
# "someone has to run --fix"; symlinks degrade to "the skill is silently empty".
#
# Usage:
#   scripts/check-agent-skills.sh          # report drift, exit 1 if any
#   scripts/check-agent-skills.sh --fix    # copy canonical over each mirror
set -euo pipefail

cd "$(dirname "$0")/.."

fix=0
for arg in "$@"; do
  case "$arg" in
    --fix) fix=1 ;;
    -h | --help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "check-agent-skills.sh: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

# Discovered by glob rather than listed, so adding a skill under `.claude/skills/`
# puts it under this check automatically instead of on someone's memory.
shopt -s nullglob
canonicals=(.claude/skills/*/SKILL.md)
if [[ ${#canonicals[@]} -eq 0 ]]; then
  echo "check-agent-skills.sh: no .claude/skills/*/SKILL.md found — run from the repo root" >&2
  exit 2
fi

drifted=0
for canonical in "${canonicals[@]}"; do
  mirror=".gemini/${canonical#.claude/}"
  if [[ ! -f $mirror ]]; then
    # A missing mirror is drift too: the skill exists for one agent only.
    if [[ $fix -eq 1 ]]; then
      mkdir -p "$(dirname "$mirror")"
      cp "$canonical" "$mirror"
      echo "created $mirror"
    else
      echo "MISSING  $mirror (canonical: $canonical)"
      drifted=1
    fi
    continue
  fi
  if cmp -s "$canonical" "$mirror"; then
    continue
  fi
  if [[ $fix -eq 1 ]]; then
    cp "$canonical" "$mirror"
    echo "synced   $mirror"
  else
    echo "DRIFTED  $mirror vs $canonical"
    diff -u "$mirror" "$canonical" || true
    drifted=1
  fi
done

if [[ $drifted -eq 1 ]]; then
  cat >&2 <<'EOF'

Agent skill mirrors are out of sync. `.claude/skills/<name>/SKILL.md` is canonical:
edit that copy, then run

  scripts/check-agent-skills.sh --fix

to bring `.gemini/` back in line, and commit both files together.
EOF
  exit 1
fi

echo "agent skill mirrors are in sync (${#canonicals[@]} skill(s))"
