#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  changelog_section.sh <version> [changelog_path]

Prints the CHANGELOG.md section body for <version> (without the
"## X.Y.Z ..." heading line itself), stopping at the next "## " heading
or end of file. <version> is matched as the semver core X.Y.Z that
follows "## " — an optional "v" prefix and any " - <date>" suffix on the
heading are ignored.

Exits 0 with the section body on stdout when found. Exits 0 with empty
output when the version has no section (callers decide whether that is
fatal). Exits non-zero only on usage / missing-file errors.
EOF
}

version="${1:-}"
changelog="${2:-CHANGELOG.md}"

if [[ -z "$version" || "$version" == "-h" || "$version" == "--help" ]]; then
  usage
  [[ -z "$version" ]] && exit 2 || exit 0
fi

if [[ ! -f "$changelog" ]]; then
  echo "ERROR: changelog not found: ${changelog}" >&2
  exit 2
fi

core="${version#v}"
core="${core%%[[:space:]]*}"

awk -v target="$core" '
  /^## / {
    if (capturing) { exit }
    heading = $0
    sub(/^## /, "", heading)
    sub(/^v/, "", heading)
    # strip a trailing " - <date>" / " — <date>" suffix and surrounding space
    sub(/[[:space:]]+[-–—].*$/, "", heading)
    gsub(/[[:space:]]+$/, "", heading)
    if (heading == target) { capturing = 1; next }
    next
  }
  capturing { print }
' "$changelog" | sed -e '/./,$!d' | awk '
  # trim trailing blank lines
  { lines[NR] = $0 }
  END {
    last = NR
    while (last > 0 && lines[last] ~ /^[[:space:]]*$/) last--
    for (i = 1; i <= last; i++) print lines[i]
  }
'
