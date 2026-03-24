#!/usr/bin/env bash
# update_version.sh — Write a dev version string into APP_VERSION.
#
# The generated format is:  {BASE_VERSION}+dev.{short_sha}
# Example:                  0.1.0+dev.39a7010
#
# Usage:
#   ./scripts/update_version.sh           # update APP_VERSION in-place, no commit
#   ./scripts/update_version.sh --push    # update, git-commit, and push
#
# The script is idempotent: if APP_VERSION already contains the current SHA
# it exits without creating a new commit.
#
# To bump the release version, change BASE_VERSION below and commit the script.
set -euo pipefail

# ---------------------------------------------------------------------------
# Base (release) version — update this when cutting a new release.
# ---------------------------------------------------------------------------
BASE_VERSION="0.1.0"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${REPO_ROOT}/APP_VERSION"
SHORT_SHA=$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)
NEW_VERSION="${BASE_VERSION}+dev.${SHORT_SHA}"

echo "${NEW_VERSION}" > "${VERSION_FILE}"
echo "APP_VERSION → ${NEW_VERSION}"

if [[ "${1:-}" == "--push" ]]; then
  git -C "${REPO_ROOT}" add "${VERSION_FILE}"

  # Nothing to commit means the version was already up to date.
  if git -C "${REPO_ROOT}" diff --cached --quiet; then
    echo "APP_VERSION already at ${NEW_VERSION}, nothing to commit."
    exit 0
  fi

  git -C "${REPO_ROOT}" commit -m "chore: set dev version ${NEW_VERSION}"
  git -C "${REPO_ROOT}" push
  echo "Pushed APP_VERSION to remote."
fi
