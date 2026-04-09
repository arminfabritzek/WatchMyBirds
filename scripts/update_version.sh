#!/usr/bin/env bash
# update_version.sh — Write the next dev version string into APP_VERSION.
#
# The generated format is:  {next_release}-dev.{short_sha}
# Example:                  0.1.1-dev.39a7010
#
# Usage:
#   ./scripts/update_version.sh           # update APP_VERSION in-place, no commit
#   ./scripts/update_version.sh --push    # update, git-commit, and push
#
# Version source of truth:
#   1. latest semver git tag (vX.Y.Z / X.Y.Z), if present
#   2. APP_VERSION as a fallback / explicit next-release override
#
# Dev builds are always derived as "next release + sha". Example:
#   latest release 0.1.0 -> dev version 0.1.1-dev.<sha>
#   APP_VERSION 1.4.8    -> dev version 1.4.8-dev.<sha>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_PATH="APP_VERSION"
VERSION_FILE="${REPO_ROOT}/${VERSION_PATH}"
VERSION_HELPER="${REPO_ROOT}/scripts/version_info.sh"

if [[ ! -x "${VERSION_HELPER}" ]]; then
  echo "ERROR: version helper missing or not executable: ${VERSION_HELPER}" >&2
  exit 1
fi

NEW_VERSION="$("${VERSION_HELPER}" dev-version "${REPO_ROOT}")"

echo "${NEW_VERSION}" > "${VERSION_FILE}"
echo "APP_VERSION → ${NEW_VERSION}"

if [[ "${1:-}" == "--push" ]]; then
  git -C "${REPO_ROOT}" add -- "${VERSION_PATH}"

  # Commit only APP_VERSION so unrelated staged changes are never swept in.
  if git -C "${REPO_ROOT}" diff --cached --quiet -- "${VERSION_PATH}"; then
    echo "APP_VERSION already at ${NEW_VERSION}, nothing to commit."
    exit 0
  fi

  git -C "${REPO_ROOT}" commit -m "chore: set dev version ${NEW_VERSION}" --only -- "${VERSION_PATH}"
  git -C "${REPO_ROOT}" push
  echo "Pushed APP_VERSION to remote."
fi
