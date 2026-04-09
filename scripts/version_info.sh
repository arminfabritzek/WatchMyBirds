#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  version_info.sh current-release [repo_root]
  version_info.sh next-release [repo_root]
  version_info.sh release-base [repo_root]
  version_info.sh dev-version [repo_root] [git_sha]

Rules:
  - Latest semver git tag is the primary source of truth for the current release.
  - APP_VERSION may match the current release, the next planned release, or a
    dev suffix for the next planned release.
  - current-release resolves the latest stable release version.
  - next-release resolves the version that the next stable release should use.
  - dev-version resolves as "{next-release}-dev.{short_sha}".
EOF
}

trim() {
  printf '%s' "$1" | tr -d '[:space:]'
}

require_semver_core() {
  local value
  value="$(trim "$1")"
  if [[ ! "$value" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "ERROR: expected semantic version core X.Y.Z, got '${value}'" >&2
    exit 1
  fi
  printf '%s\n' "$value"
}

semver_core_from_raw() {
  local raw
  raw="$(trim "$1")"
  raw="${raw#v}"
  raw="${raw%%+*}"
  raw="${raw%%-*}"
  require_semver_core "$raw"
}

increment_patch() {
  local core major minor patch
  core="$(require_semver_core "$1")"
  IFS=. read -r major minor patch <<< "$core"
  printf '%s\n' "${major}.${minor}.$((patch + 1))"
}

compare_semver() {
  local left right l_major l_minor l_patch r_major r_minor r_patch
  left="$(require_semver_core "$1")"
  right="$(require_semver_core "$2")"
  IFS=. read -r l_major l_minor l_patch <<< "$left"
  IFS=. read -r r_major r_minor r_patch <<< "$right"

  if (( l_major != r_major )); then
    (( l_major < r_major )) && echo -1 || echo 1
    return
  fi

  if (( l_minor != r_minor )); then
    (( l_minor < r_minor )) && echo -1 || echo 1
    return
  fi

  if (( l_patch != r_patch )); then
    (( l_patch < r_patch )) && echo -1 || echo 1
    return
  fi

  echo 0
}

decrement_patch() {
  local core major minor patch
  core="$(require_semver_core "$1")"
  IFS=. read -r major minor patch <<< "$core"
  if (( patch == 0 )); then
    echo "ERROR: cannot infer previous release from patch-zero dev version '${core}'" >&2
    exit 1
  fi
  printf '%s\n' "${major}.${minor}.$((patch - 1))"
}

read_raw_app_version() {
  local repo_root="${1:-.}"
  local version_file="${repo_root}/APP_VERSION"
  if [[ ! -f "$version_file" ]]; then
    echo "ERROR: APP_VERSION not found: ${version_file}" >&2
    exit 1
  fi
  tr -d '[:space:]' < "$version_file"
}

latest_tag_release() {
  local repo_root="${1:-.}"
  git -C "$repo_root" tag --list --sort=-version:refname \
    | sed -n 's/^v\{0,1\}\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$/\1/p' \
    | head -n 1
}

current_release_from_raw_app_version() {
  local raw core
  raw="$(trim "$1")"
  core="$(semver_core_from_raw "$raw")"

  if [[ "$raw" == *"-dev."* || "$raw" == *"+dev."* ]]; then
    decrement_patch "$core"
    return
  fi

  printf '%s\n' "$core"
}

next_release_from_raw_app_version() {
  local raw
  raw="$(trim "$1")"
  semver_core_from_raw "$raw"
}

resolve_current_release() {
  local repo_root="${1:-.}"
  local tagged
  tagged="$(latest_tag_release "$repo_root")"

  if [[ -n "$tagged" ]]; then
    printf '%s\n' "$tagged"
    return
  fi

  current_release_from_raw_app_version "$(read_raw_app_version "$repo_root")"
}

resolve_next_release() {
  local repo_root="${1:-.}"
  local tagged candidate cmp

  tagged="$(latest_tag_release "$repo_root")"
  candidate="$(next_release_from_raw_app_version "$(read_raw_app_version "$repo_root")")"

  if [[ -z "$tagged" ]]; then
    printf '%s\n' "$candidate"
    return
  fi

  cmp="$(compare_semver "$candidate" "$tagged")"
  if [[ "$cmp" == "-1" || "$cmp" == "0" ]]; then
    increment_patch "$tagged"
    return
  fi

  printf '%s\n' "$candidate"
}

resolve_dev_version() {
  local repo_root="${1:-.}"
  local git_sha="${2:-}"
  local next_release short_sha

  next_release="$(resolve_next_release "$repo_root")"
  if [[ -n "$git_sha" ]]; then
    short_sha="$(printf '%s' "$git_sha" | cut -c1-7)"
  else
    short_sha="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
  fi

  printf '%s-dev.%s\n' "$next_release" "$short_sha"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cmd="${1:-}"
  repo_root="${2:-.}"

  case "$cmd" in
    current-release)
      resolve_current_release "$repo_root"
      ;;
    next-release)
      resolve_next_release "$repo_root"
      ;;
    release-base)
      resolve_current_release "$repo_root"
      ;;
    dev-version)
      resolve_dev_version "$repo_root" "${3:-}"
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "ERROR: unknown command '${cmd}'" >&2
      usage >&2
      exit 1
      ;;
  esac
fi
