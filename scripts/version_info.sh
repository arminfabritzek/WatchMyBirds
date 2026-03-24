#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  version_info.sh release-base [repo_root]
  version_info.sh dev-version [repo_root] [git_sha]

Rules:
  - Prefer the latest semver git tag (vX.Y.Z or X.Y.Z) as the release base.
  - If no semver tag exists, derive the release base from APP_VERSION.
  - If APP_VERSION already contains a dev suffix for the next patch
    (for example 0.1.1-dev.abc1234), map it back to the current release base
    (0.1.0) so repeated runs stay idempotent.
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

latest_semver_tag() {
  local repo_root="${1:-.}"
  git -C "$repo_root" tag --list \
    | sed -nE 's/^(v?[0-9]+\.[0-9]+\.[0-9]+)$/\1/p' \
    | sed 's/^v//' \
    | sort -V \
    | tail -n 1
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

release_base_from_raw_app_version() {
  local raw core
  raw="$(trim "$1")"
  core="$(semver_core_from_raw "$raw")"

  if [[ "$raw" == *"-dev."* || "$raw" == *"+dev."* ]]; then
    decrement_patch "$core"
    return
  fi

  printf '%s\n' "$core"
}

resolve_release_base() {
  local repo_root="${1:-.}"
  local tag_version
  tag_version="$(latest_semver_tag "$repo_root")"
  if [[ -n "$tag_version" ]]; then
    printf '%s\n' "$tag_version"
    return
  fi

  release_base_from_raw_app_version "$(read_raw_app_version "$repo_root")"
}

resolve_dev_version() {
  local repo_root="${1:-.}"
  local git_sha="${2:-}"
  local release_base short_sha

  release_base="$(resolve_release_base "$repo_root")"
  if [[ -n "$git_sha" ]]; then
    short_sha="$(printf '%s' "$git_sha" | cut -c1-7)"
  else
    short_sha="$(git -C "$repo_root" rev-parse --short=7 HEAD)"
  fi

  printf '%s-dev.%s\n' "$(increment_patch "$release_base")" "$short_sha"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cmd="${1:-}"
  repo_root="${2:-.}"

  case "$cmd" in
    release-base)
      resolve_release_base "$repo_root"
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
