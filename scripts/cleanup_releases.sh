#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <repo> <tag-prefix> [keep-count]" >&2
  exit 1
fi

repo="$1"
tag_prefix="$2"
keep_count="${3:-1}"

if [[ ! "$keep_count" =~ ^[0-9]+$ ]]; then
  echo "keep-count must be a non-negative integer" >&2
  exit 1
fi

mapfile -t matching_tags < <(
  gh release list \
    --repo "$repo" \
    --limit 100 \
    --json tagName,createdAt \
    --jq "sort_by(.createdAt) | reverse | .[] | select(.tagName | startswith(\"${tag_prefix}\")) | .tagName"
)

if (( ${#matching_tags[@]} <= keep_count )); then
  echo "No cleanup required for prefix '${tag_prefix}'."
  exit 0
fi

for idx in "${!matching_tags[@]}"; do
  tag="${matching_tags[$idx]}"

  if (( idx < keep_count )); then
    echo "Keeping release ${tag}"
    continue
  fi

  echo "Deleting old release ${tag}"
  gh release delete "$tag" --repo "$repo" --yes --cleanup-tag
done
