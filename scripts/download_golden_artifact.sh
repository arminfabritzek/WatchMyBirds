#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <repo> [run-id]" >&2
  exit 1
fi

repo="$1"
run_id="${2:-}"

rm -f golden.img.xz

if [[ -n "$run_id" ]]; then
  echo "Downloading Golden artifact from workflow run ${run_id}"
  gh run download "$run_id" --repo "$repo" --name golden-image
else
  echo "Downloading latest Golden artifact"
  gh run download --repo "$repo" --name golden-image
fi

if [[ ! -f golden.img.xz ]]; then
  echo "::error::Golden artifact did not contain golden.img.xz" >&2
  exit 1
fi
