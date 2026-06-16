#!/bin/sh
# Pre-push security-sink guard. Wired in by scripts/install_git_hooks.sh.
# See tests/test_security_sink_guard.py for what it blocks.
set -e

repo_root="$(git rev-parse --show-toplevel)"
py="$repo_root/.venv/bin/python"
[ -x "$py" ] || py="python3"

# Direct --check entry point (no pytest) keeps the push fork-sparing.
if ! "$py" "$repo_root/tests/test_security_sink_guard.py" --check; then
  echo "Push blocked: new security-sink violation (str(exc) in response / unsanitized log)."
  echo "Route through web.security helpers (see output above)."
  exit 1
fi
