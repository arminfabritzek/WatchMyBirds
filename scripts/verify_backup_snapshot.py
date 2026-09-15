#!/usr/bin/env python3
"""Validate a completed snapshot, or validate a backup before publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.usb_backup_core import verify_snapshot_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--before-completion", action="store_true")
    args = parser.parse_args()
    result = verify_snapshot_directory(
        args.snapshot, require_completed=not args.before_completion
    )
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
