#!/usr/bin/env python3
"""Run the read-only ONVIF camera identity probe."""

from scripts.ptz_lab.common import main_identity

if __name__ == "__main__":
    raise SystemExit(main_identity())
