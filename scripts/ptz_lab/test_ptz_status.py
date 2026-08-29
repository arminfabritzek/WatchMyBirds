#!/usr/bin/env python3
"""Run the read-only ONVIF PTZ status stability probe."""

from scripts.ptz_lab.common import main_status

if __name__ == "__main__":
    raise SystemExit(main_status())
