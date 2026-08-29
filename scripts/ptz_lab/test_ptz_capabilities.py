#!/usr/bin/env python3
"""Run the read-only ONVIF PTZ and imaging capability probe."""

from scripts.ptz_lab.common import main_capabilities

if __name__ == "__main__":
    raise SystemExit(main_capabilities())
