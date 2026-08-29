#!/usr/bin/env python3
"""Run one bounded physical PTZ burst and return to a known preset."""

from scripts.ptz_lab.motion import main_continuous

if __name__ == "__main__":
    raise SystemExit(main_continuous())
