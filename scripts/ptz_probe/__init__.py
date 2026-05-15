"""WMB PTZ capability probe — operator-attended ONVIF capability tool.

Invoke from the WMB repo root:

    python -m scripts.ptz_probe --ip <cam-ip> --user <username>

When you pass --camera-id together with --output-dir (or set
WMB_OUTPUT_DIR), the probe also writes the canonical empirical-
capabilities cache the Settings UI reads:

    OUTPUT_DIR/ptz_capabilities/cam<id>.yaml

See scripts/ptz_probe/README.md for the full operator manual and
the safety notes (mandatory Home preset, emergency stop, etc).
"""
