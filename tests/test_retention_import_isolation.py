"""Retention import isolation.

Importing the retention service (or its core) must not transitively load
the ONVIF / camera-discovery subsystem (camera.network_scanner, ifaddr).
That subsystem is unrelated to retention; pulling it in via the
``web.services`` package __init__ made the retention-API tests fragile
against an undeclared/optional dependency.

Run in a fresh subprocess because module-import state is process-global —
another test may already have imported the camera stack in-process.
"""

import subprocess
import sys
import textwrap


def _leaked_modules(import_line: str) -> list[str]:
    # Import may print incidental lines (e.g. a generated secret key); emit
    # the result behind a sentinel so we parse only our own line.
    code = textwrap.dedent(
        f"""
        import sys
        {import_line}
        leaked = sorted(
            m for m in sys.modules
            if m == "ifaddr" or m.startswith("camera")
        )
        print("LEAKED:" + ",".join(leaked))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"OUTPUT_DIR": "/tmp/wmb_import_isolation", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    line = next(
        ln for ln in out.stdout.splitlines() if ln.startswith("LEAKED:")
    )
    return [m for m in line[len("LEAKED:") :].split(",") if m]


def test_importing_retention_service_does_not_load_camera_or_ifaddr():
    assert _leaked_modules("import web.services.retention_service") == []


def test_importing_retention_core_does_not_load_camera_or_ifaddr():
    assert _leaked_modules("import core.retention_core") == []
