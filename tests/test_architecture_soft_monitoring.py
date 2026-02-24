"""
Architecture soft monitoring checks.

These checks are intentionally non-blocking and report trend metrics only.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_INTERFACE = PROJECT_ROOT / "web" / "web_interface.py"
BLUEPRINT_DIR = PROJECT_ROOT / "web" / "blueprints"
ROUTE_FILES = [WEB_INTERFACE, *sorted(BLUEPRINT_DIR.glob("*.py"))]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _append_report(lines: list[str]) -> None:
    report_path = os.environ.get("SOFT_MONITOR_REPORT", "").strip()
    if not report_path:
        return
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


@pytest.mark.arch_soft
def test_monitor_route_footprint() -> None:
    if not WEB_INTERFACE.exists():
        print("SOFT MONITOR: web/web_interface.py missing")
        return

    text = _read(WEB_INTERFACE)
    route_count = len(re.findall(r"@server\.route\(", text))
    add_rule_count = len(re.findall(r"server\.add_url_rule\(", text))

    lines = [
        "SOFT MONITOR: route footprint",
        f"  web_interface.route_decorators={route_count}",
        f"  web_interface.add_url_rule_calls={add_rule_count}",
    ]
    for line in lines:
        print(line)
    _append_report(lines)


@pytest.mark.arch_soft
def test_monitor_direct_io_patterns_in_routes() -> None:
    pattern_defs = {
        "db_connection_calls": r"db_service\.get_connection\(",
        "direct_sql_execute": r"(?:cursor|conn)\.execute\(",
        "subprocess_calls": r"subprocess\.run\(",
        "thread_spawns": r"threading\.Thread\(",
        "direct_file_open": r"\bopen\(",
    }

    totals = {name: 0 for name in pattern_defs}
    for path in ROUTE_FILES:
        text = _read(path)
        for name, pattern in pattern_defs.items():
            totals[name] += len(re.findall(pattern, text))

    lines = ["SOFT MONITOR: direct IO/system patterns in route files"]
    lines.extend(f"  {name}={count}" for name, count in totals.items())
    for line in lines:
        print(line)
    _append_report(lines)


@pytest.mark.arch_soft
def test_monitor_blueprint_mutable_state_markers() -> None:
    mutable_state_pattern = re.compile(r"^_[A-Za-z0-9_]+\s*=\s*", re.MULTILINE)
    total = 0

    for path in sorted(BLUEPRINT_DIR.glob("*.py")):
        text = _read(path)
        total += len(mutable_state_pattern.findall(text))

    lines = [
        "SOFT MONITOR: mutable module state markers",
        f"  web_blueprints.module_level_mutable_bindings={total}",
    ]
    for line in lines:
        print(line)
    _append_report(lines)
