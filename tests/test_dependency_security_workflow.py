"""The complete shipped dependency graph must pass the security audit."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("audit_exit_code", [0, 17])
def test_shipped_dependency_audit_propagates_failure(
    tmp_path: Path, audit_exit_code: int
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/security-scan.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["pip-audit"]["steps"]
        if step.get("name") == "Run pip-audit (shipped application)"
    )
    assert not step.get("continue-on-error", False)

    auditor = tmp_path / "pip-audit"
    auditor.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$AUDIT_ARGUMENTS"\nexit "$AUDIT_EXIT_CODE"\n'
    )
    auditor.chmod(0o755)
    arguments = tmp_path / "arguments.txt"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "AUDIT_ARGUMENTS": str(arguments),
            "AUDIT_EXIT_CODE": str(audit_exit_code),
        },
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == audit_exit_code
    args = arguments.read_text().splitlines()
    requirement_files = [
        args[index + 1] for index, arg in enumerate(args) if arg == "-r"
    ]
    assert set(requirement_files) == {"requirements.txt", "requirements-aesthetic.txt"}
    assert "--strict" in args


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_torch_requirements_select_cpu_builds_off_macos(platform: str) -> None:
    requirements = [
        Requirement(line)
        for line in (ROOT / "requirements-aesthetic.txt").read_text().splitlines()
        if line.startswith(("torch==", "torchvision=="))
    ]
    selected = [
        req
        for req in requirements
        if req.marker is None or req.marker.evaluate({"sys_platform": platform})
    ]
    assert sorted(req.name for req in selected) == ["torch", "torchvision"]
    for req in selected:
        pins = list(req.specifier)
        assert len(pins) == 1 and pins[0].operator == "=="
        assert pins[0].version.endswith("+cpu") == (platform != "darwin")
