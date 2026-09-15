"""Check the files actually handed from image assembly to provisioning."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["harden.sh", "harden-dev.sh"])
def test_maintenance_tmpfiles_installation_is_executable_in_isolation(tmp_path, script):
    source = (ROOT / "rpi" / script).read_text()
    start = source.index("# Recreate the shared lock")
    end = source.index("install -d -o root -g watchmybirds", start)
    fragment = source[start:end].replace("/etc/tmpfiles.d", str(tmp_path))
    # Only execute the extracted file-writing fragment; never host provisioning.
    fragment = fragment.replace("systemd-tmpfiles --create", "test -f")
    subprocess.run(["bash", "-eu", "-c", fragment], check=True)
    lines = (tmp_path / "watchmybirds-maintenance.conf").read_text().splitlines()
    assert lines == [
        "d /run/lock/watchmybirds 0750 root watchmybirds -",
        "f /run/lock/watchmybirds/maintenance.lock 0660 root watchmybirds -",
    ]


def test_backup_writers_have_matching_sandbox_paths():
    for unit in ["systemd/app.service", "rpi/systemd/wmb-backup.service"]:
        text = (ROOT / unit).read_text()
        paths = re.search(r"^ReadWritePaths=(.*)$", text, re.M)[1].split()
        assert "/mnt/wmb-backup" in paths
        assert "/run/lock/watchmybirds" in paths
    for script in ["rpi/backup.sh", "rpi/format_backup_stick.sh"]:
        assert (
            'exec 8<>"/run/lock/watchmybirds/maintenance.lock"'
            in (ROOT / script).read_text()
        )


@pytest.mark.parametrize("workflow", ["build-release.yml", "build-release-test.yml"])
def test_production_injection_preserves_provisioned_runtime_directories(workflow):
    source = (ROOT / ".github/workflows" / workflow).read_text()
    injection = next(
        line for line in source.splitlines() if "rsync -av --delete" in line
    )
    assert "--exclude='/data/'" in injection


@pytest.mark.parametrize(
    "workflow", ["build-dev.yml", "build-release.yml", "build-release-test.yml"]
)
def test_pi_wheelhouse_and_install_include_both_dependency_stacks(workflow):
    config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    steps = next(
        job["steps"]
        for job in config["jobs"].values()
        if "steps" in job
        and any("Wheelhouse" in step.get("name", "") for step in job["steps"])
    )
    download = next(
        step["run"] for step in steps if step.get("name") == "Create Wheelhouse (Host)"
    )
    install = next(
        step["run"]
        for step in steps
        if step.get("name", "").startswith("Mount and Inject")
    )
    for requirements in ["requirements.txt", "requirements-aesthetic.txt"]:
        assert f"-r app_code/{requirements}" in download
        assert f"-r /opt/app/{requirements}" in install


def test_docker_includes_offline_cli_and_no_pi_provisioning():
    ignore = (ROOT / ".dockerignore").read_text()
    assert ignore.index("!scripts/recover_from_snapshot.py") > ignore.index(
        "scripts/\n"
    )
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY scripts ./scripts" in dockerfile
    assert "harden.sh" not in dockerfile
    assert "harden.sh" not in (ROOT / "entrypoint.sh").read_text()
