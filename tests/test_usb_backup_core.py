"""Unit tests for core.usb_backup_core.

The core handles snapshot inspection, stick-state probing, and
deletion. We test against a tmp-path that mimics the on-stick layout
(/mnt/wmb-backup/snapshots/<stamp>_<kind>/...) by monkey-patching the
module-level path constants.

Why monkey-patch instead of dependency-injection? The real script
(rpi/backup.sh) hardcodes the same paths, so the constants are the
documentation -- making them injectable would weaken the contract.
Tests get to override them; production code does not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import usb_backup_core  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures: mimic /mnt/wmb-backup/ in tmp_path
# ----------------------------------------------------------------------


@pytest.fixture
def fake_stick(tmp_path, monkeypatch):
    """Build a tmp tree shaped like /mnt/wmb-backup/ and re-point the
    core's module-level constants at it.

    Returns the mount-point path; helpers below add snapshots to it.
    """
    mount = tmp_path / "mnt-wmb-backup"
    mount.mkdir()
    snaps = mount / "snapshots"
    snaps.mkdir()

    monkeypatch.setattr(usb_backup_core, "MOUNT_POINT", mount)
    monkeypatch.setattr(usb_backup_core, "BACKUP_DEVICE", mount / "dev" / "WMB-BACKUP")
    monkeypatch.setattr(usb_backup_core, "SNAPSHOTS_DIR", snaps)
    monkeypatch.setattr(usb_backup_core, "LATEST_LINK", mount / "latest")
    monkeypatch.setattr(usb_backup_core, "BACKUP_LOG", mount / "BACKUP_LOG.txt")

    # _is_mounted normally checks os.path.ismount -- in tests, the tmp
    # path is a regular dir, not a mount. Stub the check.
    monkeypatch.setattr(usb_backup_core, "_is_mounted", lambda p: True)
    # findmnt would also fail for a regular dir; pretend we're on ext4.
    monkeypatch.setattr(usb_backup_core, "_findmnt_fstype", lambda p: "ext4")
    # No labelled device probe unless a test opts into it.
    monkeypatch.setattr(usb_backup_core, "_blkid_fstype", lambda p: None)

    return mount


def _make_snapshot(
    stick: Path,
    name: str,
    *,
    kind: str = "scheduled",
    completed: bool = True,
    corrupt: bool = False,
    corrupt_reason: str | None = None,
    manifest_extra: dict | None = None,
) -> Path:
    """Materialize one snapshot directory under <stick>/snapshots/<name>/."""
    d = stick / "snapshots" / name
    (d / "data").mkdir(parents=True)
    (d / "app").mkdir()
    (d / "kind").write_text(kind)

    manifest: dict = {
        "schema_version": 1,
        "snapshot_name": name,
        "kind": kind,
        "started_at": "2026-04-29T03:00:00Z",
        "completed_at": "2026-04-29T03:01:23Z",
        "app_version": "0.2.8",
        "host": "test-pi",
        "user": "watchmybirds",
        "previous_snapshot": "",
        "sizes": {
            "db_bytes": 1024,
            "output_bytes": 4096,
            "app_bytes": 2048,
            "total_bytes": 8192,
        },
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (d / "manifest.json").write_text(json.dumps(manifest))

    if completed:
        (d / "COMPLETED").write_text("2026-04-29T03:01:23Z")
    if corrupt:
        (d / "CORRUPT").write_text(
            f"reason: {corrupt_reason or 'integrity_check failed'}\n"
            "ts: 2026-04-29T03:01:23Z"
        )
    return d


# ----------------------------------------------------------------------
# get_stick_status
# ----------------------------------------------------------------------


class TestStickStatus:
    def test_connected_when_ext4_and_writable(self, fake_stick):
        status = usb_backup_core.get_stick_status()
        assert status.state == "connected"
        assert status.fstype == "ext4"
        assert status.total_bytes is not None
        assert status.free_bytes is not None
        assert status.detail is None

    def test_missing_when_not_mounted(self, fake_stick, monkeypatch):
        monkeypatch.setattr(usb_backup_core, "_is_mounted", lambda p: False)
        status = usb_backup_core.get_stick_status()
        assert status.state == "missing"
        assert status.detail and "stick not detected" in status.detail.lower()

    def test_wrong_fs_rejected_with_helpful_detail(self, fake_stick, monkeypatch):
        monkeypatch.setattr(usb_backup_core, "_findmnt_fstype", lambda p: "vfat")
        status = usb_backup_core.get_stick_status()
        assert status.state == "wrong_fs"
        assert status.fstype == "vfat"
        # The detail must mention ext4 -- this is how the UI helps the
        # user understand they need to reformat.
        assert "ext4" in (status.detail or "")

    def test_wrong_fs_reported_when_labelled_device_is_not_mounted(
        self, fake_stick, monkeypatch
    ):
        monkeypatch.setattr(usb_backup_core, "_is_mounted", lambda p: False)
        monkeypatch.setattr(usb_backup_core, "_blkid_fstype", lambda p: "exfat")
        status = usb_backup_core.get_stick_status()
        assert status.state == "wrong_fs"
        assert status.fstype == "exfat"
        assert "ext4" in (status.detail or "")


# ----------------------------------------------------------------------
# list_snapshots / get_snapshot
# ----------------------------------------------------------------------


class TestListSnapshots:
    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            usb_backup_core, "SNAPSHOTS_DIR", tmp_path / "does-not-exist"
        )
        monkeypatch.setattr(usb_backup_core, "_is_mounted", lambda p: True)
        assert usb_backup_core.list_snapshots() == []

    def test_newest_first(self, fake_stick):
        _make_snapshot(fake_stick, "20260427_030000_scheduled")
        _make_snapshot(fake_stick, "20260428_030000_scheduled")
        _make_snapshot(fake_stick, "20260429_030000_scheduled")
        out = usb_backup_core.list_snapshots()
        assert [s.name for s in out] == [
            "20260429_030000_scheduled",
            "20260428_030000_scheduled",
            "20260427_030000_scheduled",
        ]

    def test_limit_caps_returned(self, fake_stick):
        for i in range(5):
            _make_snapshot(fake_stick, f"2026042{i}_030000_scheduled")
        out = usb_backup_core.list_snapshots(limit=2)
        assert len(out) == 2

    def test_corrupt_snapshot_carries_reason(self, fake_stick):
        _make_snapshot(
            fake_stick, "20260429_030000_scheduled",
            corrupt=True, corrupt_reason="sha256 mismatch",
        )
        out = usb_backup_core.list_snapshots()
        assert out[0].corrupt is True
        assert out[0].corrupt_reason == "sha256 mismatch"

    def test_incomplete_snapshot_is_listed_but_marked(self, fake_stick):
        _make_snapshot(
            fake_stick, "20260429_030000_scheduled", completed=False
        )
        out = usb_backup_core.list_snapshots()
        assert len(out) == 1
        assert out[0].completed is False

    def test_latest_symlink_resolves_correctly(self, fake_stick):
        target = _make_snapshot(fake_stick, "20260429_030000_scheduled")
        _make_snapshot(fake_stick, "20260428_030000_scheduled")
        # backup.sh writes /mnt/wmb-backup/latest -> snapshots/<stamp>
        os.symlink(target, fake_stick / "latest")
        out = usb_backup_core.list_snapshots()
        latest = next(s for s in out if s.name == "20260429_030000_scheduled")
        not_latest = next(
            s for s in out if s.name == "20260428_030000_scheduled"
        )
        assert latest.is_latest is True
        assert not_latest.is_latest is False


class TestGetSnapshot:
    def test_resolves_known_snapshot(self, fake_stick):
        _make_snapshot(fake_stick, "20260429_120000_manual", kind="manual")
        snap = usb_backup_core.get_snapshot("20260429_120000_manual")
        assert snap is not None
        assert snap.kind == "manual"
        assert snap.app_version == "0.2.8"

    def test_returns_none_for_unknown(self, fake_stick):
        assert usb_backup_core.get_snapshot("nope") is None

    @pytest.mark.parametrize(
        "evil",
        [
            "../etc/passwd",
            "..",
            "/absolute/path",
            "with/slash",
            "",
        ],
    )
    def test_rejects_path_traversal(self, fake_stick, evil):
        # Even if such a path "exists" outside snapshots/, the helper
        # must refuse it -- this is the API endpoint's only line of
        # defense once a name reaches the route.
        assert usb_backup_core.get_snapshot(evil) is None


# ----------------------------------------------------------------------
# delete_snapshot
# ----------------------------------------------------------------------


class TestDeleteSnapshot:
    def test_deletes_existing(self, fake_stick):
        _make_snapshot(fake_stick, "20260429_120000_manual", kind="manual")
        ok, msg = usb_backup_core.delete_snapshot("20260429_120000_manual")
        assert ok is True
        assert "deleted" in msg.lower()
        assert not (fake_stick / "snapshots" / "20260429_120000_manual").exists()

    def test_refuses_unknown(self, fake_stick):
        ok, msg = usb_backup_core.delete_snapshot("nope")
        assert ok is False
        assert "not found" in msg.lower()

    @pytest.mark.parametrize(
        "evil",
        [
            "../snapshots/x",
            "../../etc/passwd",
            "/absolute/path",
            "with/slash",
            "",
        ],
    )
    def test_refuses_path_traversal(self, fake_stick, evil):
        # Refuse any name that contains a separator or '..'.  We don't
        # care WHICH error fires -- the postcondition is that nothing
        # outside snapshots/ is touched.
        ok, _ = usb_backup_core.delete_snapshot(evil)
        assert ok is False


# ----------------------------------------------------------------------
# get_backup_summary
# ----------------------------------------------------------------------


class TestSummary:
    def test_summary_with_snapshots(self, fake_stick):
        _make_snapshot(fake_stick, "20260429_030000_scheduled")
        _make_snapshot(fake_stick, "20260428_030000_scheduled")
        summary = usb_backup_core.get_backup_summary()
        assert summary["stick"]["state"] == "connected"
        assert len(summary["snapshots_recent"]) == 2
        assert summary["most_recent_completed"] is not None
        assert (
            summary["most_recent_completed"]["name"]
            == "20260429_030000_scheduled"
        )

    def test_summary_skips_corrupt_for_most_recent(self, fake_stick):
        # Newest is corrupt; the "most recent completed" should fall
        # back to the older clean one.
        _make_snapshot(
            fake_stick, "20260429_030000_scheduled",
            corrupt=True, corrupt_reason="sha mismatch",
        )
        _make_snapshot(fake_stick, "20260428_030000_scheduled")
        summary = usb_backup_core.get_backup_summary()
        assert summary["most_recent_completed"] is not None
        assert (
            summary["most_recent_completed"]["name"]
            == "20260428_030000_scheduled"
        )

    def test_summary_handles_missing_stick(self, fake_stick, monkeypatch):
        monkeypatch.setattr(usb_backup_core, "_is_mounted", lambda p: False)
        summary = usb_backup_core.get_backup_summary()
        assert summary["stick"]["state"] == "missing"
        assert summary["snapshots_recent"] == []
        assert summary["most_recent_completed"] is None

    def test_warn_almost_full_threshold(self, fake_stick, monkeypatch):
        # Force free_pct < 20 by mocking shutil.disk_usage.
        from collections import namedtuple

        Usage = namedtuple("Usage", "total used free")
        monkeypatch.setattr(
            usb_backup_core.shutil,
            "disk_usage",
            lambda p: Usage(total=100, used=85, free=15),
        )
        summary = usb_backup_core.get_backup_summary()
        assert summary["warn_almost_full"] is True
