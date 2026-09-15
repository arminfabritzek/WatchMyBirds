"""Manual backup diagnostics must survive subprocess failure."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

from web.services import usb_backup_service as service


def test_manual_backup_captures_diagnostics_and_exit(monkeypatch):
    monkeypatch.setattr(service, "is_trigger_supported", lambda: True)
    monkeypatch.setattr(
        service.usb_backup_core,
        "get_stick_status",
        lambda: SimpleNamespace(state="connected"),
    )
    process = SimpleNamespace(
        pid=123, stdout=io.StringIO("permission denied\n"), wait=lambda: 15
    )
    spawn = Mock(return_value=process)
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        service.threading,
        "Thread",
        lambda target, **kwargs: SimpleNamespace(start=target),
    )
    log = Mock()
    monkeypatch.setattr(service, "logger", log)
    monkeypatch.setattr(service, "_LAST_MANUAL_TRIGGER", None)
    started, _, _ = service.trigger_manual_backup()
    assert started
    assert spawn.call_args.kwargs["stderr"] == service.subprocess.STDOUT
    log.info.assert_any_call("USB backup: %s", "permission denied")
    log.error.assert_called_once_with("Manual USB backup failed (exit code %s)", 15)
