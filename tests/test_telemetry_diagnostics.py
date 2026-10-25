from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from flask import Flask

from core import telemetry_metrics as tm
from web.services import telemetry_service as ts


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    value = {
        "OUTPUT_DIR": str(tmp_path),
        "telemetry_enabled": True,
        "telemetry_consent_version": 2,
        "telemetry_installation_id": "a" * 32,
    }
    monkeypatch.setattr(tm, "get_config", lambda: value)
    monkeypatch.setattr(ts, "metrics", tm.DailyMetrics())
    monkeypatch.setattr(
        ts,
        "_delivery",
        {
            "status": "not_attempted",
            "last_attempt": None,
            "last_success": None,
            "failures": 0,
            "retry_at": 0.0,
        },
    )
    return value


@pytest.mark.parametrize("version", [None, 0, 1, True, "2", 3])
def test_no_diagnostics_without_explicit_current_consent(
    cfg: dict, version: object
) -> None:
    cfg["telemetry_consent_version"] = version
    ts.metrics.record("review_used")
    assert not list(Path(cfg["OUTPUT_DIR"]).iterdir())
    assert ts.metrics.pending(cfg, int(time.time())) == []


def test_only_bounded_metrics_are_persisted_and_recovered(cfg: dict) -> None:
    for key, value in [
        ("review_used", 1),
        ("inference_latency_bucket", 3),
        ("camera_url", 1),
        ("model_error", 2),
        ("uptime_bucket", 99),
    ]:
        ts.metrics.record(key, value)
    ts.metrics.record("inference_latency_bucket", 1)
    restored = tm.DailyMetrics()
    pending = restored.pending(cfg, int(time.time()))
    assert len(pending) == 1
    assert pending[0][1] == {"review_used": 1, "inference_latency_bucket": 3}


def test_retry_keeps_daily_data_until_acknowledged(cfg: dict) -> None:
    ts.metrics.record("review_used")
    response = Mock(status_code=503, headers={})
    with patch.object(ts.requests, "post", return_value=response) as post:
        ts.send_pending_diagnostics(cfg)
        ts.send_pending_diagnostics(cfg)
    assert post.call_count == 1
    assert len(ts.metrics.pending(cfg, int(time.time()))) == 1
    assert ts._read_last_sent_date(cfg["OUTPUT_DIR"]) == ""
    ts._delivery["retry_at"] = 0
    response.status_code = 204
    with patch.object(ts.requests, "post", return_value=response) as post:
        ts.send_pending_diagnostics(cfg)
        ts.send_pending_diagnostics(cfg)
    assert post.call_count == 1
    sent = post.call_args.kwargs["json"]
    assert sent["consent_version"] == 2
    assert sent["metrics"]["review_used"] == 1
    assert ts._read_last_sent_date(cfg["OUTPUT_DIR"]) == ts._today_utc()


def test_queue_catches_up_completed_days_and_expires_old_data(cfg: dict) -> None:
    today = datetime.now(UTC).date()
    path = Path(cfg["OUTPUT_DIR"]) / "telemetry_metrics.json"
    path.write_text(
        json.dumps(
            {
                "id": cfg["telemetry_installation_id"],
                "days": {
                    (today - timedelta(days=8)).isoformat(): {"review_used": 1},
                    (today - timedelta(days=7)).isoformat(): {"export_used": 1},
                    (today - timedelta(days=1)).isoformat(): {"camera_frame": 1},
                },
                "sent": {},
            }
        )
    )
    pending = ts.metrics.pending(cfg, int(time.time()))
    assert len(pending) == 2
    for day, values in pending:
        ts.metrics.acknowledge(cfg, day, values, int(time.time()))
    assert ts.metrics.pending(cfg, int(time.time())) == []
    assert (today - timedelta(days=8)).isoformat() not in path.read_text()


@pytest.mark.parametrize("enabled,version", [(False, 2), (True, 1)])
def test_revoke_or_downgrade_clears_queue(
    cfg: dict, enabled: bool, version: int
) -> None:
    ts.metrics.record("review_used")
    ts.set_consent(cfg, enabled, version)
    ts.metrics.record("camera_frame")
    data = json.loads((Path(cfg["OUTPUT_DIR"]) / "telemetry_metrics.json").read_text())
    assert data["days"] == {}
    assert data["sent"] == {}
    assert cfg["telemetry_consent_at"]
    assert cfg["telemetry_installation_id"] == "a" * 32


def test_rotated_id_does_not_reuse_queued_events(cfg: dict) -> None:
    ts.metrics.record("review_used")
    cfg["telemetry_installation_id"] = "b" * 32
    assert ts.metrics.pending(cfg, int(time.time())) == []


def test_transport_refuses_redirects_and_plaintext_external_hosts(cfg: dict) -> None:
    with patch.object(
        ts.requests, "post", return_value=Mock(status_code=307, headers={})
    ) as post:
        assert not ts._send_heartbeat("http://external.example/heartbeat", {})
        post.assert_not_called()
        assert not ts._send_heartbeat("https://example.invalid/heartbeat", {})
        assert post.call_args.kwargs["allow_redirects"] is False
        assert post.call_args.kwargs["stream"] is True
        post.return_value.close.assert_called_once()


def test_web_metrics_require_successful_authenticated_actions(cfg: dict) -> None:
    ts.record_web_response("review.review_page", "GET", 200, False)
    ts.record_web_response("human_labels.label_answer", "POST", 400, True)
    assert ts.metrics.pending(cfg, int(time.time())) == []
    ts.record_web_response("review.review_page", "GET", 200, True)
    ts.record_web_response("human_labels.label_answer", "POST", 200, True)
    ts.record_web_response(
        "canonical_dataset.canonical_dataset_prepare", "POST", 201, True
    )
    ts.record_web_response("other.route", "GET", 500, True)
    assert ts.metrics.pending(cfg, int(time.time()))[0][1] == {
        "review_used": 1,
        "correction_used": 1,
        "export_used": 1,
        "web_error": 1,
    }


def test_source_version_falls_back_to_git_without_sending_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APP_VERSION", raising=False)
    with (
        patch.object(ts.Path, "exists", return_value=False),
        patch.object(
            ts.subprocess, "run", return_value=Mock(stdout="v1.2.3-4-gabcdef-dirty\n")
        ) as git,
    ):
        assert ts._detect_app_version() == "v1.2.3-4-gabcdef-dirty"
        assert git.call_args.kwargs["timeout"] == 2


def test_consent_api_requires_boolean_and_explicit_version(
    cfg: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from web.blueprints import api_v1
    from web.blueprints.auth import auth_service

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(api_v1.api_v1, url_prefix="/api/v1")
    monkeypatch.setattr(api_v1, "get_config", lambda: cfg)
    monkeypatch.setattr(auth_service, "should_require_password_setup", lambda: False)
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        for body in (
            [],
            {"enabled": "yes"},
            {"enabled": True, "consent_version": True},
            {"enabled": True, "consent_version": 3},
        ):
            assert (
                client.post("/api/v1/settings/telemetry", json=body).status_code == 400
            )
        assert (
            client.post(
                "/api/v1/settings/telemetry", json={"enabled": True}
            ).status_code
            == 200
        )
        assert cfg["telemetry_consent_version"] == 1
        assert (
            client.post(
                "/api/v1/settings/telemetry",
                json={"enabled": True, "consent_version": 2},
            ).status_code
            == 200
        )
        assert cfg["telemetry_consent_version"] == 2


def test_corrupt_queue_cannot_inject_payload_fields(cfg: dict) -> None:
    day = datetime.now(UTC).date().isoformat()
    path = Path(cfg["OUTPUT_DIR"]) / "telemetry_metrics.json"
    path.write_text(
        json.dumps(
            {
                "id": cfg["telemetry_installation_id"],
                "days": {
                    day: {
                        "camera_url": "secret",
                        "review_used": 1,
                        "uptime_bucket": 99,
                    },
                    "bad-date": ["secret"],
                },
                "sent": {day: "invalid"},
            }
        )
    )
    assert ts.metrics.pending(cfg, int(time.time())) == [(day, {"review_used": 1})]


def test_observations_after_hourly_ack_are_not_lost_at_day_end(cfg: dict) -> None:
    day = datetime.now(UTC).date().isoformat()
    ts.metrics.record("review_used")
    ts.metrics.acknowledge(cfg, day, {"review_used": 1}, int(time.time()))
    ts.metrics.record("export_used")
    assert ts.metrics.pending(cfg, int(time.time())) == []
    assert ts.metrics.pending(cfg, int(time.time()) + 3601)[0][1]["export_used"] == 1


@pytest.mark.parametrize(
    "basic,diagnostics,expected", [(False, True, 0), (True, False, 1), (True, True, 2)]
)
def test_setup_requires_both_choices_for_diagnostics(
    cfg: dict,
    monkeypatch: pytest.MonkeyPatch,
    basic: bool,
    diagnostics: bool,
    expected: int,
) -> None:
    import config
    from web.blueprints.auth import auth_bp, auth_service, settings_service

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(auth_bp)
    cfg.update(telemetry_enabled=False, telemetry_consent_version=0)
    monkeypatch.setattr(config, "get_config", lambda: cfg)
    monkeypatch.setattr(auth_service, "should_require_password_setup", lambda: True)
    monkeypatch.setattr(settings_service, "update_settings", lambda values: (True, []))
    form = {
        "password": "valid-test-password",
        "password_confirm": "valid-test-password",
    }
    if basic:
        form["telemetry_optin"] = "1"
    if diagnostics:
        form["telemetry_diagnostics"] = "2"
    with app.test_client() as client:
        assert client.post("/setup/password", data=form).status_code == 302
    assert cfg["telemetry_consent_version"] == expected
    assert cfg["telemetry_enabled"] is basic


def test_new_consent_period_never_replays_old_queue(cfg: dict) -> None:
    ts.metrics.record("review_used")
    cfg["telemetry_consent_at"] = "new-consent-period"
    assert tm.DailyMetrics().pending(cfg, int(time.time())) == []


def test_record_retries_local_persistence_failure(cfg: dict) -> None:
    with patch.object(ts.metrics, "_save", side_effect=OSError("disk unavailable")):
        ts.metrics.record("camera_frame")
    ts.metrics.record("camera_frame")
    assert tm.DailyMetrics().pending(cfg, int(time.time()))[0][1] == {"camera_frame": 1}
