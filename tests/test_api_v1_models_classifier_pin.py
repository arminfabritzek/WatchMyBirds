"""Tests for /api/v1/models/classifier/pin endpoint.

Targets the regression fix: a pin click MUST force-refresh the
classifier companion files (model_config.yaml + metrics.json) from
HuggingFace so model-side threshold updates land instantly, mirroring
the detector pin's behaviour.

Mocking strategy: ``_fetch_companion_files`` is patched so the test
suite never hits the network. We assert on call args (was it called
with ``force_refresh=True``? against the classifier HF base URL?
for the right model_id?) instead of on file content.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A fake classifier cache dir with two pinnable variants.

    Layout mirrors the production directory so
    ``build_classifier_registry_payload`` finds two
    ``is_available_locally`` variants. ``model_config.yaml`` files are
    intentionally minimal — pin doesn't read them.
    """
    cls_dir = tmp_path / "classifier"
    cls_dir.mkdir()

    variants = ["20260427_143835", "20260427_104449"]
    for mid in variants:
        (cls_dir / f"{mid}_best.onnx").write_bytes(b"fake onnx")
        (cls_dir / f"{mid}_classes.txt").write_text("Parus_major\nCyanistes_caeruleus\n")
        # Minimal YAML — pin only needs the variant to be locally available.
        (cls_dir / f"{mid}_model_config.yaml").write_text(
            "detection:\n  confidence_threshold: 0.98\n  genus_fallback_threshold: 0.55\n"
        )
        (cls_dir / f"{mid}_metrics.json").write_text(
            json.dumps({"model_id": mid, "aggregate": {"top1_accuracy": 0.94}})
        )

    (cls_dir / "latest_models.json").write_text(
        json.dumps(
            {
                "latest": "20260427_143835",
                "project_name": "WatchMyBirds",
                "weights_path": "classifier/20260427_143835_best.onnx",
                "classes_path": "classifier/20260427_143835_classes.txt",
                "pinned_models": {
                    mid: {
                        "weights_path": f"classifier/{mid}_best.onnx",
                        "classes_path": f"classifier/{mid}_classes.txt",
                    }
                    for mid in variants
                },
            }
        )
    )

    import config as config_mod

    monkeypatch.setitem(config_mod.get_config(), "MODEL_BASE_PATH", str(tmp_path))

    # Clear pin env vars so the test's pin click is the only signal.
    for key in (
        "WMB_PINNED_MODEL_ID",
        "WMB_PINNED_MODEL_ID_OBJECT_DETECTION",
        "WMB_PINNED_MODEL_ID_CLASSIFIER",
    ):
        monkeypatch.delenv(key, raising=False)

    return cls_dir


@pytest.fixture
def app(model_dir):
    """Flask app with API v1 wired to a fake DetectionManager.

    The fake DM exposes a ``classifier`` with the attributes the pin
    endpoint clears (``_initialized``, ``ort_session``, etc.) so we
    can verify the live-reload trigger fired.
    """
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    fake_classifier = MagicMock()
    fake_classifier._initialized = True
    fake_classifier.ort_session = MagicMock()
    fake_classifier.model_path = str(model_dir / "20260427_143835_best.onnx")
    fake_classifier.class_path = str(model_dir / "20260427_143835_classes.txt")
    fake_classifier.model_id = "20260427_143835"
    # Pre-pin decision_config — must get cleared so the new YAML on
    # disk actually takes effect at next lazy-init.
    fake_classifier.decision_config = MagicMock()
    fake_classifier.classes = ["Parus_major", "Cyanistes_caeruleus"]

    mock_dm = MagicMock()
    mock_dm.classifier = fake_classifier
    mock_dm.classifier_model_id = "20260427_143835"

    from web.blueprints.api_v1 import api_v1 as api_v1_bp
    from web.blueprints.auth import auth_bp

    app.register_blueprint(auth_bp)
    api_v1_bp.detection_manager = mock_dm
    app.register_blueprint(api_v1_bp)

    app.config["_mock_dm"] = mock_dm
    app.config["_fake_classifier"] = fake_classifier
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


# ---------------------------------------------------------------------------
# Companion-refresh contract — the regression fix
# ---------------------------------------------------------------------------


def test_pin_force_refreshes_companion_files_from_hf(client, model_dir):
    """A pin click MUST trigger _fetch_companion_files with force_refresh=True.

    Without this, model-side updates to the per_species_thresholds in
    model_config.yaml never reach the Pi — the local cache wins and the
    classifier keeps its old gallery/review thresholds until a manual
    intervention (delete-and-restart). Mirrors the detector pin path.
    """
    with patch(
        "utils.model_downloader._fetch_companion_files"
    ) as mock_fetch:
        resp = client.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": "20260427_104449"},
        )

    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["model_id"] == "20260427_104449"

    # The contract: called exactly once with force_refresh=True,
    # against the classifier HF URL (not the detector's), for the
    # model_id we just pinned.
    assert mock_fetch.call_count == 1
    call_args = mock_fetch.call_args
    # Positional args: (base_url, model_dir, model_id)
    base_url_arg = call_args.args[0]
    model_dir_arg = call_args.args[1]
    model_id_arg = call_args.args[2]
    assert "classifier" in base_url_arg.lower()
    assert "huggingface.co" in base_url_arg
    assert str(model_dir) in model_dir_arg or model_dir_arg.endswith("classifier")
    assert model_id_arg == "20260427_104449"
    # And the all-important kwarg
    assert call_args.kwargs.get("force_refresh") is True


def test_pin_hf_refresh_failure_is_fail_soft(client, model_dir):
    """An HF outage during refresh must NOT fail the pin click.

    Operators expect the pin to flip latest_models.json + clear the
    classifier even when HF is unreachable. The refresh is a
    best-effort optimisation, not a hard prerequisite.
    """
    with patch(
        "utils.model_downloader._fetch_companion_files",
        side_effect=ConnectionError("HF unreachable"),
    ):
        resp = client.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": "20260427_104449"},
        )

    # Pin still succeeds end-to-end despite the refresh failure.
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["reload_triggered"] is True

    # latest_models.json reflects the new variant even when HF was down.
    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["latest"] == "20260427_104449"


def test_pin_clears_decision_config_to_force_yaml_reload(client, app):
    """The fix's second half: decision_config MUST be cleared on pin.

    Even if the YAML on disk got refreshed, an in-memory cached
    decision_config from the previous boot would keep the classifier
    routing against stale thresholds until the next process restart.
    """
    fake_classifier = app.config["_fake_classifier"]
    # Sanity-check the fixture set up a non-None decision_config to clear
    assert fake_classifier.decision_config is not None

    with patch("utils.model_downloader._fetch_companion_files"):
        resp = client.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": "20260427_104449"},
        )

    assert resp.status_code == 200, resp.data
    # decision_config is cleared so load_cls_decision_config re-reads
    # the freshly-refreshed YAML on next lazy-init.
    assert fake_classifier.decision_config is None
    # And the rest of the existing clear-for-reload contract holds.
    assert fake_classifier._initialized is False
    assert fake_classifier.ort_session is None
    assert fake_classifier.model_id == ""


# ---------------------------------------------------------------------------
# Existing pin contract — guard against regression in the path we modified
# ---------------------------------------------------------------------------


def test_pin_valid_variant_flips_latest_models_and_triggers_reload(
    client, model_dir
):
    with patch("utils.model_downloader._fetch_companion_files"):
        resp = client.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": "20260427_104449"},
        )

    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["model_id"] == "20260427_104449"
    assert data["reload_triggered"] is True

    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["latest"] == "20260427_104449"
    # pinned_models block is preserved (round-trip-safe)
    assert "20260427_143835" in updated.get("pinned_models", {})


def test_pin_rejects_unknown_variant(client, model_dir):
    with patch("utils.model_downloader._fetch_companion_files") as mock_fetch:
        resp = client.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": "not_a_real_classifier"},
        )

    assert resp.status_code == 400
    # latest_models.json untouched
    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["latest"] == "20260427_143835"
    # And critically: NO fetch attempt for unknown variants — wasted
    # HF bandwidth + log noise if we did.
    mock_fetch.assert_not_called()


def test_pin_rejects_empty_model_id(client):
    with patch("utils.model_downloader._fetch_companion_files") as mock_fetch:
        resp = client.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": ""},
        )

    assert resp.status_code == 400
    mock_fetch.assert_not_called()


def test_pin_requires_auth(app):
    with app.test_client() as anon:
        resp = anon.post(
            "/api/v1/models/classifier/pin",
            json={"model_id": "20260427_104449"},
        )
    assert resp.status_code in (302, 401, 403)
