"""Tests for /api/v1/models/detector GET + pin POST endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from flask import Flask


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A fake object_detection cache dir with a latest_models.json setup.

    Two variants listed under pinned_models:
      - 20260417_yolox_tiny_locator_ep120  (active by latest + local files present)
      - 20260417_yolox_s_locator_ep120     (installed, pinnable alternative)
    Plus one unavailable variant to exercise the is_available_locally flag:
      - 20260417_yolox_phantom_ep120       (listed but files missing)
    """
    od_dir = tmp_path / "object_detection"
    od_dir.mkdir()

    # Touch weight + label files for tiny and s, PLUS their _model_config.yaml
    # so the pin endpoint can regenerate model_metadata.json with the right
    # thresholds per variant (Tiny: conf=0.15, S: conf=0.30).
    yaml_thresholds = {
        "20260417_yolox_tiny_locator_ep120": (0.15, "tiny"),
        "20260417_yolox_s_locator_ep120":    (0.30, "s"),
    }
    for mid, (conf, variant) in yaml_thresholds.items():
        (od_dir / f"{mid}_best.onnx").write_bytes(b"fake onnx")
        (od_dir / f"{mid}_labels.json").write_text(
            json.dumps({"0": "bird", "1": "squirrel", "2": "cat",
                        "3": "marten_mustelid", "4": "hedgehog"})
        )
        (od_dir / f"{mid}_model_config.yaml").write_text(
            "detection:\n"
            f"  confidence_threshold: {conf}\n"
            "  nms_iou_threshold: 0.5\n"
            "  input_size:\n"
            "  - 640\n"
            "  - 640\n"
            "  input_format: BGR\n"
            "  input_normalize: false\n"
            "  output_format: yolox_raw\n"
            f"  architecture: yolox_{variant}_locator_5cls\n"
            "meta:\n"
            "  num_classes: 5\n"
        )

    (od_dir / "latest_models.json").write_text(json.dumps({
        "latest": "20260417_yolox_tiny_locator_ep120",
        "project_name": "WatchMyBirds",
        "weights_path": "object_detection/20260417_yolox_tiny_locator_ep120_best.onnx",
        "labels_path":  "object_detection/20260417_yolox_tiny_locator_ep120_labels.json",
        "pinned_models": {
            "20260417_yolox_tiny_locator_ep120": {
                "weights_path": "object_detection/20260417_yolox_tiny_locator_ep120_best.onnx",
                "labels_path":  "object_detection/20260417_yolox_tiny_locator_ep120_labels.json",
            },
            "20260417_yolox_s_locator_ep120": {
                "weights_path": "object_detection/20260417_yolox_s_locator_ep120_best.onnx",
                "labels_path":  "object_detection/20260417_yolox_s_locator_ep120_labels.json",
            },
            "20260417_yolox_phantom_ep120": {
                "weights_path": "object_detection/20260417_yolox_phantom_ep120_best.onnx",
                "labels_path":  "object_detection/20260417_yolox_phantom_ep120_labels.json",
            },
        },
    }))

    (od_dir / "model_metadata.json").write_text(json.dumps({
        "framework": "yolox",
        "variant": "tiny",
        "input_size": [640, 640],
        "output_format": "yolox_raw",
        "num_classes": 5,
        "inference_thresholds": {"confidence": 0.15, "iou_nms": 0.5},
        "metrics": {"bird_recall": 0.993, "bird_precision": 0.9914,
                    "anim_to_bird": 0.1429, "empty_fp": 0.0, "f1": 0.9922},
    }))

    # Point MODEL_BASE_PATH to tmp_path so _model_dir() resolves to od_dir.
    import config as config_mod
    monkeypatch.setitem(config_mod.get_config(), "MODEL_BASE_PATH", str(tmp_path))

    # Clear any pin env vars that might leak from the host shell.
    for key in (
        "WMB_PINNED_MODEL_ID",
        "WMB_PINNED_MODEL_ID_OBJECT_DETECTION",
        "WMB_PINNED_MODEL_ID_CLASSIFIER",
    ):
        monkeypatch.delenv(key, raising=False)

    return od_dir


@pytest.fixture
def app(model_dir):
    """Flask app with API v1 wired to a fake DetectionManager.

    The fake DM exposes a ``detection_service._detector.model`` chain with
    the fields the registry service reads (model_id, output_format, …).
    """
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    fake_detector_model = MagicMock()
    fake_detector_model.model_id = "20260417_yolox_tiny_locator_ep120"
    fake_detector_model.model_path = str(
        model_dir / "20260417_yolox_tiny_locator_ep120_best.onnx"
    )
    fake_detector_model.output_format = "yolox_raw"
    fake_detector_model.input_size = (640, 640)
    fake_detector_model.class_names = {
        "0": "bird", "1": "squirrel", "2": "cat",
        "3": "marten_mustelid", "4": "hedgehog",
    }
    fake_detector_model.conf_threshold_default = 0.15
    fake_detector_model.iou_threshold_default = 0.5

    detector_wrapper = MagicMock()
    detector_wrapper.model = fake_detector_model

    detection_service = MagicMock()
    detection_service._detector = detector_wrapper
    detection_service._initialized = True
    detection_service._model_id = fake_detector_model.model_id

    mock_dm = MagicMock()
    mock_dm.paused = False
    mock_dm.detection_service = detection_service
    mock_dm.detector_model_id = fake_detector_model.model_id

    from web.blueprints.auth import auth_bp
    from web.blueprints.api_v1 import api_v1 as api_v1_bp

    app.register_blueprint(auth_bp)
    api_v1_bp.detection_manager = mock_dm
    app.register_blueprint(api_v1_bp)

    app.config["_mock_dm"] = mock_dm  # expose for tests
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


# ---------------------------------------------------------------------------
# GET /api/v1/models/detector
# ---------------------------------------------------------------------------


def test_get_returns_active_and_variants(client):
    resp = client.get("/api/v1/models/detector")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()

    assert data["active"]["id"] == "20260417_yolox_tiny_locator_ep120"
    # No env-var pin in this test -> source is the latest_models.json pointer
    assert data["active"]["source"] == "latest_models"
    assert data["active"]["hf_latest_id"] == "20260417_yolox_tiny_locator_ep120"

    variant_ids = sorted(v["id"] for v in data["variants"])
    assert variant_ids == [
        "20260417_yolox_phantom_ep120",
        "20260417_yolox_s_locator_ep120",
        "20260417_yolox_tiny_locator_ep120",
    ]

    by_id = {v["id"]: v for v in data["variants"]}
    assert by_id["20260417_yolox_tiny_locator_ep120"]["is_active"] is True
    assert by_id["20260417_yolox_tiny_locator_ep120"]["is_hf_latest"] is True
    assert by_id["20260417_yolox_tiny_locator_ep120"]["is_available_locally"] is True

    assert by_id["20260417_yolox_s_locator_ep120"]["is_active"] is False
    assert by_id["20260417_yolox_s_locator_ep120"]["is_available_locally"] is True

    assert by_id["20260417_yolox_phantom_ep120"]["is_available_locally"] is False


def test_get_includes_runtime_from_live_detector(client):
    resp = client.get("/api/v1/models/detector")
    data = resp.get_json()
    rt = data["runtime"]
    assert rt["model_id"] == "20260417_yolox_tiny_locator_ep120"
    assert rt["output_format"] == "yolox_raw"
    assert rt["num_classes"] == 5
    assert "bird" in rt["class_names"]
    assert rt["conf_threshold_default"] == 0.15
    assert rt["iou_threshold_default"] == 0.5


def test_get_metadata_block_present(client):
    resp = client.get("/api/v1/models/detector")
    data = resp.get_json()
    md = data["metadata"]
    assert md["framework"] == "yolox"
    assert md["metrics"]["bird_recall"] == pytest.approx(0.993)


def test_get_requires_auth(app):
    # Anonymous client (no session) should get redirected/blocked.
    with app.test_client() as anon:
        resp = anon.get("/api/v1/models/detector")
    # login_required redirects to the login page; we just want to confirm
    # the route is gated.
    assert resp.status_code in (302, 401, 403), resp.status_code


# ---------------------------------------------------------------------------
# POST /api/v1/models/detector/pin
# ---------------------------------------------------------------------------


def test_pin_valid_variant_flips_latest_models_and_triggers_reload(client, app, model_dir):
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_s_locator_ep120"},
    )
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["model_id"] == "20260417_yolox_s_locator_ep120"
    assert data["effective_id"] == "20260417_yolox_s_locator_ep120"
    assert data["effective_source"] == "latest_models"
    assert data["env_pin_overrides"] is False
    assert data["reload_triggered"] is True
    assert data["latest_models_path"].endswith("latest_models.json")
    assert data["metadata_path"].endswith("model_metadata.json")

    # latest_models.json now points at the S variant + has the matching paths
    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["latest"] == "20260417_yolox_s_locator_ep120"
    assert updated["weights_path"].endswith("20260417_yolox_s_locator_ep120_best.onnx")
    assert updated["labels_path"].endswith("20260417_yolox_s_locator_ep120_labels.json")
    # pinned_models block is preserved so a swap back is always possible
    assert "20260417_yolox_tiny_locator_ep120" in updated.get("pinned_models", {})

    # DetectionService was cleared to force a lazy reload
    ds = app.config["_mock_dm"].detection_service
    assert ds._detector is None
    assert ds._initialized is False
    assert ds._model_id == ""

    # No stale active_pin.json from the old R0 design
    assert not (model_dir / "active_pin.json").exists()


def test_pin_regenerates_metadata_with_new_variant_thresholds(client, model_dir):
    """The key regression test: switching from Tiny to S must update
    model_metadata.json so the next detector reload picks up S's
    confidence threshold (0.30) instead of inheriting Tiny's (0.15)."""

    # Seed: metadata is Tiny's (conf=0.15)
    (model_dir / "model_metadata.json").write_text(json.dumps({
        "variant": "tiny",
        "inference_thresholds": {"confidence": 0.15, "iou_nms": 0.5},
    }))

    # Switch to S
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_s_locator_ep120"},
    )
    assert resp.status_code == 200, resp.data

    # Metadata is now S's (conf=0.30)
    md = json.loads((model_dir / "model_metadata.json").read_text())
    assert md["variant"] == "s"
    assert md["inference_thresholds"]["confidence"] == 0.30
    assert md["inference_thresholds"]["iou_nms"] == 0.5

    # Switch back to Tiny
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_tiny_locator_ep120"},
    )
    assert resp.status_code == 200, resp.data

    # Metadata flipped back
    md = json.loads((model_dir / "model_metadata.json").read_text())
    assert md["variant"] == "tiny"
    assert md["inference_thresholds"]["confidence"] == 0.15


def test_pin_rejects_unknown_variant(client, model_dir):
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "not_a_real_model_id"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "not a known locally-available variant" in data["message"]
    # latest_models.json left untouched
    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["latest"] == "20260417_yolox_tiny_locator_ep120"


def test_pin_rejects_unavailable_variant(client, model_dir):
    """A pinned_models entry whose weights/labels are missing on disk must not be pinnable."""
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_phantom_ep120"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"


def test_pin_rejects_empty_model_id(client, model_dir):
    """Empty model_id is invalid (no 'unpin' concept anymore — you always pick a variant)."""
    resp = client.post("/api/v1/models/detector/pin", json={"model_id": ""})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "required" in data["message"].lower() or "non-empty" in data["message"].lower()


def test_pin_requires_auth(app):
    with app.test_client() as anon:
        resp = anon.post(
            "/api/v1/models/detector/pin",
            json={"model_id": "20260417_yolox_s_locator_ep120"},
        )
    assert resp.status_code in (302, 401, 403)


def test_pin_env_var_overrides_written_latest(client, model_dir, monkeypatch):
    """When a systemd env-pin is set, pin still writes latest_models.json
    (so startup without the env pin picks up the UI choice) but reports
    env_pin_overrides=True so the UI can show a warning."""
    monkeypatch.setenv(
        "WMB_PINNED_MODEL_ID_OBJECT_DETECTION", "20260417_yolox_tiny_locator_ep120"
    )
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_s_locator_ep120"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["model_id"] == "20260417_yolox_s_locator_ep120"
    # Effective loaded model is still the env-pin one
    assert data["effective_id"] == "20260417_yolox_tiny_locator_ep120"
    assert data["effective_source"] == "env_var_pin"
    assert data["env_pin_overrides"] is True

    # But the on-disk pointer IS updated, so the next env-pin-free startup
    # picks up the UI choice.
    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["latest"] == "20260417_yolox_s_locator_ep120"


# ---------------------------------------------------------------------------
# POST /api/v1/models/detector/precision
# ---------------------------------------------------------------------------


def _add_int8_qdq_to_fixture(model_dir, variant_id: str) -> None:
    """Extend the fixture's latest_models.json so *variant_id* declares
    int8 QDQ artefacts, and touch the corresponding ONNX file on disk so
    the primary QDQ candidate counts as 'available'."""
    payload = json.loads((model_dir / "latest_models.json").read_text())
    primary_rel = (
        f"object_detection/{variant_id}_best_int8_qdq.onnx"
    )
    fallbacks_rel = [
        primary_rel,
        f"object_detection/{variant_id}_best_int8_qdq_u8a.onnx",
    ]

    entry = payload.setdefault("pinned_models", {}).setdefault(variant_id, {})
    entry["weights_int8_qdq_path"] = primary_rel
    entry["weights_int8_qdq_fallback_paths"] = fallbacks_rel

    # Also mirror onto top-level so the loader path sees it for the default.
    if payload.get("latest") == variant_id:
        payload["weights_int8_qdq_path"] = primary_rel
        payload["weights_int8_qdq_fallback_paths"] = fallbacks_rel

    (model_dir / "latest_models.json").write_text(json.dumps(payload))
    # Drop a fake primary int8 ONNX so the "int8_qdq_available" flag is True.
    (model_dir / f"{variant_id}_best_int8_qdq.onnx").write_bytes(b"fake int8 qdq")


def test_get_reports_precision_availability_per_variant(client, model_dir):
    _add_int8_qdq_to_fixture(model_dir, "20260417_yolox_tiny_locator_ep120")

    resp = client.get("/api/v1/models/detector")
    assert resp.status_code == 200
    by_id = {v["id"]: v for v in resp.get_json()["variants"]}

    tiny = by_id["20260417_yolox_tiny_locator_ep120"]
    assert tiny["int8_qdq_available"] is True
    # Default: no active_precision field yet in the registry, so fp32.
    assert tiny["active_precision"] == "fp32"

    s = by_id["20260417_yolox_s_locator_ep120"]
    assert s["int8_qdq_available"] is False
    assert s["active_precision"] == "fp32"


def test_precision_switch_writes_active_precision_and_triggers_reload(
    client, app, model_dir
):
    _add_int8_qdq_to_fixture(model_dir, "20260417_yolox_tiny_locator_ep120")

    resp = client.post(
        "/api/v1/models/detector/precision",
        json={
            "model_id": "20260417_yolox_tiny_locator_ep120",
            "precision": "int8_qdq",
        },
    )
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["precision"] == "int8_qdq"
    assert data["model_id"] == "20260417_yolox_tiny_locator_ep120"
    assert data["reload_triggered"] is True

    # Choice persisted on disk so a reload picks it up.
    updated = json.loads((model_dir / "latest_models.json").read_text())
    assert updated["active_precision"] == "int8_qdq"  # stamped on top-level
    # for the current default
    pinned = updated.get("pinned_models", {})
    assert (
        pinned["20260417_yolox_tiny_locator_ep120"]["active_precision"]
        == "int8_qdq"
    )
    # Other variants are untouched.
    assert (
        pinned["20260417_yolox_s_locator_ep120"].get("active_precision")
        is None
    )

    # DetectionService was cleared to force a lazy reload on next cycle.
    ds = app.config["_mock_dm"].detection_service
    assert ds._detector is None
    assert ds._initialized is False

    # GET reports the switched precision.
    get_resp = client.get("/api/v1/models/detector")
    tiny = {
        v["id"]: v for v in get_resp.get_json()["variants"]
    }["20260417_yolox_tiny_locator_ep120"]
    assert tiny["active_precision"] == "int8_qdq"


def test_precision_switch_survives_variant_swap(client, model_dir):
    """Switching Tiny to int8_qdq, then switching the active variant to
    S, then back to Tiny, must preserve Tiny's precision choice. This
    is the regression the _apply_pin key-whitelist extension prevents."""
    _add_int8_qdq_to_fixture(model_dir, "20260417_yolox_tiny_locator_ep120")

    # 1. Tiny -> int8_qdq
    resp = client.post(
        "/api/v1/models/detector/precision",
        json={
            "model_id": "20260417_yolox_tiny_locator_ep120",
            "precision": "int8_qdq",
        },
    )
    assert resp.status_code == 200

    # 2. Switch the active variant to S (unrelated, should not drop the flag)
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_s_locator_ep120"},
    )
    assert resp.status_code == 200

    # 3. Switch back to Tiny
    resp = client.post(
        "/api/v1/models/detector/pin",
        json={"model_id": "20260417_yolox_tiny_locator_ep120"},
    )
    assert resp.status_code == 200

    # Tiny's precision is still int8_qdq.
    updated = json.loads((model_dir / "latest_models.json").read_text())
    pinned = updated.get("pinned_models", {})
    assert (
        pinned["20260417_yolox_tiny_locator_ep120"]["active_precision"]
        == "int8_qdq"
    )
    # And the top-level copy got applied on variant switch back to Tiny
    # (via _apply_pin's key propagation).
    assert updated["active_precision"] == "int8_qdq"


def test_precision_switch_rejects_unknown_precision(client):
    resp = client.post(
        "/api/v1/models/detector/precision",
        json={
            "model_id": "20260417_yolox_tiny_locator_ep120",
            "precision": "fp16",
        },
    )
    assert resp.status_code == 400
    assert "precision" in resp.get_json()["message"].lower()


def test_precision_switch_rejects_unknown_model_id(client):
    resp = client.post(
        "/api/v1/models/detector/precision",
        json={"model_id": "not_a_real_id", "precision": "int8_qdq"},
    )
    assert resp.status_code == 400


def test_precision_switch_rejects_missing_fields(client):
    resp = client.post(
        "/api/v1/models/detector/precision",
        json={"model_id": "20260417_yolox_tiny_locator_ep120"},
    )
    assert resp.status_code == 400
    resp = client.post(
        "/api/v1/models/detector/precision",
        json={"precision": "int8_qdq"},
    )
    assert resp.status_code == 400


def test_precision_switch_requires_auth(app):
    with app.test_client() as anon:
        resp = anon.post(
            "/api/v1/models/detector/precision",
            json={
                "model_id": "20260417_yolox_tiny_locator_ep120",
                "precision": "int8_qdq",
            },
        )
    assert resp.status_code in (302, 401, 403)


# ---------------------------------------------------------------------------
# resolve_active_precision_artefacts  (utility-level)
# ---------------------------------------------------------------------------


def test_resolve_precision_returns_none_when_no_registry(tmp_path):
    from utils.model_downloader import resolve_active_precision_artefacts
    assert resolve_active_precision_artefacts(str(tmp_path)) is None


def test_resolve_precision_returns_fp32_by_default(model_dir):
    from utils.model_downloader import resolve_active_precision_artefacts
    info = resolve_active_precision_artefacts(str(model_dir))
    assert info is not None
    assert info["requested_precision"] == "fp32"
    assert info["load_candidates"] == []
    assert info["fp32_fallback_path"].endswith("_best.onnx")


def test_hf_refresh_preserves_active_precision_stamp(model_dir, monkeypatch):
    """Regression: the preservation guard in fetch_latest_json must keep
    the locally-written ``active_precision`` stamp even when HF responds
    with a remote ``latest_models.json`` that does not carry the flag.

    Without this protection, the next detector reload would refresh the
    cache, silently drop the UI choice, and flip the loaded precision
    back to fp32 — which is exactly what the 2026-04-18 smoke test hit
    live on the Pi."""
    from utils.model_downloader import (
        fetch_latest_json,
        set_active_precision,
    )

    _add_int8_qdq_to_fixture(
        model_dir, "20260417_yolox_tiny_locator_ep120"
    )
    set_active_precision(
        str(model_dir),
        "20260417_yolox_tiny_locator_ep120",
        "int8_qdq",
    )
    # Verify the stamp exists pre-refresh.
    before = json.loads((model_dir / "latest_models.json").read_text())
    assert before["active_precision"] == "int8_qdq"
    assert (
        before["pinned_models"]["20260417_yolox_tiny_locator_ep120"][
            "active_precision"
        ]
        == "int8_qdq"
    )

    # Fake HF's "latest" response — same latest id, same paths, but no
    # precision key (mirrors what pipeline-dev's HF file looks like).
    remote_payload = {
        "latest": "20260417_yolox_tiny_locator_ep120",
        "project_name": "WatchMyBirds",
        "weights_path": (
            "object_detection/"
            "20260417_yolox_tiny_locator_ep120_best.onnx"
        ),
        "labels_path": (
            "object_detection/"
            "20260417_yolox_tiny_locator_ep120_labels.json"
        ),
        "pinned_models": {
            "20260417_yolox_tiny_locator_ep120": {
                "weights_path": (
                    "object_detection/"
                    "20260417_yolox_tiny_locator_ep120_best.onnx"
                ),
                "labels_path": (
                    "object_detection/"
                    "20260417_yolox_tiny_locator_ep120_labels.json"
                ),
            },
            "20260417_yolox_s_locator_ep120": {
                "weights_path": (
                    "object_detection/"
                    "20260417_yolox_s_locator_ep120_best.onnx"
                ),
                "labels_path": (
                    "object_detection/"
                    "20260417_yolox_s_locator_ep120_labels.json"
                ),
            },
        },
    }

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return remote_payload

    def _fake_get(url, timeout=None):  # noqa: ARG001
        return _FakeResp()

    import utils.model_downloader as md
    monkeypatch.setattr(md.requests, "get", _fake_get)

    result = fetch_latest_json(
        "https://example.test/object_detection", str(model_dir)
    )
    # Returned payload MUST carry the local precision stamp.
    assert result.get("active_precision") == "int8_qdq"
    assert (
        result["pinned_models"]["20260417_yolox_tiny_locator_ep120"][
            "active_precision"
        ]
        == "int8_qdq"
    )
    # On-disk file MUST carry it too (next reload loads the right weights).
    after = json.loads((model_dir / "latest_models.json").read_text())
    assert after["active_precision"] == "int8_qdq"
    assert (
        after["pinned_models"]["20260417_yolox_tiny_locator_ep120"][
            "active_precision"
        ]
        == "int8_qdq"
    )


def test_resolve_precision_returns_ordered_qdq_candidates(model_dir):
    from utils.model_downloader import resolve_active_precision_artefacts
    _add_int8_qdq_to_fixture(model_dir, "20260417_yolox_tiny_locator_ep120")
    # Simulate the UI switch by stamping the top-level precision flag.
    data = json.loads((model_dir / "latest_models.json").read_text())
    data["active_precision"] = "int8_qdq"
    (model_dir / "latest_models.json").write_text(json.dumps(data))

    info = resolve_active_precision_artefacts(str(model_dir))
    assert info is not None
    assert info["requested_precision"] == "int8_qdq"
    # Primary QDQ first, then the fallback.
    assert len(info["load_candidates"]) == 2
    assert info["load_candidates"][0].endswith("_best_int8_qdq.onnx")
    assert info["load_candidates"][1].endswith("_best_int8_qdq_u8a.onnx")


# ---------------------------------------------------------------------------
# POST /api/v1/models/detector/install
# ---------------------------------------------------------------------------


def test_install_downloads_missing_variant_from_hf(client, model_dir, monkeypatch):
    """Install fetches weights + labels + _model_config.yaml from HF."""
    calls: list[tuple[str, str]] = []

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        calls.append((url, dest))
        import os as _os
        _os.makedirs(_os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"fetched-by-test")
        return True

    import utils.model_downloader as md
    monkeypatch.setattr(md, "_download_file", fake_download)

    resp = client.post(
        "/api/v1/models/detector/install",
        json={"model_id": "20260417_yolox_phantom_ep120"},
    )
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["model_id"] == "20260417_yolox_phantom_ep120"
    assert data["already_installed"] is False
    # Four downloads: weights + labels + _model_config.yaml + _metrics.json.
    # YAML enables correct per-variant threshold regeneration; metrics
    # lets the AI panel show accurate recall/precision numbers.
    assert len(calls) == 4
    fetched_dests = [c[1] for c in calls]
    assert any(d.endswith("_best.onnx") for d in fetched_dests)
    assert any(d.endswith("_labels.json") for d in fetched_dests)
    assert any(d.endswith("_model_config.yaml") for d in fetched_dests)
    assert any(d.endswith("_metrics.json") for d in fetched_dests)
    for url, dest in calls:
        assert url.startswith(
            "https://huggingface.co/arminfabritzek/WatchMyBirds-Models"
        )
        assert str(model_dir) in dest
    # Regression guard: the YAML URL must be a direct sibling of the
    # weights URL. An earlier bug concatenated 'object_detection/' twice
    # because HF_BASE_URL already ends in that subfolder (observed
    # 2026-04-17 22:53 on RPi: 404 for .../object_detection/object_detection/...yaml).
    yaml_url = next(u for u, _ in calls if u.endswith(".yaml"))
    assert yaml_url.count("/object_detection/") == 1, (
        f"YAML URL has duplicated subfolder: {yaml_url}"
    )
    # Files actually landed on disk.
    assert (model_dir / "20260417_yolox_phantom_ep120_best.onnx").exists()
    assert (model_dir / "20260417_yolox_phantom_ep120_labels.json").exists()
    assert (model_dir / "20260417_yolox_phantom_ep120_model_config.yaml").exists()
    assert (model_dir / "20260417_yolox_phantom_ep120_metrics.json").exists()
    assert data["model_config_path"] is not None
    assert data["metrics_path"] is not None


def test_install_tolerates_missing_yaml_on_hf(client, model_dir, monkeypatch):
    """Older releases may not ship a _model_config.yaml. Install must
    succeed anyway — the pin endpoint already logs a warning and falls
    back to hard-coded defaults in that case."""
    import utils.model_downloader as md

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        if url.endswith(".yaml"):
            return False  # simulate 404 on HF
        import os as _os
        _os.makedirs(_os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"fetched-by-test")
        return True

    monkeypatch.setattr(md, "_download_file", fake_download)

    resp = client.post(
        "/api/v1/models/detector/install",
        json={"model_id": "20260417_yolox_phantom_ep120"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["model_config_path"] is None


def test_install_is_idempotent_for_already_installed_variant(client, monkeypatch):
    """Installing a variant whose files are already local returns success
    without calling the downloader — UI can safely retry."""
    import utils.model_downloader as md
    monkeypatch.setattr(
        md, "_download_file", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not download"))
    )
    resp = client.post(
        "/api/v1/models/detector/install",
        json={"model_id": "20260417_yolox_s_locator_ep120"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["already_installed"] is True


def test_install_rejects_unknown_model_id(client):
    """Arbitrary strings from the request body must not be fetched —
    whitelist gate is the registry payload."""
    resp = client.post(
        "/api/v1/models/detector/install",
        json={"model_id": "../../../etc/passwd"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "registry" in data["message"].lower()


def test_install_rejects_empty_model_id(client):
    resp = client.post("/api/v1/models/detector/install", json={"model_id": ""})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"


def test_install_surfaces_download_failure(client, monkeypatch):
    """When the HF download fails, the endpoint reports 502 + the URL."""
    import utils.model_downloader as md
    monkeypatch.setattr(md, "_download_file", lambda *a, **kw: False)

    resp = client.post(
        "/api/v1/models/detector/install",
        json={"model_id": "20260417_yolox_phantom_ep120"},
    )
    assert resp.status_code == 502
    data = resp.get_json()
    assert data["status"] == "error"
    assert "failed to download" in data["message"].lower()


def test_install_requires_auth(app):
    with app.test_client() as anon:
        resp = anon.post(
            "/api/v1/models/detector/install",
            json={"model_id": "20260417_yolox_phantom_ep120"},
        )
    assert resp.status_code in (302, 401, 403)
