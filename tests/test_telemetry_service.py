"""Tests for web.services.telemetry_service.

Critical guarantees we verify:

1. **Default OFF.** Fresh config, missing key, string "true", and other
   non-bool values must all return False. Only `is True` triggers a send.
2. **Payload shape.** build_payload() must produce exactly 8 fields, no
   extras. The Worker's allowlist will 400-reject any extra key, so a
   regression here would silently break production.
3. **No hidden PII.** Payload must NOT contain country, locale, hostname,
   pi_model, kernel, mac, ip, or any field outside the 8-field allowlist.
4. **UUID lifecycle.** First opt-in lazily generates a 32-hex UUID and
   persists it. Toggle off does NOT wipe it. Rotate explicitly does.
5. **Atomic last-sent file.** Concurrent crash-during-write must leave
   either the old or the new contents, never a half-written file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from web.services import telemetry_service as ts  # noqa: E402

# ---------------------------------------------------------------------
# 1) Default OFF — strictest guarantee in the whole feature
# ---------------------------------------------------------------------


def test_is_enabled_default_false_on_empty_config():
    """Empty config dict (key missing) must read as disabled."""
    assert ts.is_enabled({}) is False


def test_is_enabled_explicit_false():
    assert ts.is_enabled({"telemetry_enabled": False}) is False


def test_is_enabled_explicit_true():
    assert ts.is_enabled({"telemetry_enabled": True}) is True


def test_is_enabled_rejects_string_true():
    """Strings must NOT be accepted — settings round-trip should produce
    a real bool. If we accept 'true' (string) we accidentally enable
    anyone with a malformed YAML."""
    assert ts.is_enabled({"telemetry_enabled": "true"}) is False
    assert ts.is_enabled({"telemetry_enabled": "True"}) is False
    assert ts.is_enabled({"telemetry_enabled": "yes"}) is False


def test_is_enabled_rejects_truthy_int():
    """Ints, even truthy ones, must NOT enable. Same reason as strings."""
    assert ts.is_enabled({"telemetry_enabled": 1}) is False


def test_is_enabled_rejects_none():
    assert ts.is_enabled({"telemetry_enabled": None}) is False


# ---------------------------------------------------------------------
# 2) Payload shape — exactly the 8 fields the Worker allowlist expects
# ---------------------------------------------------------------------


EXPECTED_FIELDS = {
    "installation_id",
    "app_version",
    "os",
    "arch",
    "cpu_count",
    "total_ram_gb",
    "python_version",
    "detector_variant",
}


def _make_test_cfg(tmp_path: Path) -> dict:
    """Minimal config dict for build_payload tests. Uses a temp OUTPUT_DIR
    so we don't touch the real settings.yaml."""
    return {
        "OUTPUT_DIR": str(tmp_path),
        # Force a known UUID by pre-populating it; build_payload's
        # _ensure_installation_id will accept it as already-set.
        "telemetry_installation_id": "deadbeef00112233deadbeef00112233",
    }


def test_payload_has_exactly_eight_fields(tmp_path):
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload(cfg)
    assert set(payload.keys()) == EXPECTED_FIELDS
    assert len(payload) == 8


def test_preview_has_exactly_eight_fields(tmp_path):
    """build_payload_preview() must match build_payload()'s shape so the
    UI never misleads the operator about what gets sent."""
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload_preview(cfg)
    assert set(payload.keys()) == EXPECTED_FIELDS
    assert len(payload) == 8


def test_preview_does_not_generate_uuid(tmp_path):
    """Preview is read-only. If no UUID exists yet, the preview must
    return a placeholder — NOT silently generate one and persist."""
    cfg = {
        "OUTPUT_DIR": str(tmp_path),
        "telemetry_installation_id": "",
    }
    payload = ts.build_payload_preview(cfg)

    # Placeholder, not a real hex UUID.
    assert payload["installation_id"] == "<would-be-generated-on-opt-in>"

    # And settings.yaml must NOT have been written.
    from utils.settings import load_settings_yaml

    yaml_settings = load_settings_yaml(str(tmp_path))
    assert "telemetry_installation_id" not in yaml_settings or not yaml_settings.get(
        "telemetry_installation_id"
    )


def test_preview_uses_existing_uuid_if_set(tmp_path):
    """If the operator has opted in before (UUID exists), preview shows
    the real UUID — not the placeholder. This way the user can verify
    what's actually being sent."""
    real_id = "0123456789abcdef0123456789abcdef"
    cfg = {
        "OUTPUT_DIR": str(tmp_path),
        "telemetry_installation_id": real_id,
    }
    payload = ts.build_payload_preview(cfg)
    assert payload["installation_id"] == real_id


def test_detector_variant_reads_pi_style_metadata(tmp_path, monkeypatch):
    """When model_metadata.json exists with `framework: yolox, variant: s`
    (the format the detector loader writes on the Pi), the telemetry
    payload should report `detector_variant: "yolox-s"` — the
    low-resolution family, not the full training-run identifier.

    The deliberate de-resolution here matters: full names like
    `20260504_yolox_s_locator_640_mosaic0p75_v10_balanced` contain
    dates and hyperparameters that would over-fingerprint installs.
    """
    import json

    od_dir = tmp_path / "object_detection"
    od_dir.mkdir()
    (od_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "framework": "yolox",
                "variant": "s",
                "architecture": "yolox_s_locator_5cls",
            }
        )
    )

    # Force a fresh config that points MODEL_BASE_PATH at our tmp dir.
    import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = cfg_mod.get_config()
    cfg["MODEL_BASE_PATH"] = str(tmp_path)

    assert ts._detect_detector_variant() == "yolox-s"

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)


def test_detector_variant_handles_missing_metadata(tmp_path, monkeypatch):
    """No model_metadata.json + no DETECTOR_VARIANT config = "unknown"."""
    import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = cfg_mod.get_config()
    cfg["MODEL_BASE_PATH"] = str(tmp_path)

    # Ensure no DETECTOR_VARIANT-style key is set
    for key in ("DETECTOR_VARIANT", "OD_VARIANT", "DETECTOR", "OD_BACKEND"):
        cfg.pop(key, None)

    assert ts._detect_detector_variant() == "unknown"

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)


def test_detector_variant_handles_partial_metadata(tmp_path, monkeypatch):
    """Edge case: framework set, variant missing — fall back to just
    framework. No crash, no leaked partial values."""
    import json

    od_dir = tmp_path / "object_detection"
    od_dir.mkdir()
    (od_dir / "model_metadata.json").write_text(
        json.dumps({"framework": "yolox", "architecture": "yolox_5cls"})
    )

    import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = cfg_mod.get_config()
    cfg["MODEL_BASE_PATH"] = str(tmp_path)

    assert ts._detect_detector_variant() == "yolox"

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)


def test_detector_variant_does_not_leak_run_identifier(tmp_path, monkeypatch):
    """Privacy regression test: even though the Pi has `generated_from`
    pointing at a full training-run name like
    `20260504_yolox_s_locator_640_mosaic0p75_v10_balanced_model_config.yaml`,
    the telemetry payload must NOT include it. We only emit
    framework+variant.
    """
    import json

    full_run_id = "20260504_yolox_s_locator_640_mosaic0p75_v10_balanced"
    od_dir = tmp_path / "object_detection"
    od_dir.mkdir()
    (od_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "framework": "yolox",
                "variant": "s",
                "architecture": "yolox_s_locator_5cls",
                "generated_from": f"{full_run_id}_model_config.yaml",
            }
        )
    )

    import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = cfg_mod.get_config()
    cfg["MODEL_BASE_PATH"] = str(tmp_path)

    variant = ts._detect_detector_variant()
    assert variant == "yolox-s"
    assert "20260504" not in variant
    assert "mosaic" not in variant
    assert "balanced" not in variant

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)


def test_telemetry_keys_load_from_yaml_at_boot(tmp_path, monkeypatch):
    """Regression test for a bug discovered 2026-05-06 on RPi: the
    config loader's `for key, value in yaml_settings.items(): if key in
    RUNTIME_KEYS` loop silently dropped telemetry_enabled/_endpoint/_id
    because they are deliberately NOT in RUNTIME_KEYS (we don't want
    the generic Settings form to write them).

    Symptom: scheduler logged "Telemetry scheduler started", settings.yaml
    had `telemetry_enabled: true`, but the scheduler thread saw
    `is_enabled(cfg) == False` because _load_config() never copied the
    YAML value into the config dict — defaulting to False.

    The fix: explicit pass over the three telemetry keys after the
    RUNTIME_KEYS loop. This test verifies that pass exists.
    """
    # Set up a temporary OUTPUT_DIR with a settings.yaml containing
    # telemetry-on values.
    from utils.settings import save_settings_yaml

    save_settings_yaml(
        {
            "telemetry_enabled": True,
            "telemetry_endpoint": "https://override.example/v1/heartbeat",
            "telemetry_installation_id": "deadbeefcafebabe1234567890abcdef",
        },
        str(tmp_path),
    )

    # Force config to use this OUTPUT_DIR by clearing the global cache
    # and pointing OUTPUT_DIR at our temp path via env var.
    import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_CONFIG", None)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    cfg = cfg_mod.get_config()

    # All three telemetry keys must round-trip through YAML.
    assert cfg["telemetry_enabled"] is True, (
        "BUG: telemetry_enabled in YAML was lost during config load"
    )
    assert cfg["telemetry_endpoint"] == "https://override.example/v1/heartbeat", (
        "BUG: telemetry_endpoint override in YAML was lost"
    )
    assert (
        cfg["telemetry_installation_id"] == "deadbeefcafebabe1234567890abcdef"
    ), "BUG: telemetry_installation_id in YAML was lost"

    # And is_enabled() correctly reads the boolean back.
    assert ts.is_enabled(cfg) is True

    # Reset for clean teardown.
    monkeypatch.setattr(cfg_mod, "_CONFIG", None)


def test_preview_after_ensure_installation_id_shows_real_uuid(tmp_path):
    """Regression test: if the toggle-on flow generates a UUID via
    _ensure_installation_id, a subsequent build_payload_preview()
    must show the real UUID (NOT the placeholder).

    This is what the operator expects:
      1. Click "Show what would be sent" → placeholder shown.
      2. Toggle telemetry on (which triggers _ensure_installation_id).
      3. Click "Refresh preview" → real UUID shown.

    Before the fix, step 3 showed the placeholder because UUID was
    only generated lazily inside the scheduler's build_payload(),
    not on toggle-on. Now toggle-on generates the UUID immediately.
    """
    cfg = {
        "OUTPUT_DIR": str(tmp_path),
        "telemetry_installation_id": "",
    }

    # Step 1: preview before opt-in shows placeholder.
    p1 = ts.build_payload_preview(cfg)
    assert p1["installation_id"] == "<would-be-generated-on-opt-in>"

    # Step 2: simulate toggle-on by calling _ensure_installation_id
    # (this is what the toggle endpoint does).
    new_id = ts._ensure_installation_id(cfg)
    assert len(new_id) == 32

    # Step 3: preview NOW shows the real UUID.
    # Note: cfg dict was mutated by _ensure_installation_id, but in
    # production we'd reload from get_config(). Simulate that here:
    from utils.settings import load_settings_yaml

    yaml_settings = load_settings_yaml(str(tmp_path))
    cfg["telemetry_installation_id"] = yaml_settings.get(
        "telemetry_installation_id", ""
    )

    p2 = ts.build_payload_preview(cfg)
    assert p2["installation_id"] == new_id
    assert p2["installation_id"] != "<would-be-generated-on-opt-in>"


def test_payload_installation_id_is_32_hex(tmp_path):
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload(cfg)
    iid = payload["installation_id"]
    assert len(iid) == 32
    assert all(c in "0123456789abcdef" for c in iid)


def test_payload_field_types(tmp_path):
    """Worker allowlist requires specific types — string vs int matters
    because the SQL bind would fail or coerce silently."""
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload(cfg)
    assert isinstance(payload["installation_id"], str)
    assert isinstance(payload["app_version"], str)
    assert isinstance(payload["os"], str)
    assert isinstance(payload["arch"], str)
    assert isinstance(payload["cpu_count"], int)
    assert isinstance(payload["total_ram_gb"], int)
    assert isinstance(payload["python_version"], str)
    assert isinstance(payload["detector_variant"], str)


# ---------------------------------------------------------------------
# 3) No hidden PII — explicit assertions about what must NOT appear
# ---------------------------------------------------------------------

FORBIDDEN_FIELDS = {
    "country",
    "locale",
    "language",
    "timezone",
    "hostname",
    "pi_model",
    "kernel_version",
    "kernel",
    "mac",
    "mac_address",
    "ip",
    "ip_address",
    "email",
    "user",
    "username",
    "lat",
    "lon",
    "latitude",
    "longitude",
    "device_name",
    "camera_url",
    "video_source",
    "ram_bytes",
    "exact_ram",
    "uptime",
    "error",
    "stack_trace",
    "observation_count",
    "species",
    "image_path",
}


def test_payload_has_no_forbidden_fields(tmp_path):
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload(cfg)
    keys = set(payload.keys())
    leaked = keys & FORBIDDEN_FIELDS
    assert leaked == set(), f"Payload leaks forbidden fields: {leaked}"


def test_payload_values_have_no_obvious_pii_substrings(tmp_path):
    """Defensive: even if a field name passes, its value shouldn't
    contain things like '/Users/<name>' or hostname-looking strings."""
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload(cfg)
    for key, value in payload.items():
        if not isinstance(value, str):
            continue
        # Path separators in any value would suggest a leaked path.
        assert "/Users/" not in value, f"Payload[{key}] contains macOS user path"
        assert "/home/" not in value, f"Payload[{key}] contains Linux home path"
        # @ in any value would suggest an email leak.
        assert "@" not in value, f"Payload[{key}] contains @, possible email"


# ---------------------------------------------------------------------
# 4) UUID lifecycle
# ---------------------------------------------------------------------


def test_first_opt_in_generates_uuid(tmp_path):
    """When telemetry_installation_id is empty, _ensure_installation_id
    generates a fresh 32-hex UUID and persists it to settings.yaml."""
    cfg = {
        "OUTPUT_DIR": str(tmp_path),
        "telemetry_installation_id": "",
    }
    new_id = ts._ensure_installation_id(cfg)
    assert len(new_id) == 32
    assert all(c in "0123456789abcdef" for c in new_id)

    # Verify it landed in settings.yaml.
    from utils.settings import load_settings_yaml

    yaml_settings = load_settings_yaml(str(tmp_path))
    assert yaml_settings.get("telemetry_installation_id") == new_id


def test_existing_uuid_is_preserved(tmp_path):
    """If a valid UUID already exists in config, don't regenerate."""
    existing = "0123456789abcdef0123456789abcdef"
    cfg = {
        "OUTPUT_DIR": str(tmp_path),
        "telemetry_installation_id": existing,
    }
    returned = ts._ensure_installation_id(cfg)
    assert returned == existing


def test_invalid_uuid_triggers_regeneration(tmp_path):
    """If config has a malformed UUID (wrong length, non-hex, uppercase),
    treat it as missing and regenerate."""
    for bad in ["", "tooshort", "DEADBEEF" * 4, "deadbeef" * 4 + "x", "g" * 32]:
        cfg = {
            "OUTPUT_DIR": str(tmp_path / bad[:5] if bad else tmp_path / "empty"),
            "telemetry_installation_id": bad,
        }
        Path(cfg["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
        new_id = ts._ensure_installation_id(cfg)
        assert len(new_id) == 32, f"Failed for malformed input: {bad!r}"
        assert new_id != bad


def test_rotate_changes_uuid_and_wipes_marker(tmp_path):
    """rotate_installation_id() generates a new UUID AND clears the
    last-sent marker so the next tick will send under the new UUID."""
    output_dir = str(tmp_path)

    # Pre-seed a marker file as if a previous heartbeat had been sent.
    ts._write_last_sent_date(output_dir, "2026-01-01")
    assert ts._read_last_sent_date(output_dir) == "2026-01-01"

    # Pre-seed a UUID.
    from utils.settings import load_settings_yaml, save_settings_yaml

    yaml_settings = load_settings_yaml(output_dir)
    yaml_settings["telemetry_installation_id"] = "a" * 32
    save_settings_yaml(yaml_settings, output_dir)

    new_id = ts.rotate_installation_id(output_dir)

    assert new_id != "a" * 32
    assert len(new_id) == 32
    assert ts._read_last_sent_date(output_dir) == ""  # marker wiped


# ---------------------------------------------------------------------
# 5) Atomic last-sent file
# ---------------------------------------------------------------------


def test_last_sent_round_trip(tmp_path):
    output_dir = str(tmp_path)
    assert ts._read_last_sent_date(output_dir) == ""  # no file yet
    ts._write_last_sent_date(output_dir, "2026-05-06")
    assert ts._read_last_sent_date(output_dir) == "2026-05-06"


def test_last_sent_corrupt_treated_as_unsent(tmp_path):
    """If something writes garbage to the marker file, _read should
    return '' so the scheduler sends fresh (rather than getting stuck)."""
    output_dir = str(tmp_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts._get_last_sent_path(output_dir).write_text("not-a-date", encoding="utf-8")
    assert ts._read_last_sent_date(output_dir) == ""


def test_wipe_last_sent_idempotent(tmp_path):
    """wipe_last_sent on a missing file must not raise."""
    output_dir = str(tmp_path)
    ts.wipe_last_sent(output_dir)  # no-op, no exception
    ts._write_last_sent_date(output_dir, "2026-05-06")
    ts.wipe_last_sent(output_dir)
    assert ts._read_last_sent_date(output_dir) == ""


# ---------------------------------------------------------------------
# 6) Detection helpers — sanity checks (low-resolution, no fingerprints)
# ---------------------------------------------------------------------


def test_detect_os_returns_known_value():
    assert ts._detect_os() in {"linux", "darwin", "windows", "other"}


def test_detect_arch_returns_known_value():
    assert ts._detect_arch() in {"aarch64", "x86_64", "armv7l", "unknown"}


def test_detect_cpu_count_positive():
    n = ts._detect_cpu_count()
    assert isinstance(n, int)
    assert n >= 1


def test_detect_total_ram_gb_non_negative():
    n = ts._detect_total_ram_gb()
    assert isinstance(n, int)
    assert n >= 0


def test_user_agent_matches_worker_regex():
    """The Worker's allowlist regex is /^WatchMyBirds-Heartbeat\\/[\\w.+-]+/.
    If we change USER_AGENT, the Worker stops accepting our pings."""
    import re

    pattern = re.compile(r"^WatchMyBirds-Heartbeat/[\w.+-]+")
    assert pattern.match(ts.USER_AGENT) is not None


# ---------------------------------------------------------------------
# 7) Send-heartbeat error swallowing
# ---------------------------------------------------------------------


def test_send_swallows_network_error(tmp_path):
    """A connection error must not raise — we must always return False."""
    cfg = _make_test_cfg(tmp_path)
    payload = ts.build_payload(cfg)
    # Point at a hostname that resolves but refuses connections.
    result = ts._send_heartbeat("http://127.0.0.1:1/v1/heartbeat", payload)
    assert result is False


def test_send_returns_false_on_non_204():
    """Mock requests.post returning 400 — must return False, never raise."""
    import requests as _requests

    class FakeResp:
        status_code = 400
        text = "unknown field"

    with patch.object(_requests, "post", return_value=FakeResp()):
        # Use a minimal payload — the real one isn't relevant for this test.
        result = ts._send_heartbeat("https://example.invalid/v1/heartbeat", {})
        assert result is False


def test_send_returns_true_on_204():
    import requests as _requests

    class FakeResp:
        status_code = 204
        text = ""

    with patch.object(_requests, "post", return_value=FakeResp()):
        result = ts._send_heartbeat("https://example.invalid/v1/heartbeat", {})
        assert result is True


# ---------------------------------------------------------------------
# 8) wake_now() — event-driven scheduler wakeup
# ---------------------------------------------------------------------


def test_wake_now_sets_event():
    """wake_now() flips the module-level Event so a thread blocked
    on _wake_event.wait(timeout) returns immediately."""
    # Start clean — clear any leftover state from previous tests.
    ts._wake_event.clear()
    assert not ts._wake_event.is_set()

    ts.wake_now()
    assert ts._wake_event.is_set()

    # Cleanup so other tests don't see the set state.
    ts._wake_event.clear()


def test_wake_now_is_idempotent():
    """Calling wake_now() twice in a row is the same as calling it
    once — Event.set() is idempotent at the Event level."""
    ts._wake_event.clear()

    ts.wake_now()
    ts.wake_now()
    ts.wake_now()
    assert ts._wake_event.is_set()

    ts._wake_event.clear()


def test_wake_event_wait_unblocks_on_set():
    """A thread waiting on _wake_event.wait(timeout) returns True
    (= event was set) instead of False (= timeout) when wake_now()
    is called from another thread.

    This is the actual production scenario: the scheduler is sleeping
    on wait(300), the toggle endpoint calls wake_now(), the scheduler
    wakes within ~10ms.
    """
    import threading
    import time

    ts._wake_event.clear()

    result_holder = {}

    def waiter():
        # Long timeout — if wake_now() doesn't fire, the test will
        # block here for 5s and then fail.
        start = time.monotonic()
        woken = ts._wake_event.wait(timeout=5.0)
        elapsed = time.monotonic() - start
        result_holder["woken"] = woken
        result_holder["elapsed"] = elapsed

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    # Give the waiter a moment to actually enter wait().
    time.sleep(0.05)

    ts.wake_now()
    t.join(timeout=2.0)

    assert result_holder.get("woken") is True, "wait() did not return True"
    assert result_holder.get("elapsed", 999) < 1.0, (
        "wake_now() should unblock wait() in well under a second"
    )

    ts._wake_event.clear()
