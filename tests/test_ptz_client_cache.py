"""Tests for the per-camera PtzClient cache and command serialization.

Before this cache, core.ptz_core built a fresh PtzClient on every PTZ
command, so the ONVIF handshake (create_*_service + GetProfiles) ran on
every move/goto/stop. These tests pin the new behaviour:

- repeated commands reuse one client (the responsiveness fix),
- a connection-fingerprint change rebuilds the client,
- explicit invalidation drops the client,
- an ONVIF error evicts the client so the next command self-heals,
- commands on one camera are serialized (manual + auto never interleave).
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import core.ptz_core as ptz_core


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Each test starts and ends with empty client + move-backpressure state."""
    ptz_core.clear_ptz_client_cache()
    with ptz_core._ptz_move_state_lock:
        ptz_core._ptz_move_waiting.clear()
        ptz_core._ptz_stop_generation.clear()
    yield
    ptz_core.clear_ptz_client_cache()
    with ptz_core._ptz_move_state_lock:
        ptz_core._ptz_move_waiting.clear()
        ptz_core._ptz_stop_generation.clear()


def _storage_for(camera: dict):
    storage = MagicMock()
    storage.get_camera.return_value = camera
    return storage


_CAM = {
    "id": 0,
    "ip": "10.0.0.5",
    "port": 80,
    "username": "admin",
    "password": "pw",
    "ptz": {"enabled": True, "profile_index": 0},
}


def test_repeated_commands_reuse_one_client():
    """N commands construct the PtzClient exactly once.

    This is the core fix: before, each command rebuilt the client and
    re-ran the ONVIF handshake. The assertion `call_count == 1` is the
    measurable contract.
    """
    fake_client = MagicMock()
    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=fake_client) as ctor,
    ):
        for _ in range(10):
            ptz_core.continuous_move(0, pan=0.5, tilt=0.0, zoom=0.0, duration_ms=100)
        ptz_core.stop(0)
        ptz_core.goto_preset(0, "Preset001")

    assert ctor.call_count == 1, "client must be built once and reused"
    assert fake_client.continuous_move.call_count == 10
    assert fake_client.stop.call_count == 1
    assert fake_client.goto_preset.call_count == 1


def test_fingerprint_change_rebuilds_client():
    """Editing ip/port/credentials/profile_index rebuilds the client."""
    cam_a = dict(_CAM)
    cam_b = dict(_CAM, password="rotated")
    storage = MagicMock()
    storage.get_camera.side_effect = [cam_a, cam_b, cam_b]

    with patch("core.ptz_core.get_camera_storage", return_value=storage):
        with patch("core.ptz_core.PtzClient", return_value=MagicMock()) as ctor:
            ptz_core.stop(0)  # builds with cam_a
            ptz_core.stop(0)  # cam_b fingerprint differs -> rebuild
            ptz_core.stop(0)  # cam_b unchanged -> reuse

    assert ctor.call_count == 2


def test_explicit_invalidation_drops_client():
    """clear_ptz_client_cache(camera_id) forces a rebuild on next command."""
    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=MagicMock()) as ctor,
    ):
        ptz_core.stop(0)
        ptz_core.clear_ptz_client_cache(0)
        ptz_core.stop(0)

    assert ctor.call_count == 2


def test_clear_auto_ptz_camera_cache_also_drops_client():
    """The two caches are coherent: clearing camera config clears clients."""
    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=MagicMock()) as ctor,
    ):
        ptz_core.stop(0)
        ptz_core.clear_auto_ptz_camera_cache()
        ptz_core.stop(0)

    assert ctor.call_count == 2


def test_error_evicts_client_and_next_command_rebuilds():
    """An ONVIF error drops the cached client so the next command rebuilds.

    Self-healing: a dropped socket / rebooted cam / rotated credential
    must not leave a permanently broken cached client.
    """
    bad = MagicMock()
    bad.stop.side_effect = RuntimeError("onvif boom")
    good = MagicMock()

    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", side_effect=[bad, good]) as ctor,
    ):
        with pytest.raises(RuntimeError):
            ptz_core.stop(0)
        # cache was evicted on error; next command builds a fresh client
        ptz_core.stop(0)

    assert ctor.call_count == 2
    assert good.stop.call_count == 1


def test_commands_on_one_camera_are_serialized():
    """The per-camera lock prevents ContinuousMove/Stop interleaving.

    Simulates a manual move and an auto-PTZ goto racing on one camera.
    The fake client records enter/exit around a sleep; with the lock,
    the two command bodies must not overlap.
    """
    overlap = {"max_concurrent": 0, "current": 0}
    overlap_lock = threading.Lock()

    class _SlowClient:
        def _enter(self):
            with overlap_lock:
                overlap["current"] += 1
                overlap["max_concurrent"] = max(
                    overlap["max_concurrent"], overlap["current"]
                )

        def _exit(self):
            with overlap_lock:
                overlap["current"] -= 1

        def continuous_move(self, **_):
            self._enter()
            time.sleep(0.05)
            self._exit()

        def goto_preset(self, **_):
            self._enter()
            time.sleep(0.05)
            self._exit()

    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=_SlowClient()),
    ):
        t1 = threading.Thread(
            target=lambda: ptz_core.continuous_move(
                0, pan=0.5, tilt=0.0, zoom=0.0, duration_ms=50
            )
        )
        t2 = threading.Thread(target=lambda: ptz_core.goto_preset(0, "Preset002"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert overlap["max_concurrent"] == 1, "commands on one camera must not overlap"


def test_different_cameras_use_independent_clients_and_locks():
    """Two cameras get separate cached clients and separate locks."""
    cam0 = dict(_CAM, id=0, ip="10.0.0.5")
    cam1 = dict(_CAM, id=1, ip="10.0.0.6")
    storage = MagicMock()
    storage.get_camera.side_effect = lambda cid, **_: cam0 if cid == 0 else cam1

    with (
        patch("core.ptz_core.get_camera_storage", return_value=storage),
        patch(
            "core.ptz_core.PtzClient", side_effect=[MagicMock(), MagicMock()]
        ) as ctor,
    ):
        ptz_core.stop(0)
        ptz_core.stop(1)
        ptz_core.stop(0)
        ptz_core.stop(1)

    assert ctor.call_count == 2  # one client per camera, both reused
    assert ptz_core._command_lock_for_camera(
        0
    ) is not ptz_core._command_lock_for_camera(1)


# ---------------------------------------------------------------------------
# Hold-to-move backpressure: coalescing + stop supersession.
# These pin the fix for the post-release run-on observed on the real camera
# (2s hold -> ~4s run-on) without touching the frontend heartbeat or the
# dead-man-switch sleep.
# ---------------------------------------------------------------------------


def test_move_coalesced_while_one_runs_and_one_waits():
    """At most one move runs and one waits; extra concurrent moves drop.

    Three moves fired at once on a slow client: one executes, one waits for
    the lock, the third is coalesced away (never reaches the camera).
    """
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    calls_lock = threading.Lock()

    class _BlockingClient:
        def continuous_move(self, **_):
            with calls_lock:
                calls["n"] += 1
            started.set()
            release.wait(timeout=2.0)

    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=_BlockingClient()),
    ):

        def fire():
            ptz_core.continuous_move(0, pan=0.5, tilt=0.0, zoom=0.0, duration_ms=50)

        t1 = threading.Thread(target=fire)
        t1.start()
        assert started.wait(timeout=1.0)  # first move is now running (holds lock)

        # With one running and (soon) one waiting, a burst of further moves
        # must be coalesced away. The first claims the waiting slot; the rest
        # return immediately without ever reaching the camera.
        t2 = threading.Thread(target=fire)  # claims the single waiting slot
        t2.start()
        time.sleep(0.05)  # let t2 register as waiting
        for _ in range(5):
            ptz_core.continuous_move(0, pan=0.5, tilt=0.0, zoom=0.0, duration_ms=50)

        release.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

    # Only the running move + the single waiting move ever reached the camera;
    # the five burst moves were coalesced.
    assert calls["n"] == 2, f"expected 2 camera calls, got {calls['n']}"


def test_stop_supersedes_move_waiting_for_lock():
    """A stop issued while a move waits cancels that move's ONVIF call.

    This is the release-Stop-cancels-backlog contract: the waiting move must
    not move the camera after the operator let go.
    """
    move_started = threading.Event()
    release_first = threading.Event()
    record = {"moves": 0, "stops": 0}
    rec_lock = threading.Lock()

    class _Client:
        def continuous_move(self, **_):
            with rec_lock:
                record["moves"] += 1
            move_started.set()
            release_first.wait(timeout=2.0)

        def stop(self, **_):
            with rec_lock:
                record["stops"] += 1

    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=_Client()),
    ):
        # Move A starts and holds the lock.
        ta = threading.Thread(
            target=lambda: ptz_core.continuous_move(
                0, pan=0.5, tilt=0.0, zoom=0.0, duration_ms=50
            )
        )
        ta.start()
        assert move_started.wait(timeout=1.0)

        # Move B queues behind the lock (claims the waiting slot).
        tb = threading.Thread(
            target=lambda: ptz_core.continuous_move(
                0, pan=0.5, tilt=0.0, zoom=0.0, duration_ms=50
            )
        )
        tb.start()
        time.sleep(0.05)  # let B register as waiting

        # Operator releases: stop bumps the generation, then waits for lock.
        ts = threading.Thread(target=lambda: ptz_core.stop(0))
        ts.start()
        time.sleep(0.05)

        # Let A finish; B and the stop now drain.
        release_first.set()
        ta.join(timeout=2.0)
        tb.join(timeout=2.0)
        ts.join(timeout=2.0)

    # Move A ran (1). Move B was superseded by the stop -> did NOT run.
    assert record["moves"] == 1, (
        f"waiting move should be cancelled, got {record['moves']}"
    )
    assert record["stops"] == 1


def test_goto_preset_records_generation_for_supersession():
    """A goto issued normally still executes (no stop in between)."""
    fake = MagicMock()
    with (
        patch("core.ptz_core.get_camera_storage", return_value=_storage_for(_CAM)),
        patch("core.ptz_core.PtzClient", return_value=fake),
    ):
        ptz_core.goto_preset(0, "Preset005")

    assert fake.goto_preset.call_count == 1
