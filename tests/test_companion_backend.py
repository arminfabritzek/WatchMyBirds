"""Companion backend unit tests.

Exercises the service in isolation with a fake ``CompanionInferenceClient``
and a real ``CompanionRecorder`` writing into ``tmp_path``.
"""

from __future__ import annotations

from typing import Any

from web.services.companion.cleaner import clean_model_text
from web.services.companion.inference import (
    CompanionInferenceClient,
    CompanionInferenceResult,
)
from web.services.companion.recorder import CompanionRecorder
from web.services.companion.safety import check
from web.services.companion.service import CompanionService
from web.services.compute_lease_service import ComputeLeaseService


class _DM:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused


class _FakeClient(CompanionInferenceClient):
    """Configurable inference adapter for tests."""

    def __init__(self, *, result: CompanionInferenceResult) -> None:
        self.model_id = result.model_id or "fake-model:test"
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, *, system_prompt: str, messages: list[dict[str, str]], timeout_s: float
    ) -> CompanionInferenceResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "timeout_s": timeout_s,
            }
        )
        return self._result


def _service(
    *,
    tmp_path,
    result: CompanionInferenceResult,
    enabled: bool = True,
    pause_detection: bool = True,
    dm: _DM | None = None,
):
    dm = dm or _DM()
    lease = ComputeLeaseService(dm)
    client = _FakeClient(result=result)
    recorder = CompanionRecorder(base_dir=tmp_path)
    svc = CompanionService(
        client=client,
        recorder=recorder,
        lease=lease,
        enabled=enabled,
        pause_detection=pause_detection,
        timeout_s=5.0,
    )
    return svc, dm, lease, client, recorder


def test_chat_pauses_detection_during_inference(tmp_path):
    """detection_manager.paused must be True only while the adapter runs."""
    seen: dict[str, bool] = {"paused_during_call": False}

    class _Probe(CompanionInferenceClient):
        model_id = "probe:1"

        def generate(self, *, system_prompt, messages, timeout_s):
            seen["paused_during_call"] = dm.paused
            return CompanionInferenceResult(
                status="ok",
                text="Hallo Welt.",
                raw="Hallo Welt.",
                model_id=self.model_id,
            )

    dm = _DM(paused=False)
    lease = ComputeLeaseService(dm)
    svc = CompanionService(
        client=_Probe(),
        recorder=CompanionRecorder(base_dir=tmp_path),
        lease=lease,
        enabled=True,
        pause_detection=True,
        timeout_s=5.0,
    )
    out = svc.chat(message="Hi", language="de", tone="kid_friendly")
    assert out["ok"] is True
    assert out["text"] == "Hallo Welt."
    assert seen["paused_during_call"] is True
    # Restored afterwards.
    assert dm.paused is False


def test_chat_disabled_returns_disabled(tmp_path):
    svc, dm, _lease, _client, _rec = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(status="ok", text="x", model_id="m"),
        enabled=False,
    )
    out = svc.chat(message="Hi")
    assert out["ok"] is False
    assert out["status"] == "disabled"
    assert dm.paused is False  # never touched


def test_chat_records_utterance_and_appears_in_recent(tmp_path):
    svc, _dm, _lease, _client, recorder = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(
            status="ok",
            text="Speaker A chirps briefly.",
            raw="Speaker A chirps briefly.",
            model_id="m:1",
            elapsed_ms=42,
        ),
    )
    out = svc.chat(message="Was geht?", language="de", tone="kid_friendly")
    assert out["ok"] is True
    assert out["trigger_id"]
    recents = recorder.recent(limit=10)
    assert len(recents) == 1
    assert recents[0]["text"] == "Speaker A chirps briefly."
    assert recents[0]["language"] == "de"
    assert recents[0]["status"] == "ok"
    assert recents[0]["feedback"] is None


def test_event_records_with_species_in_trigger(tmp_path):
    svc, _dm, _lease, _client, recorder = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(
            status="ok",
            text="Speaker C gets it.",
            raw="Speaker C gets it.",
            model_id="m",
        ),
    )
    out = svc.event(species="kohlmeise", count=2, rare=False, language="de")
    assert out["ok"] is True
    recents = recorder.recent()
    assert recents[0]["trigger"] == "event:kohlmeise"


def test_unreachable_returns_non_ok_and_records_diagnostic(tmp_path):
    svc, _dm, _lease, _client, recorder = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(
            status="unreachable",
            filter_reason="transport: URLError",
            elapsed_ms=10,
            model_id="m",
        ),
    )
    out = svc.chat(message="Hi")
    assert out["ok"] is False
    assert out["status"] == "unreachable"
    recents = recorder.recent()
    assert recents and recents[0]["status"] == "unreachable"


def test_filtered_output_replaced_with_safe_fallback(tmp_path):
    svc, _dm, _lease, _client, recorder = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(
            status="ok",
            text="Alle Ausländer raus aus dem Garten.",  # trips xenophobic
            raw="Alle Ausländer raus aus dem Garten.",
            model_id="m",
        ),
    )
    out = svc.chat(message="x")
    assert out["ok"] is True
    assert out["status"] == "filtered"
    assert "Lieber nichts" in out["text"]
    recents = recorder.recent()
    assert recents[0]["status"] == "filtered"
    assert recents[0]["filter_reason"] == "xenophobic"


def test_feedback_updates_recorded_utterance(tmp_path):
    svc, _dm, _lease, _client, recorder = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(
            status="ok", text="Speaker B observes.", raw="x", model_id="m"
        ),
    )
    out = svc.chat(message="hi")
    tid = out["trigger_id"]
    updated = svc.feedback(trigger_id=tid, vote="up")
    assert updated is not None
    assert updated["feedback"] == "up"


def test_feedback_unknown_trigger_id_returns_none(tmp_path):
    svc, *_ = _service(
        tmp_path=tmp_path,
        result=CompanionInferenceResult(status="ok", text="x", model_id="m"),
    )
    assert svc.feedback(trigger_id="utt_does_not_exist", vote="down") is None


def test_concurrent_tagger_during_companion_inference_sees_busy(tmp_path):
    """While Companion holds the lease, the tagger lease attempt fails fast."""
    from web.services.compute_lease_service import LeaseBusy

    held = {"during": False}

    class _SlowProbe(CompanionInferenceClient):
        model_id = "slow:1"

        def generate(self, *, system_prompt, messages, timeout_s):
            # While we're "running", a tagger acquire attempt must fail.
            try:
                with lease.acquire("aesthetic_tagger", pause_detection=False):
                    held["during"] = True  # would mean overlap is allowed
            except LeaseBusy:
                # Expected: tagger must NOT acquire while companion holds.
                pass
            return CompanionInferenceResult(
                status="ok", text="ok.", raw="ok.", model_id=self.model_id
            )

    dm = _DM()
    lease = ComputeLeaseService(dm)
    svc = CompanionService(
        client=_SlowProbe(),
        recorder=CompanionRecorder(base_dir=tmp_path),
        lease=lease,
        enabled=True,
        pause_detection=True,
        timeout_s=5.0,
    )
    out = svc.chat(message="hi")
    assert out["ok"] is True
    assert held["during"] is False  # tagger correctly refused


def test_cleaner_strips_think_blocks_and_caps_length():
    raw = "<think>noise here</think>Frieda: Hallo Welt." + " ja." * 200
    cleaned = clean_model_text(raw)
    assert "<think>" not in cleaned
    assert cleaned.startswith("Hallo Welt") or cleaned.startswith("Frieda")
    # The cleaner strips role prefix; sentences are capped, total length capped.
    assert len(cleaned) <= 720


def test_safety_check_passes_clean_text():
    assert check("Speaker A chirps briefly, very British.") is None


def test_safety_check_catches_partisan_marker():
    hit = check("AfD ist Mist, sage ich.")
    assert hit is not None
    assert hit.category == "partisan"
