"""Companion service orchestrator.

Single entry point the API blueprint calls into. Owns the inference
client, the recorder, the safety guard, and the prompt builder. Takes
the compute lease around every inference call so OD/CLS pause for
Companion runs (configurable) and Companion + tagger never overlap.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .cleaner import clean_model_text
from .inference import CompanionInferenceClient, CompanionInferenceResult
from .prompt import (
    CompanionContext,
    Language,
    Tone,
    build_messages,
    build_system_prompt,
)
from .recorder import CompanionRecorder
from .safety import check as safety_check

logger = logging.getLogger(__name__)


_FALLBACK_FILTERED_DE = "Lieber nichts dazu — der Spruch passte nicht."
_FALLBACK_FILTERED_EN = "Better not — the line did not pass review."

_LEASE_HOLDER = "companion_inference"


class CompanionService:
    def __init__(
        self,
        *,
        client: CompanionInferenceClient | None,
        recorder: CompanionRecorder,
        lease,
        enabled: bool,
        pause_detection: bool,
        timeout_s: float,
    ) -> None:
        self._client = client
        self._recorder = recorder
        self._lease = lease
        self.enabled = enabled
        self.pause_detection = pause_detection
        self.timeout_s = float(timeout_s)
        self._last_utterance_ts: str | None = None

    # ------------------------------------------------------------------ status

    def state(self) -> dict[str, Any]:
        lease_status = self._lease.status() if self._lease is not None else None
        busy_holder: str | None = None
        if lease_status is not None and lease_status.holder is not None:
            busy_holder = lease_status.holder
        return {
            "enabled": self.enabled,
            "configured": self._client is not None,
            "model_id": getattr(self._client, "model_id", ""),
            "pause_detection": self.pause_detection,
            "timeout_s": self.timeout_s,
            "busy": busy_holder is not None and busy_holder == _LEASE_HOLDER,
            "lease_holder": busy_holder,
            "last_utterance_ts": self._last_utterance_ts,
        }

    def recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._recorder.recent(limit=limit)

    def feedback(self, *, trigger_id: str, vote: str | None) -> dict[str, Any] | None:
        return self._recorder.set_feedback(trigger_id, vote)

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        *,
        message: str,
        language: Language = "de",
        tone: Tone = "kid_friendly",
    ) -> dict[str, Any]:
        ctx = CompanionContext(language=language, tone=tone)
        return self._generate(
            ctx=ctx,
            message=message,
            trigger="chat",
            user_message=message,
        )

    # ------------------------------------------------------------------ event

    def event(
        self,
        *,
        species: str,
        count: int = 1,
        rare: bool = False,
        language: Language = "de",
        tone: Tone = "kid_friendly",
    ) -> dict[str, Any]:
        species = (species or "").strip()
        count = max(1, int(count))
        rare_note = " (rare)" if rare else ""
        count_note = f" ×{count}" if count > 1 else ""
        ev_string = f"{species}{count_note}{rare_note}"
        ctx = CompanionContext(
            language=language,
            tone=tone,
            recent_events=(ev_string,),
        )
        if language == "de":
            user_message = (
                f"Auf der Kamera: {species}{count_note}{rare_note}. Was sagst du?"
            )
        else:
            user_message = (
                f"On camera: {species}{count_note}{rare_note}. What do you say?"
            )
        return self._generate(
            ctx=ctx,
            message=user_message,
            trigger=f"event:{species}",
            user_message=user_message,
        )

    # ------------------------------------------------------------------ core

    def _generate(
        self,
        *,
        ctx: CompanionContext,
        message: str,
        trigger: str,
        user_message: str | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "ok": False,
                "status": "disabled",
                "reason": "COMPANION_ENABLED is false",
            }

        if self._client is None:
            return {
                "ok": False,
                "status": "unreachable",
                "reason": "inference adapter not configured",
            }

        if self._lease is None:
            # Hard-required dependency in production; fail loud rather
            # than silently bypassing the contention guard.
            logger.error("Companion: compute lease service is not initialised")
            return {
                "ok": False,
                "status": "disabled",
                "reason": "compute lease unavailable",
            }

        # The lease module's exception types live in another module;
        # local import avoids a cycle at module load.
        from web.services.compute_lease_service import LeaseBusy

        try:
            with self._lease.acquire(
                _LEASE_HOLDER,
                pause_detection=self.pause_detection,
                reason=f"companion: {trigger}",
                timeout_s=self.timeout_s + 5.0,
            ):
                result = self._client.generate(
                    system_prompt=build_system_prompt(ctx),
                    messages=build_messages(ctx, message=message),
                    timeout_s=self.timeout_s,
                )
        except LeaseBusy as exc:
            return {
                "ok": False,
                "status": "busy",
                "reason": f"compute lease held by {exc.current_holder}",
            }

        return self._finalise(
            ctx=ctx,
            result=result,
            trigger=trigger,
            user_message=user_message,
        )

    def _finalise(
        self,
        *,
        ctx: CompanionContext,
        result: CompanionInferenceResult,
        trigger: str,
        user_message: str | None,
    ) -> dict[str, Any]:
        # Even non-OK results get logged so we can debug Pi reachability.
        if result.status != "ok":
            entry = self._recorder.record(
                text="",
                raw_text=result.raw,
                source="adapter",
                model_id=result.model_id,
                status=result.status,
                filter_reason=result.filter_reason,
                language=ctx.language,
                tone=ctx.tone,
                daypart=ctx.time_of_day,
                trigger=trigger,
                context_echo=ctx.echo(),
                user_message=user_message,
                elapsed_ms=result.elapsed_ms,
            )
            return {
                "ok": False,
                "status": result.status,
                "reason": result.filter_reason or "non_ok",
                "model_id": result.model_id,
                "elapsed_ms": result.elapsed_ms,
                "trigger_id": entry["trigger_id"],
            }

        # Re-clean defensively: adapters may forget. cleaner is idempotent.
        cleaned = clean_model_text(result.text)
        hit = safety_check(cleaned)
        if hit is not None:
            fallback = (
                _FALLBACK_FILTERED_DE
                if ctx.language == "de"
                else _FALLBACK_FILTERED_EN
            )
            entry = self._recorder.record(
                text=fallback,
                raw_text=result.raw or cleaned,
                source="fallback_filtered",
                model_id=result.model_id,
                status="filtered",
                filter_reason=hit.category,
                language=ctx.language,
                tone=ctx.tone,
                daypart=ctx.time_of_day,
                trigger=trigger,
                context_echo=ctx.echo(),
                user_message=user_message,
                elapsed_ms=result.elapsed_ms,
            )
            self._last_utterance_ts = entry["ts"]
            return {
                "ok": True,
                "status": "filtered",
                "text": fallback,
                "model_id": result.model_id,
                "elapsed_ms": result.elapsed_ms,
                "trigger_id": entry["trigger_id"],
                "filter_reason": hit.category,
            }

        entry = self._recorder.record(
            text=cleaned,
            raw_text=result.raw,
            source="adapter",
            model_id=result.model_id,
            status="ok",
            filter_reason="",
            language=ctx.language,
            tone=ctx.tone,
            daypart=ctx.time_of_day,
            trigger=trigger,
            context_echo=ctx.echo(),
            user_message=user_message,
            elapsed_ms=result.elapsed_ms,
        )
        self._last_utterance_ts = entry["ts"]
        return {
            "ok": True,
            "status": "ok",
            "text": cleaned,
            "model_id": result.model_id,
            "elapsed_ms": result.elapsed_ms,
            "trigger_id": entry["trigger_id"],
        }


def now_iso() -> str:
    return datetime.now().isoformat()
