"""Inference client protocol and result type for the Companion.

The Companion service depends on this protocol, never on a concrete
adapter module. Adapters (Ollama HTTP today, possibly llama.cpp or
MLC tomorrow) implement the protocol and are injected at construction
time.

The protocol is intentionally narrow: the Companion service builds the
prompt and the messages list, the adapter only knows how to talk to
its runtime. Sampling parameters that vary across runtimes (top_p,
temperature, etc.) are the adapter's responsibility — they are not
part of this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

InferenceStatus = Literal[
    "ok",
    "unreachable",
    "filtered",
    "disabled",
    "timeout",
]


@dataclass(frozen=True)
class CompanionInferenceResult:
    """Diagnostic record for one inference attempt.

    The Companion service uses every field; the API surfaces only the
    safe ones (status, model_id, elapsed_ms, the cleaned text). The
    raw text and filter reason go to the JSONL recorder for review.
    """

    status: InferenceStatus
    text: str = ""
    raw: str = ""
    filter_reason: str = ""
    elapsed_ms: int = 0
    model_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@runtime_checkable
class CompanionInferenceClient(Protocol):
    """Adapter interface — one runtime, one class."""

    model_id: str

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        timeout_s: float,
    ) -> CompanionInferenceResult:
        """Run one inference and return a diagnostic result.

        Implementations must NOT raise on transport failures: failures
        are reported via ``status`` and ``filter_reason``. Implementations
        MAY raise on programmer errors (bad arguments, missing config).
        """
        ...
