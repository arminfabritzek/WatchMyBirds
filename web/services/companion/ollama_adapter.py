"""Ollama HTTP adapter for the Companion.

First and currently only concrete inference adapter. Talks to a local
Ollama daemon over HTTP using stdlib only (urllib) so we do not pull
the ``ollama`` python client into WMB just for this.

This adapter is deliberately small. Sampling parameters are kept here
because they are runtime-specific; the rest of the Companion stack
(prompt building, lease, recorder, API) does not import this file.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cleaner import clean_model_text
from .inference import CompanionInferenceClient, CompanionInferenceResult

logger = logging.getLogger(__name__)


class OllamaInferenceAdapter(CompanionInferenceClient):
    """Concrete inference client backed by a local Ollama daemon."""

    def __init__(self, *, base_url: str, model_tag: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_tag

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        timeout_s: float,
    ) -> CompanionInferenceResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": False,
            "think": False,
            "options": self._sampling_for(self.model_id),
        }
        started = time.monotonic()
        try:
            req = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=max(0.1, timeout_s)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            status = "timeout" if isinstance(exc, TimeoutError) else "unreachable"
            return CompanionInferenceResult(
                status=status,
                filter_reason=f"transport: {type(exc).__name__}",
                elapsed_ms=elapsed_ms,
                model_id=self.model_id,
            )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        raw = str((data.get("message") or {}).get("content") or "").strip()
        cleaned = clean_model_text(raw)
        if not cleaned:
            return CompanionInferenceResult(
                status="unreachable",
                raw=raw,
                filter_reason="empty",
                elapsed_ms=elapsed_ms,
                model_id=self.model_id,
            )
        return CompanionInferenceResult(
            status="ok",
            text=cleaned,
            raw=raw,
            elapsed_ms=elapsed_ms,
            model_id=self.model_id,
        )

    @staticmethod
    def _sampling_for(model_tag: str) -> dict[str, Any]:
        """Per-family sampling. Conservative defaults that avoid the
        Qwen3 repetition trap on low-temp greedy decoding.
        """
        name = model_tag.lower()
        if "qwen3" in name or "gemma3" in name:
            return {
                "temperature": 0.85,
                "top_p": 0.9,
                "top_k": 30,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "num_predict": 200,
                "repeat_penalty": 1.1,
            }
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "num_predict": 200,
            "repeat_penalty": 1.1,
        }
