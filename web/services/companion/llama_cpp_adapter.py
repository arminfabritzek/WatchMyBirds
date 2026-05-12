"""llama-cpp-python adapter for the Companion.

Loads a GGUF model in-process and holds the ``Llama`` instance alive
for the lifetime of the WMB Flask process. This is the default
production adapter on the RPi 5 — in-process inference avoids the
HTTP roundtrip an external daemon adapter would add per call.

Threading note: ``llama-cpp-python`` is not safe for concurrent
``__call__`` from multiple threads. The Companion always runs through
``CompanionService`` which acquires the compute lease before calling
``generate``; the lease is single-holder, so this adapter never sees
concurrent calls under normal flow. The internal ``_lock`` is belt-
and-suspenders for the case where someone uses the adapter outside
the service.

Lazy load: the GGUF is only opened on the first ``generate`` call.
This keeps boot fast even when ``COMPANION_ENABLED=true`` but no chat
ever happens. The first call pays the ~2-3 s load cost; subsequent
calls reuse the loaded instance.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from .cleaner import clean_model_text
from .inference import CompanionInferenceClient, CompanionInferenceResult

logger = logging.getLogger(__name__)

# Llama 3.2 chat template literal. The Llama tokenizer adds the
# <|begin_of_text|> BOS token itself (add_bos_token=True by default),
# so we MUST NOT include it in the template — otherwise llama-cpp-python
# emits "duplicate leading <|begin_of_text|>" and reduces output quality.
# An older variant of this adapter predates the BOS-aware default and
# included it verbatim; for our adapter we drop it.
_LLAMA32_CHAT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
_LLAMA32_STOP = ["<|eot_id|>"]


class LlamaCppInferenceAdapter(CompanionInferenceClient):
    """Persistent llama-cpp-python adapter."""

    def __init__(
        self,
        *,
        gguf_path: str,
        n_ctx: int = 4096,
        n_threads: int = 0,
        n_predict: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self._gguf_path = Path(gguf_path)
        self._n_ctx = int(n_ctx)
        self._n_threads = int(n_threads)
        self._n_predict = int(n_predict)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._llm: Any = None  # llama_cpp.Llama once loaded
        self._load_error: str = ""
        self._lock = threading.Lock()
        self.model_id = self._gguf_path.name or "llama_cpp:unknown"

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> bool:
        """Load the GGUF on first use. Returns True on success."""
        if self._llm is not None:
            return True
        if self._load_error:
            return False
        with self._lock:
            if self._llm is not None:
                return True
            if not self._gguf_path.exists():
                self._load_error = f"gguf_not_found: {self._gguf_path}"
                logger.error("Companion llama_cpp: GGUF missing at %s", self._gguf_path)
                return False
            try:
                from llama_cpp import Llama  # type: ignore[import-not-found]
            except ImportError as exc:
                self._load_error = f"llama_cpp_import: {exc}"
                logger.error(
                    "Companion llama_cpp: llama-cpp-python not installed (%s); "
                    "install requirements-companion.txt to enable.",
                    exc,
                )
                return False
            try:
                t0 = time.monotonic()
                kwargs: dict[str, Any] = {
                    "model_path": str(self._gguf_path),
                    "n_ctx": self._n_ctx,
                    "verbose": False,
                }
                if self._n_threads > 0:
                    kwargs["n_threads"] = self._n_threads
                self._llm = Llama(**kwargs)
                load_s = time.monotonic() - t0
                size_mb = self._gguf_path.stat().st_size // (1024 * 1024)
                logger.info(
                    "Companion llama_cpp: loaded %s (%d MB) in %.1fs (n_ctx=%d, n_threads=%s)",
                    self._gguf_path.name, size_mb, load_s, self._n_ctx,
                    self._n_threads or "auto",
                )
                return True
            except Exception as exc:
                self._load_error = f"llama_load: {type(exc).__name__}: {exc}"
                logger.error(
                    "Companion llama_cpp: failed to load %s: %s",
                    self._gguf_path, exc, exc_info=True,
                )
                return False

    # ------------------------------------------------------------------ generate

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        timeout_s: float,
    ) -> CompanionInferenceResult:
        # The protocol takes a messages list; this adapter only honours
        # the *last* user message. Few-shot history would need a
        # different chat template.
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = str(msg.get("content") or "")
                break

        started = time.monotonic()
        if not self._ensure_loaded():
            return CompanionInferenceResult(
                status="unreachable",
                filter_reason=self._load_error or "load_failed",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                model_id=self.model_id,
            )

        prompt = _LLAMA32_CHAT_TEMPLATE.format(
            system=system_prompt, user=user_content
        )
        # llama-cpp-python is not safe for concurrent __call__; the
        # compute lease guarantees single-holder, but the local lock is
        # cheap insurance for callers outside the service path.
        with self._lock:
            try:
                result = self._llm(
                    prompt,
                    max_tokens=self._n_predict,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    stop=_LLAMA32_STOP,
                )
            except Exception as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.error(
                    "Companion llama_cpp: inference error: %s", exc, exc_info=True,
                )
                return CompanionInferenceResult(
                    status="unreachable",
                    filter_reason=f"inference_error: {type(exc).__name__}",
                    elapsed_ms=elapsed_ms,
                    model_id=self.model_id,
                )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        # Soft timeout: llama-cpp-python does not honour a per-call
        # wallclock; if the run is slower than `timeout_s`, surface that
        # in the diagnostic but still hand the text up — the operator
        # can tighten the limit or pick a smaller model.
        soft_timeout_ms = int(timeout_s * 1000)
        text_raw = ""
        try:
            text_raw = str(
                (result.get("choices") or [{}])[0].get("text") or ""
            ).strip()
        except Exception:
            text_raw = ""

        cleaned = clean_model_text(text_raw)
        if not cleaned:
            return CompanionInferenceResult(
                status="unreachable",
                raw=text_raw,
                filter_reason="empty",
                elapsed_ms=elapsed_ms,
                model_id=self.model_id,
            )
        if elapsed_ms > soft_timeout_ms:
            return CompanionInferenceResult(
                status="timeout",
                text=cleaned,
                raw=text_raw,
                filter_reason=f"slow:{elapsed_ms}ms_over_{soft_timeout_ms}ms",
                elapsed_ms=elapsed_ms,
                model_id=self.model_id,
            )
        return CompanionInferenceResult(
            status="ok",
            text=cleaned,
            raw=text_raw,
            elapsed_ms=elapsed_ms,
            model_id=self.model_id,
        )
