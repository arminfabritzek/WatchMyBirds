"""Backend selector + GGUF auto-discovery for the Companion.

Single place that knows about all available adapters. The Flask
bootstrap calls ``build_inference_client(config, ...)`` and gets back a
ready ``CompanionInferenceClient`` (or ``None`` if the chosen backend
cannot be configured — the API stays mounted, but ``/chat`` returns
``503`` until the operator fixes the config).

Adding a third adapter is one new branch in ``build_inference_client``
plus the adapter file itself; no other Companion module changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .inference import CompanionInferenceClient

logger = logging.getLogger(__name__)


def find_default_gguf(model_base_path: str) -> Path | None:
    """Return the newest .gguf under ``<model_base_path>/companion/`` or None.

    The operator drops a GGUF into that directory; we pick the newest
    one by mtime so a re-quantised drop-in replacement is auto-picked.
    """
    base = Path(model_base_path) / "companion"
    if not base.exists() or not base.is_dir():
        return None
    candidates = sorted(
        (p for p in base.glob("*.gguf") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_inference_client(
    config: dict[str, Any],
    *,
    model_base_path: str,
) -> CompanionInferenceClient | None:
    """Construct the configured inference adapter, or None on misconfig.

    A misconfigured backend (e.g. llama_cpp picked but no GGUF on disk
    and llama-cpp-python not installed) is logged and returns None.
    The Companion service can still be constructed; ``/chat`` then
    returns ``503 unreachable`` instead of crashing.
    """
    backend = str(config.get("COMPANION_INFERENCE_BACKEND") or "llama_cpp").strip().lower()

    if backend == "llama_cpp":
        return _build_llama_cpp(config, model_base_path)
    if backend == "ollama":
        return _build_ollama(config)

    logger.error(
        "Companion: unknown COMPANION_INFERENCE_BACKEND=%r; falling back to llama_cpp.",
        backend,
    )
    return _build_llama_cpp(config, model_base_path)


def _build_llama_cpp(
    config: dict[str, Any], model_base_path: str
) -> CompanionInferenceClient | None:
    from .llama_cpp_adapter import LlamaCppInferenceAdapter

    explicit = str(config.get("COMPANION_LLAMA_CPP_GGUF_PATH") or "").strip()
    gguf_path: Path | None
    if explicit:
        gguf_path = Path(explicit)
    else:
        gguf_path = find_default_gguf(model_base_path)
        if gguf_path is None:
            logger.warning(
                "Companion llama_cpp: no GGUF found under %s/companion/ and "
                "COMPANION_LLAMA_CPP_GGUF_PATH is empty; adapter will report "
                "'unreachable' until a GGUF is provided.",
                model_base_path,
            )
            # Return adapter anyway with a path that does not exist; on
            # first call it surfaces the load error in the diagnostic.
            gguf_path = Path(model_base_path) / "companion" / "missing.gguf"
    return LlamaCppInferenceAdapter(
        gguf_path=str(gguf_path),
        n_ctx=int(config.get("COMPANION_LLAMA_CPP_N_CTX") or 4096),
        n_threads=int(config.get("COMPANION_LLAMA_CPP_N_THREADS") or 0),
    )


def _build_ollama(config: dict[str, Any]) -> CompanionInferenceClient | None:
    from .ollama_adapter import OllamaInferenceAdapter

    return OllamaInferenceAdapter(
        base_url=str(config.get("COMPANION_OLLAMA_URL") or "http://127.0.0.1:11434"),
        model_tag=str(config.get("COMPANION_OLLAMA_MODEL_TAG") or "wmb-companion:1b-q4"),
    )
