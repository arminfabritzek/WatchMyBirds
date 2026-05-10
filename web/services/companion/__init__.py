"""Companion v1 backend.

Backend-only: inference, persistence, safety guard, and a small Flask
blueprint. UI surfaces are deliberately out of scope here.

The package is structured so the inference adapter is swappable:

- ``inference`` defines the ``CompanionInferenceClient`` protocol and
  the ``CompanionInferenceResult`` dataclass.
- ``ollama_adapter`` is the first concrete adapter shipped because the
  Pi already has Ollama installed; future adapters live next to it.
- ``safety`` runs a content guard on every cleaned model output.
- ``cleaner`` strips ``<think>...</think>`` blocks and trims long
  responses to the v1 length contract.
- ``recorder`` writes utterances to JSONL under ``OUTPUT_DIR``.
- ``service`` is the orchestrator the API blueprint calls into.
"""
