"""Companion factory tests: backend selection + GGUF auto-discovery."""

from __future__ import annotations

import time

from web.services.companion.factory import (
    build_inference_client,
    find_default_gguf,
)
from web.services.companion.llama_cpp_adapter import LlamaCppInferenceAdapter
from web.services.companion.ollama_adapter import OllamaInferenceAdapter


def test_find_default_gguf_returns_none_when_dir_missing(tmp_path):
    assert find_default_gguf(str(tmp_path)) is None


def test_find_default_gguf_returns_none_when_dir_empty(tmp_path):
    (tmp_path / "companion").mkdir()
    assert find_default_gguf(str(tmp_path)) is None


def test_find_default_gguf_picks_newest(tmp_path):
    cdir = tmp_path / "companion"
    cdir.mkdir()
    older = cdir / "v1.gguf"
    newer = cdir / "v2.gguf"
    older.write_bytes(b"x")
    time.sleep(0.02)
    newer.write_bytes(b"x")
    found = find_default_gguf(str(tmp_path))
    assert found is not None
    assert found.name == "v2.gguf"


def test_build_default_backend_is_llama_cpp(tmp_path):
    cdir = tmp_path / "companion"
    cdir.mkdir()
    gguf = cdir / "test.gguf"
    gguf.write_bytes(b"x")
    cli = build_inference_client({}, model_base_path=str(tmp_path))
    assert isinstance(cli, LlamaCppInferenceAdapter)
    assert cli.model_id == "test.gguf"


def test_build_explicit_gguf_path_overrides_discovery(tmp_path):
    cdir = tmp_path / "companion"
    cdir.mkdir()
    auto = cdir / "auto.gguf"
    auto.write_bytes(b"x")
    explicit = tmp_path / "custom.gguf"
    explicit.write_bytes(b"x")
    cli = build_inference_client(
        {"COMPANION_LLAMA_CPP_GGUF_PATH": str(explicit)},
        model_base_path=str(tmp_path),
    )
    assert isinstance(cli, LlamaCppInferenceAdapter)
    assert cli.model_id == "custom.gguf"


def test_build_llama_cpp_without_gguf_still_returns_adapter(tmp_path):
    """Even with no GGUF available, the factory hands back an adapter
    that surfaces the missing file as an 'unreachable' diagnostic on
    the first call. Keeps the API surface predictable."""
    cli = build_inference_client({}, model_base_path=str(tmp_path))
    assert isinstance(cli, LlamaCppInferenceAdapter)
    res = cli.generate(
        system_prompt="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=1.0,
    )
    assert res.status == "unreachable"
    assert "gguf_not_found" in res.filter_reason


def test_build_explicit_ollama_backend(tmp_path):
    cli = build_inference_client(
        {
            "COMPANION_INFERENCE_BACKEND": "ollama",
            "COMPANION_OLLAMA_URL": "http://example.test:11434",
            "COMPANION_OLLAMA_MODEL_TAG": "smoke:test",
        },
        model_base_path=str(tmp_path),
    )
    assert isinstance(cli, OllamaInferenceAdapter)
    assert cli.model_id == "smoke:test"


def test_build_unknown_backend_falls_back_to_llama_cpp(tmp_path):
    cdir = tmp_path / "companion"
    cdir.mkdir()
    (cdir / "x.gguf").write_bytes(b"x")
    cli = build_inference_client(
        {"COMPANION_INFERENCE_BACKEND": "nonsense"},
        model_base_path=str(tmp_path),
    )
    assert isinstance(cli, LlamaCppInferenceAdapter)


def test_n_ctx_and_n_threads_propagate_to_adapter(tmp_path):
    cdir = tmp_path / "companion"
    cdir.mkdir()
    gguf = cdir / "x.gguf"
    gguf.write_bytes(b"x")
    cli = build_inference_client(
        {
            "COMPANION_LLAMA_CPP_N_CTX": 2048,
            "COMPANION_LLAMA_CPP_N_THREADS": 2,
        },
        model_base_path=str(tmp_path),
    )
    assert isinstance(cli, LlamaCppInferenceAdapter)
    assert cli._n_ctx == 2048
    assert cli._n_threads == 2
