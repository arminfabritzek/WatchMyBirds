"""LlamaCppInferenceAdapter unit tests.

The real ``llama_cpp.Llama`` class is heavy (binary load, ~770 MB
GGUF). These tests inject a stub Llama into ``llama_cpp`` so we can
exercise the adapter's load/generate/cleanup paths in milliseconds.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from web.services.companion.llama_cpp_adapter import LlamaCppInferenceAdapter


class _FakeLlama:
    """Stand-in for llama_cpp.Llama with deterministic outputs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return {
            "choices": [
                {"text": "Speaker A chirps briefly and accepts the situation."}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9},
        }


def _install_fake_llama_cpp(monkeypatch, llama_cls=_FakeLlama):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = llama_cls
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return fake


def _make_gguf(tmp_path: Path) -> Path:
    """Create a tiny stub GGUF file the adapter can stat()."""
    p = tmp_path / "stub.gguf"
    p.write_bytes(b"\x00" * 1024)
    return p


def test_generate_happy_path(monkeypatch, tmp_path):
    _install_fake_llama_cpp(monkeypatch)
    gguf = _make_gguf(tmp_path)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(gguf))

    res = adapter.generate(
        system_prompt="be brief",
        messages=[{"role": "user", "content": "Hallo"}],
        timeout_s=5.0,
    )
    assert res.status == "ok"
    assert "Speaker A" in res.text
    assert res.model_id == "stub.gguf"
    assert res.elapsed_ms >= 0


def test_missing_gguf_returns_unreachable(monkeypatch, tmp_path):
    _install_fake_llama_cpp(monkeypatch)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(tmp_path / "missing.gguf"))
    res = adapter.generate(
        system_prompt="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=1.0,
    )
    assert res.status == "unreachable"
    assert res.filter_reason.startswith("gguf_not_found")


def test_llama_cpp_not_installed_returns_unreachable(monkeypatch, tmp_path):
    """Without llama_cpp module, generate must surface a clean diagnostic."""
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    # Block re-import: the adapter does `from llama_cpp import Llama`
    # which will raise ImportError when the module truly isn't there.
    # We achieve that by inserting a finder that explicitly fails.
    import importlib.abc
    import importlib.machinery

    class _BlockLlamaCpp(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "llama_cpp":
                raise ImportError("blocked for test")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockLlamaCpp(), *sys.meta_path])
    gguf = _make_gguf(tmp_path)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(gguf))
    res = adapter.generate(
        system_prompt="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=1.0,
    )
    assert res.status == "unreachable"
    assert res.filter_reason.startswith("llama_cpp_import")


def test_inference_error_is_caught(monkeypatch, tmp_path):
    class _ExplodingLlama(_FakeLlama):
        def __call__(self, prompt, **kwargs):
            raise RuntimeError("kernel oof")

    _install_fake_llama_cpp(monkeypatch, llama_cls=_ExplodingLlama)
    gguf = _make_gguf(tmp_path)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(gguf))
    res = adapter.generate(
        system_prompt="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=1.0,
    )
    assert res.status == "unreachable"
    assert res.filter_reason.startswith("inference_error")


def test_empty_output_returns_unreachable(monkeypatch, tmp_path):
    class _EmptyLlama(_FakeLlama):
        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "   "}], "usage": {}}

    _install_fake_llama_cpp(monkeypatch, llama_cls=_EmptyLlama)
    gguf = _make_gguf(tmp_path)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(gguf))
    res = adapter.generate(
        system_prompt="x",
        messages=[{"role": "user", "content": "hi"}],
        timeout_s=1.0,
    )
    assert res.status == "unreachable"
    assert res.filter_reason == "empty"


def test_load_happens_only_once(monkeypatch, tmp_path):
    """Persistent: the GGUF is loaded on first generate, reused after."""
    instances: list[_FakeLlama] = []

    class _CountingLlama(_FakeLlama):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

    _install_fake_llama_cpp(monkeypatch, llama_cls=_CountingLlama)
    gguf = _make_gguf(tmp_path)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(gguf))
    for _ in range(3):
        res = adapter.generate(
            system_prompt="x",
            messages=[{"role": "user", "content": "hi"}],
            timeout_s=1.0,
        )
        assert res.status == "ok"
    assert len(instances) == 1


def test_chat_template_carries_system_and_user(monkeypatch, tmp_path):
    seen = {}

    class _CapturingLlama(_FakeLlama):
        def __call__(self, prompt, **kwargs):
            seen["prompt"] = prompt
            return super().__call__(prompt, **kwargs)

    _install_fake_llama_cpp(monkeypatch, llama_cls=_CapturingLlama)
    gguf = _make_gguf(tmp_path)
    adapter = LlamaCppInferenceAdapter(gguf_path=str(gguf))
    adapter.generate(
        system_prompt="SYS_TEXT",
        messages=[{"role": "user", "content": "USER_TEXT"}],
        timeout_s=1.0,
    )
    assert "SYS_TEXT" in seen["prompt"]
    assert "USER_TEXT" in seen["prompt"]
    # BOS is added by the Llama tokenizer itself; the template must NOT
    # include it to avoid duplicate-BOS warnings from llama-cpp-python.
    assert "<|begin_of_text|>" not in seen["prompt"]
    assert "<|start_header_id|>system<|end_header_id|>" in seen["prompt"]


def test_threads_kwarg_passed_when_positive(monkeypatch, tmp_path):
    """n_threads=0 must NOT pass the kwarg (let library default); >0 passes."""
    captured = {}

    class _Capture(_FakeLlama):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    _install_fake_llama_cpp(monkeypatch, llama_cls=_Capture)
    gguf = _make_gguf(tmp_path)

    adapter1 = LlamaCppInferenceAdapter(gguf_path=str(gguf), n_threads=0)
    adapter1.generate(system_prompt="x", messages=[{"role": "user", "content": "x"}], timeout_s=1.0)
    assert "n_threads" not in captured

    captured.clear()
    adapter2 = LlamaCppInferenceAdapter(gguf_path=str(gguf), n_threads=3)
    adapter2.generate(system_prompt="x", messages=[{"role": "user", "content": "x"}], timeout_s=1.0)
    assert captured.get("n_threads") == 3
