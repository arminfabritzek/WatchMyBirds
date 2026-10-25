"""Disable upstream telemetry before ONNX Runtime initializes native code."""

from __future__ import annotations

import ast
import builtins
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "utils/onnx_runtime.py"


def test_onnx_telemetry_is_disabled_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    runtime = SimpleNamespace(
        disable_telemetry_events=lambda: events.append("api-disabled")
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            events.append(f"import:{os.environ['ORT_DISABLE_TELEMETRY']}")
            return runtime
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    namespace = runpy.run_path(str(RUNTIME))

    assert events == ["import:1", "api-disabled"]
    assert namespace["onnxruntime"] is runtime


def test_application_onnx_imports_use_privacy_policy() -> None:
    bypasses = []
    for directory in ["core", "detectors", "scripts", "utils"]:
        for source in (ROOT / directory).rglob("*.py"):
            if source == RUNTIME:
                continue
            for node in ast.walk(ast.parse(source.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if any(name.split(".")[0] == "onnxruntime" for name in names):
                    bypasses.append(str(source.relative_to(ROOT)))
    assert not bypasses, f"Use utils.onnx_runtime to disable telemetry: {bypasses}"
