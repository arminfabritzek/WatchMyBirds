from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import config


@pytest.mark.parametrize(
    ("value", "od_expected", "cls_expected"),
    [(None, 2, 1), ("0", 0, 0), ("2", 2, 2), ("-1", 0, 0), ("bad", 2, 1)],
)
def test_thread_settings_load_from_environment_as_boot_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str | None,
    od_expected: int,
    cls_expected: int | None,
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CPU_LIMIT", "3")
    monkeypatch.setattr(config, "load_settings_yaml", lambda _output: {})
    for key in ("OD_NUM_THREADS", "CLS_NUM_THREADS"):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    settings = config._load_config()

    assert settings["OD_NUM_THREADS"] == od_expected
    assert settings["CLS_NUM_THREADS"] == cls_expected
    assert settings["CPU_LIMIT"] == 3
    assert {"OD_NUM_THREADS", "CLS_NUM_THREADS"} <= config.BOOT_KEYS
    assert not {"OD_NUM_THREADS", "CLS_NUM_THREADS"} & config.RUNTIME_KEYS


def test_default_thread_budgets_are_independent_of_cpu_affinity() -> None:
    settings = copy.deepcopy(config.DEFAULTS)
    settings["CPU_LIMIT"] = 3

    config._coerce_config_types(settings)
    config._coerce_config_types(settings)

    assert settings["CLS_NUM_THREADS"] == 1
    assert settings["OD_NUM_THREADS"] == 2
    assert settings["CPU_LIMIT"] == 3
    assert settings["DETECTION_INTERVAL_SECONDS"] == 2.0
    assert settings["MAX_FPS_DETECTION"] == 0.5


def test_pi_overrides_survive_shared_hardware_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CPU_LIMIT", "0")
    monkeypatch.setenv("OD_NUM_THREADS", "4")
    monkeypatch.setenv("CLS_NUM_THREADS", "2")
    monkeypatch.setenv("DETECTION_INTERVAL_SECONDS", "2.0")
    monkeypatch.setattr(
        config,
        "load_settings_yaml",
        lambda _output: {
            "DETECTION_INTERVAL_SECONDS": 1.0,
            "OD_NUM_THREADS": 99,
            "CLS_NUM_THREADS": 99,
        },
    )

    settings = config._load_config()

    assert settings["OD_NUM_THREADS"] == 4
    assert settings["CLS_NUM_THREADS"] == 2
    assert settings["CPU_LIMIT"] == 0
    assert settings["DETECTION_INTERVAL_SECONDS"] == 1.0
    assert settings["MAX_FPS_DETECTION"] == 1.0


@pytest.mark.parametrize(("threads", "expected"), [(None, 1), (0, 0), (2, 2)])
def test_wmb_classifier_uses_cls_budget_with_real_onnx_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, threads: int | None, expected: int
) -> None:
    onnx = pytest.importorskip("onnx")
    from detectors import classifier

    weights, labels = tmp_path / "classifier.onnx", tmp_path / "classes.txt"
    labels.write_text("Parus_major\nCyanistes_caeruleus\n")
    output = onnx.numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32))
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["scores"], value=output)],
        "thread-budget-classifier",
        [
            onnx.helper.make_tensor_value_info(
                "image", onnx.TensorProto.FLOAT, [1, 3, 2, 2]
            )
        ],
        [onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 2])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = 7
    onnx.save(model, str(weights))
    settings = {"MODEL_BASE_PATH": str(tmp_path), "CPU_LIMIT": 3, "OD_NUM_THREADS": 4}
    if threads is not None:
        settings["CLS_NUM_THREADS"] = threads
    monkeypatch.setattr(classifier, "get_config", lambda: settings)
    monkeypatch.setattr(
        classifier, "ensure_model_files", lambda *_args: (str(weights), str(labels))
    )
    monkeypatch.setattr(
        classifier, "load_latest_identifier", lambda _directory: "test-model"
    )
    monkeypatch.setattr(classifier, "load_cls_decision_config", lambda *_args: None)

    backend = classifier.ImageClassifier()
    prediction = backend.predict_from_image(np.zeros((2, 2, 3), dtype=np.uint8))

    options = backend.ort_session.get_session_options()
    assert options.intra_op_num_threads == expected
    if expected:
        assert options.inter_op_num_threads == 1
    assert prediction[2] == "Cyanistes_caeruleus"
