from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from detectors.classifier_backends.inat_tflite import (
    INAT_MODEL_ID,
    INaturalistTFLiteBirdClassifierBackend,
    _ensure_artifact,
    parse_inat_label,
)


class _FakeInterpreter:
    def __init__(self, output: np.ndarray) -> None:
        self._output = output
        self.input_tensor: np.ndarray | None = None

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self) -> list[dict]:
        return [
            {
                "name": "image",
                "index": 0,
                "shape": np.array([1, 224, 224, 3]),
                "dtype": np.uint8,
                "quantization": (0.0078125, 128),
            }
        ]

    def get_output_details(self) -> list[dict]:
        return [
            {
                "name": "prediction",
                "index": 1,
                "shape": np.array([1, 2]),
                "dtype": np.uint8,
                "quantization": (0.00390625, 0),
            }
        ]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        assert index == 0
        self.input_tensor = value.copy()

    def invoke(self) -> None:
        return None

    def get_tensor(self, index: int) -> np.ndarray:
        assert index == 1
        return self._output


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self._body[:chunk_size]
        yield self._body[chunk_size:]


def _backend(
    tmp_path: Path,
    *,
    output: np.ndarray,
) -> tuple[INaturalistTFLiteBirdClassifierBackend, _FakeInterpreter]:
    model_path = tmp_path / "model.tflite"
    labels_path = tmp_path / "labels.txt"
    model_path.write_bytes(b"fake-tflite")
    labels_path.write_text(
        "Cyanocitta cristata (Blue Jay)\nbackground\n",
        encoding="utf-8",
    )
    interpreter = _FakeInterpreter(output)
    backend = INaturalistTFLiteBirdClassifierBackend(
        model_base_path=tmp_path,
        artifact_loader=lambda _model_dir: (model_path, labels_path),
        interpreter_factory=lambda _path, _threads: interpreter,
    )
    return backend, interpreter


def test_parse_inat_label_separates_scientific_and_common_name() -> None:
    assert parse_inat_label("Cyanocitta cristata (Blue Jay)") == (
        "Cyanocitta_cristata",
        "Blue Jay",
    )
    assert parse_inat_label("background") == ("background", "Background")


def test_inat_backend_exposes_pinned_model_id_before_lazy_initialization(
    tmp_path: Path,
) -> None:
    backend = INaturalistTFLiteBirdClassifierBackend(model_base_path=tmp_path)

    assert backend.get_model_id() == INAT_MODEL_ID
    assert backend.is_ready() is False


def test_artifact_download_is_checksum_verified_and_atomic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from detectors.classifier_backends import inat_tflite

    body = b"verified model bytes"
    monkeypatch.setattr(
        inat_tflite.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(body),
    )
    target = tmp_path / "model.tflite"

    _ensure_artifact(
        target,
        url="https://example.invalid/model.tflite",
        expected_sha256=hashlib.sha256(body).hexdigest(),
    )

    assert target.read_bytes() == body
    assert list(tmp_path.glob("*.download")) == []


def test_artifact_download_rejects_bad_checksum(tmp_path: Path, monkeypatch) -> None:
    from detectors.classifier_backends import inat_tflite

    monkeypatch.setattr(
        inat_tflite.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(b"tampered"),
    )
    target = tmp_path / "labels.txt"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        _ensure_artifact(
            target,
            url="https://example.invalid/labels.txt",
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )

    assert not target.exists()
    assert list(tmp_path.glob("*.download")) == []


def test_inat_backend_uses_uint8_nhwc_rgb_without_imagenet_normalization(
    tmp_path: Path,
) -> None:
    backend, interpreter = _backend(
        tmp_path,
        output=np.array([[200, 20]], dtype=np.uint8),
    )
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 11
    rgb[..., 1] = 22
    rgb[..., 2] = 33

    prediction = backend.predict_rgb(rgb, top_k=2)

    assert interpreter.input_tensor is not None
    assert interpreter.input_tensor.shape == (1, 224, 224, 3)
    assert interpreter.input_tensor.dtype == np.uint8
    assert interpreter.input_tensor[0, 0, 0].tolist() == [11, 22, 33]
    assert prediction.class_name == "Cyanocitta_cristata"
    assert prediction.common_name == "Blue Jay"
    assert prediction.confidence == pytest.approx(200 / 256)
    assert prediction.model_id == INAT_MODEL_ID
    assert prediction.decision_level == "species"


def test_inat_backend_rejects_background_top1(tmp_path: Path) -> None:
    backend, _interpreter = _backend(
        tmp_path,
        output=np.array([[20, 220]], dtype=np.uint8),
    )

    prediction = backend.predict_rgb(
        np.zeros((224, 224, 3), dtype=np.uint8),
        top_k=2,
    )

    assert prediction.class_name == ""
    assert prediction.raw_species_name == "background"
    assert prediction.decision_level == "reject"
    assert prediction.top_k_classes == ["background", "Cyanocitta_cristata"]


def test_inat_backend_rejects_label_output_count_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.tflite"
    labels_path = tmp_path / "labels.txt"
    model_path.write_bytes(b"fake-tflite")
    labels_path.write_text("only one label\n", encoding="utf-8")
    interpreter = _FakeInterpreter(np.array([[1, 2]], dtype=np.uint8))
    backend = INaturalistTFLiteBirdClassifierBackend(
        model_base_path=tmp_path,
        artifact_loader=lambda _model_dir: (model_path, labels_path),
        interpreter_factory=lambda _path, _threads: interpreter,
    )

    with pytest.raises(ValueError, match="labels.*outputs"):
        backend.predict_rgb(np.zeros((12, 12, 3), dtype=np.uint8))
