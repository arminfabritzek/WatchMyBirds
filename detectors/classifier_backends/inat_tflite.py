"""Google Coral iNaturalist Birds MobileNet-v2 LiteRT backend."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import requests

from detectors.classifier_backends.base import (
    ClassifierBackend,
    ClassifierBackendPrediction,
)
from logging_config import get_logger
from utils.image_ops import resize_model_input

logger = get_logger(__name__)

INAT_SOURCE_COMMIT = "104342d2d3480b3e66203073dac24f4e2dbb4c41"
INAT_MODEL_ID = f"google-inat-bird-mobilenet-v2-965@{INAT_SOURCE_COMMIT[:8]}"
INAT_MODEL_FILENAME = "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
INAT_LABELS_FILENAME = "inat_bird_labels.txt"
INAT_MODEL_SHA256 = "350fcd8cf1df1560060d464595dfed8b174b05792788052896004848d9ad04f9"
INAT_LABELS_SHA256 = "a16108dfe3f8daff015b87a97ab6a17e717b9b1bccd719f6d8f747746d7b9277"
INAT_SOURCE_BASE_URL = (
    f"https://raw.githubusercontent.com/google-coral/test_data/{INAT_SOURCE_COMMIT}"
)

ArtifactLoader = Callable[[Path], tuple[Path, Path]]
InterpreterFactory = Callable[[Path, int | None], Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_artifact(
    target: Path,
    *,
    url: str,
    expected_sha256: str,
) -> None:
    if target.is_file() and _sha256_file(target) == expected_sha256:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(10, 120),
            headers={"User-Agent": "WatchMyBirds model downloader"},
        ) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".download",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary.write(chunk)

        actual_sha256 = _sha256_file(temp_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {target.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def ensure_inat_model_files(model_dir: Path) -> tuple[Path, Path]:
    """Download the pinned model and labels once, with integrity checks."""

    model_path = model_dir / INAT_MODEL_FILENAME
    labels_path = model_dir / INAT_LABELS_FILENAME
    _ensure_artifact(
        model_path,
        url=f"{INAT_SOURCE_BASE_URL}/{INAT_MODEL_FILENAME}",
        expected_sha256=INAT_MODEL_SHA256,
    )
    _ensure_artifact(
        labels_path,
        url=f"{INAT_SOURCE_BASE_URL}/{INAT_LABELS_FILENAME}",
        expected_sha256=INAT_LABELS_SHA256,
    )
    return model_path, labels_path


def parse_inat_label(label: str) -> tuple[str, str]:
    """Return WMB's species key plus the embedded English common name."""

    stripped = label.strip()
    if stripped.casefold() == "background":
        return "background", "Background"
    match = re.fullmatch(r"(.+?)\s+\((.+)\)", stripped)
    if match is None:
        return stripped.replace(" ", "_"), ""
    scientific_name, common_name = match.groups()
    return scientific_name.strip().replace(" ", "_"), common_name.strip()


def _default_interpreter_factory(model_path: Path, num_threads: int | None) -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeError(
            "The iNaturalist classifier requires ai-edge-litert. "
            "Install the standard WatchMyBirds requirements."
        ) from exc

    kwargs: dict[str, Any] = {"model_path": str(model_path)}
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    return Interpreter(**kwargs)


class INaturalistTFLiteBirdClassifierBackend(ClassifierBackend):
    """Run the 964-bird-taxa Coral model locally with LiteRT on CPU."""

    backend_name = "inat_tflite"

    def __init__(
        self,
        *,
        model_base_path: str | Path,
        num_threads: int | None = None,
        artifact_loader: ArtifactLoader = ensure_inat_model_files,
        interpreter_factory: InterpreterFactory = _default_interpreter_factory,
    ) -> None:
        self.model_dir = Path(model_base_path) / "classifier" / "inat_bird_mobilenet_v2"
        self.num_threads = num_threads if num_threads and num_threads > 0 else None
        self._artifact_loader = artifact_loader
        self._interpreter_factory = interpreter_factory
        self._initialized = False
        self._init_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._interpreter: Any | None = None
        self._input_details: dict[str, Any] | None = None
        self._output_details: dict[str, Any] | None = None
        self.classes: list[str] = []
        self.common_names: list[str] = []

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return

            model_path, labels_path = self._artifact_loader(self.model_dir)
            interpreter = self._interpreter_factory(model_path, self.num_threads)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            self._validate_tensor_contract(input_details, output_details)
            raw_labels = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            output_count = int(output_details["shape"][-1])
            if len(raw_labels) != output_count:
                raise ValueError(
                    f"iNaturalist labels ({len(raw_labels)}) do not match "
                    f"model outputs ({output_count})"
                )

            parsed_labels = [parse_inat_label(label) for label in raw_labels]
            self.classes = [scientific for scientific, _common in parsed_labels]
            self.common_names = [common for _scientific, common in parsed_labels]
            self._interpreter = interpreter
            self._input_details = input_details
            self._output_details = output_details
            self._initialized = True
            try:
                from utils.species_names import clear_species_name_caches

                clear_species_name_caches()
            except Exception as exc:
                logger.debug("Species-name cache refresh skipped: %s", exc)
            logger.info(
                "LiteRT iNaturalist bird classifier loaded: %s (%d outputs)",
                INAT_MODEL_ID,
                output_count,
            )

    @staticmethod
    def _validate_tensor_contract(
        input_details: dict[str, Any],
        output_details: dict[str, Any],
    ) -> None:
        input_shape = tuple(int(value) for value in input_details["shape"])
        output_shape = tuple(int(value) for value in output_details["shape"])
        if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[-1] != 3:
            raise ValueError(f"Unexpected iNaturalist input shape: {input_shape}")
        if np.dtype(input_details["dtype"]) != np.dtype(np.uint8):
            raise ValueError(
                f"Unexpected iNaturalist input dtype: {input_details['dtype']}"
            )
        if len(output_shape) != 2 or output_shape[0] != 1:
            raise ValueError(f"Unexpected iNaturalist output shape: {output_shape}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if self._input_details is None:
            raise RuntimeError("iNaturalist input metadata is unavailable")
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Classifier crop must be an HxWx3 RGB array")

        shape = self._input_details["shape"]
        height, width = int(shape[1]), int(shape[2])
        resized = resize_model_input(image, width=width, height=height)
        if resized.dtype != np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)

        scale, zero_point = self._input_details.get("quantization", (0.0, 0))
        if scale <= 0:
            raise ValueError("iNaturalist input tensor has no quantization scale")

        # Coral's reference preprocessing is (pixel - 128) / 128 followed by
        # quantization. For this pinned model scale=1/128 and zero_point=128,
        # so RGB uint8 pixels pass through exactly. Keep the general formula
        # here so preprocessing remains part of the backend's model contract.
        if np.isclose(scale * 128.0, 1.0) and np.isclose(zero_point, 128):
            quantized = resized
        else:
            quantized_float = (resized.astype(np.float32) - 128.0) / (
                128.0 * scale
            ) + float(zero_point)
            quantized = np.clip(quantized_float, 0, 255).astype(np.uint8)
        return np.expand_dims(quantized, axis=0)

    def _read_probabilities(self, input_tensor: np.ndarray) -> np.ndarray:
        if (
            self._interpreter is None
            or self._input_details is None
            or self._output_details is None
        ):
            raise RuntimeError("iNaturalist interpreter is unavailable")
        with self._inference_lock:
            self._interpreter.set_tensor(self._input_details["index"], input_tensor)
            self._interpreter.invoke()
            raw_output = self._interpreter.get_tensor(self._output_details["index"])[0]

        if np.issubdtype(raw_output.dtype, np.integer):
            scale, zero_point = self._output_details.get("quantization", (0.0, 0))
            if scale <= 0:
                raise ValueError("iNaturalist output tensor has no quantization scale")
            return (raw_output.astype(np.float32) - float(zero_point)) * float(scale)
        return raw_output.astype(np.float32)

    def predict_rgb(
        self,
        image: np.ndarray,
        *,
        top_k: int = 5,
    ) -> ClassifierBackendPrediction:
        self._ensure_initialized()
        probabilities = self._read_probabilities(self._preprocess(image))
        count = max(1, min(int(top_k), len(self.classes)))
        indices = np.argsort(probabilities)[::-1][:count]
        top1_index = int(indices[0])
        top1_class = self.classes[top1_index]
        top1_confidence = float(probabilities[top1_index])
        is_background = top1_class.casefold() == "background"

        return ClassifierBackendPrediction(
            class_name="" if is_background else top1_class,
            confidence=top1_confidence,
            model_id=INAT_MODEL_ID,
            top_k_classes=[self.classes[int(index)] for index in indices],
            top_k_confidences=[float(probabilities[int(index)]) for index in indices],
            decision_level="reject" if is_background else "species",
            raw_species_name=top1_class,
            common_name=self.common_names[top1_index],
        )

    def get_model_id(self) -> str:
        return INAT_MODEL_ID

    def is_ready(self) -> bool:
        return self._initialized
