"""Load ONNX Runtime with its upstream telemetry disabled."""

import os

# ORT 1.31 initializes its uploader during import, before its Python API is usable.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import onnxruntime

onnxruntime.disable_telemetry_events()
