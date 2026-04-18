# ------------------------------------------------------------------------------
# Detector Module for Object Detection (Modularized, ONNX Runtime)
# detectors/detector.py
# ------------------------------------------------------------------------------
import json
import os

import cv2
import numpy as np
import onnxruntime

from config import get_config
from logging_config import get_logger
from utils.model_downloader import (
    ensure_model_files,
    load_latest_identifier,
    resolve_active_precision_artefacts,
)

config = get_config()
logger = get_logger(__name__)

HF_BASE_URL = "https://huggingface.co/arminfabritzek/WatchMyBirds-Models/resolve/main/object_detection"

# Local-override env var: when set, skips HF download and loads ONNX + labels
# from <dir>/best.onnx, <dir>/labels.json, optional <dir>/model_metadata.json.
LOCAL_MODEL_ENV_VAR = "WMB_OBJECT_DETECTION_LOCAL_PATH"

# Precision values accepted by the detector loader. fp32 is the reference
# path; int8_qdq walks the QDQ fallback list from latest_models.json until
# one of the ONNX files loads successfully on the host's ORT version.
PRECISION_FP32 = "fp32"
PRECISION_INT8_QDQ = "int8_qdq"
PRECISION_VALUES = (PRECISION_FP32, PRECISION_INT8_QDQ)

# Output-format identifier. The detector only supports YOLOX raw output now;
# legacy FasterRCNN post-NMS models (last-axis 6, 29-species 20250810 release)
# are no longer loadable. Startup auto-cleanup in
# utils.model_downloader.prune_legacy_fasterrcnn_models removes FasterRCNN
# artefacts before the detector initializes so the HF autofetch then pulls
# the current YOLOX latest.
OUTPUT_FORMAT_YOLOX_RAW = "yolox_raw"

# Default inference thresholds (F1-optimal on the original Tiny val split).
# model_metadata.json/inference_thresholds overrides these per variant.
YOLOX_DEFAULT_CONF_THR = 0.15
YOLOX_DEFAULT_IOU_THR = 0.50

# Required class token in labels for a YOLOX-style model. Enforced by the
# Model-Compatibility-Guard so a stale 29-species labels.json (cannot happen
# via HF after the FasterRCNN removal, but could still occur via LOCAL_PATH
# override on a misconfigured developer machine) is rejected loudly rather
# than silently misclassifying every bird as non-bird.
YOLOX_REQUIRED_BIRD_LABEL = "bird"


def _normalize_class_names(raw_labels):
    """Normalize labels payload to dict[str, str].

    Accepts:
      - dict {"0": "bird", ...}     (dict-style labels.json)
      - list ["bird", "squirrel", ...]   (list-style YOLOX labels.json)

    The 29-species FasterRCNN labels.json historically used dict form. The
    dict branch is retained so LOCAL_PATH overrides with dict-form labels
    still parse — the Model-Compatibility-Guard catches the class-set
    mismatch further downstream.
    """
    if isinstance(raw_labels, dict):
        return {str(k): str(v) for k, v in raw_labels.items()}
    if isinstance(raw_labels, list):
        return {str(i): str(v) for i, v in enumerate(raw_labels)}
    raise ValueError(
        f"Unsupported labels.json format: expected dict or list, got {type(raw_labels).__name__}"
    )


def _load_local_model_files(local_dir):
    """Return (weights_path, labels_path, metadata_path_or_None) for LOCAL_PATH override."""
    weights = os.path.join(local_dir, "best.onnx")
    labels = os.path.join(local_dir, "labels.json")
    metadata = os.path.join(local_dir, "model_metadata.json")
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"{LOCAL_MODEL_ENV_VAR} is set to '{local_dir}' but that directory does not exist."
        )
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"{LOCAL_MODEL_ENV_VAR} directory '{local_dir}' is missing 'best.onnx'."
        )
    if not os.path.exists(labels):
        raise FileNotFoundError(
            f"{LOCAL_MODEL_ENV_VAR} directory '{local_dir}' is missing 'labels.json'."
        )
    return weights, labels, metadata if os.path.exists(metadata) else None


def _detect_output_format(session, class_names):
    """Verify the ONNX output matches the YOLOX raw layout.

    YOLOX raw output: 3D tensor with last axis == 5 + len(labels), holding
    [cx, cy, w, h, obj, cls_0..cls_{C-1}] per detection.

    The legacy FasterRCNN post-NMS layout (last axis 6) is no longer
    supported. If such a model is detected we fail loudly with a migration
    hint — the startup cleanup in utils.model_downloader removes legacy
    artefacts before init, so reaching this branch means a manual LOCAL_PATH
    override is pointing at a stale release.
    """
    outputs = session.get_outputs()
    if len(outputs) != 1:
        raise ValueError(
            f"Unsupported ONNX: expected 1 output tensor, got {len(outputs)}."
        )
    shape = outputs[0].shape
    if len(shape) != 3:
        raise ValueError(
            f"Unsupported ONNX output rank {len(shape)} (shape={shape}); expected 3."
        )
    last_dim = shape[-1]
    num_classes = len(class_names)
    if isinstance(last_dim, int) and last_dim == 5 + num_classes:
        return OUTPUT_FORMAT_YOLOX_RAW
    if isinstance(last_dim, int) and last_dim == 6:
        raise ValueError(
            "Legacy FasterRCNN post-NMS model detected (last-axis 6). This "
            "release supports YOLOX raw output only. Remove the legacy "
            "artefacts from the model directory and restart — the autofetch "
            "will pull the current YOLOX latest from HuggingFace."
        )
    raise ValueError(
        f"Unrecognized ONNX output shape {shape} with {num_classes} labels; "
        f"expected last-dim {5 + num_classes} (YOLOX)."
    )


def _assert_yolox_labels_compatible(class_names):
    """Model-Compatibility-Guard.

    The pipeline assumes exactly one OD class named 'bird' and a small number
    of non-bird classes. A stale 29-species labels.json loaded via LOCAL_PATH
    override would silently make every bird class be treated as non-bird
    (CLS skipped, od_class_name -> species) — this guard rejects that
    configuration loudly at load time.
    """
    names = {v for v in class_names.values()}
    if YOLOX_REQUIRED_BIRD_LABEL not in names:
        raise ValueError(
            f"Detector model incompatible: expected YOLOX-style locator with a "
            f"'{YOLOX_REQUIRED_BIRD_LABEL}' class, got {len(names)} labels "
            f"({sorted(names)[:5]}{'...' if len(names) > 5 else ''}) without "
            f"'{YOLOX_REQUIRED_BIRD_LABEL}'. Make sure the deployed model is a "
            f"current YOLOX-locator release, not a legacy 29-species model."
        )


# ------------------------------------------------------------------------------
# Base Detection Model Interface
# ------------------------------------------------------------------------------
class BaseDetectionModel:
    def detect(self, frame, confidence_threshold):
        """
        Perform detection on the provided frame.
        Must return a tuple: (annotated_frame, detection_info_list)
        """
        raise NotImplementedError


# ------------------------------------------------------------------------------
# ONNX Runtime Model Wrapper
# ------------------------------------------------------------------------------
class ONNXDetectionModel(BaseDetectionModel):
    def __init__(self, debug=False):
        """
        Initialize the ONNX Runtime model.
        """
        self.debug = debug

        local_dir = os.environ.get(LOCAL_MODEL_ENV_VAR)
        metadata_path = None
        self.active_precision = PRECISION_FP32
        self.precision_load_path = None
        self.precision_fallback_attempts = []
        if local_dir:
            self.model_path, self.labels_path, metadata_path = _load_local_model_files(
                local_dir
            )
            self.model_id = f"local:{os.path.basename(os.path.normpath(local_dir))}"
            logger.info(
                f"Loading detector from local override path: {local_dir} "
                f"({LOCAL_MODEL_ENV_VAR} set)"
            )
        else:
            model_dir = os.path.join(config["MODEL_BASE_PATH"], "object_detection")
            self.model_path, self.labels_path = ensure_model_files(
                HF_BASE_URL, model_dir, "weights_path_onnx", "labels_path"
            )
            ident = load_latest_identifier(model_dir)
            self.model_id = ident if ident else os.path.basename(self.model_path)
            candidate_metadata = os.path.join(model_dir, "model_metadata.json")
            if os.path.exists(candidate_metadata):
                metadata_path = candidate_metadata

            # If latest_models.json declares active_precision = int8_qdq for
            # this model, swap the fp32 weights for the first QDQ candidate
            # that exists on disk. If none exist or all fail to load, we
            # fall back to fp32 with a loud warning so the service keeps
            # detecting birds while the operator fixes the registry.
            try:
                precision_info = resolve_active_precision_artefacts(model_dir)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"Could not resolve precision artefacts: {exc}. "
                    f"Falling back to fp32."
                )
                precision_info = None
            if precision_info is not None:
                self.active_precision = precision_info["requested_precision"]
                if precision_info["load_candidates"]:
                    # Try each QDQ candidate until one loads. Keep fp32 as
                    # the last-resort safety net.
                    tried = []
                    chosen = None
                    for candidate in precision_info["load_candidates"]:
                        if not os.path.exists(candidate):
                            tried.append((candidate, "missing_on_disk"))
                            continue
                        try:
                            _probe = onnxruntime.InferenceSession(
                                candidate, providers=["CPUExecutionProvider"]
                            )
                        except Exception as exc:
                            tried.append((candidate, f"load_failed: {exc}"))
                            continue
                        # Probe loaded — discard it and keep the path; the
                        # real session is created below to keep the rest of
                        # the init path unchanged.
                        del _probe
                        chosen = candidate
                        break
                    self.precision_fallback_attempts = tried
                    if chosen is not None:
                        self.model_path = chosen
                        self.precision_load_path = chosen
                        logger.info(
                            f"Active precision: {self.active_precision} "
                            f"(loaded {os.path.basename(chosen)})"
                        )
                    else:
                        logger.warning(
                            f"Active precision was requested as "
                            f"{self.active_precision!r} but no candidate "
                            f"loaded on this ORT build: {tried}. Falling "
                            f"back to fp32."
                        )
                        self.active_precision = PRECISION_FP32

        # Initialize ONNX Runtime session.
        try:
            self.session = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(
                f"ONNX model loaded from {self.model_path} using CPUExecutionProvider"
            )
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise

        self.input_size = ONNXDetectionModel.get_model_input_size(self.session)

        # Load class names.
        raw_labels = None
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path) as f:
                    raw_labels = json.load(f)
            except Exception as e:
                logger.error(
                    f"Error loading labels from {self.labels_path}: {e}", exc_info=True
                )
        else:
            logger.warning(
                f"Label JSON file not found at {self.labels_path}. Using default class name."
            )
        self.class_names = _normalize_class_names(raw_labels) if raw_labels is not None else {}

        # Verify ONNX shape + class compatibility. YOLOX raw is the only
        # supported layout after the FasterRCNN removal.
        self.output_format = _detect_output_format(self.session, self.class_names)
        _assert_yolox_labels_compatible(self.class_names)

        # Load thresholds from model_metadata.json (set by the pin endpoint
        # when variants are switched). Falls back to default constants when
        # metadata is absent.
        self.conf_threshold_default = YOLOX_DEFAULT_CONF_THR
        self.iou_threshold_default = YOLOX_DEFAULT_IOU_THR
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                thresholds = meta.get("inference_thresholds", {}) or {}
                self.conf_threshold_default = float(
                    thresholds.get("confidence", self.conf_threshold_default)
                )
                self.iou_threshold_default = float(
                    thresholds.get("iou_nms", self.iou_threshold_default)
                )
                logger.info(
                    f"Loaded thresholds from {metadata_path}: "
                    f"conf={self.conf_threshold_default}, iou={self.iou_threshold_default}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load thresholds from {metadata_path}: {e}. Using defaults."
                )

        logger.info(
            f"Detector output format: {self.output_format} with {len(self.class_names)} labels"
        )

        self.inference_error_count = 0

        # Warm-up (optional, but good practice)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dummy_path = os.path.join(base_dir, "assets", "static_placeholder.jpg")

        # Check if file exists before trying to load (for better error msg)
        if not os.path.exists(dummy_path):
            logger.warning(f"Warm-up image not found at expected path: {dummy_path}")
        else:
            # Try to load
            dummy_image = cv2.imread(dummy_path)
            if dummy_image is not None:
                try:
                    self.detect(dummy_image, 0.5)  # Warm-up call
                    logger.info(f"Model warm-up successful using {dummy_path}")
                except Exception as e:
                    logger.error(f"Model warm-up failed: {e}", exc_info=True)
            else:
                logger.warning(
                    f"cv2.imread failed to load image at {dummy_path} (File exists, size: {os.path.getsize(dummy_path)} bytes)"
                )

    @staticmethod
    def get_model_input_size(session):
        """Gets the model's expected input size from the ONNX session."""
        input_shape = session.get_inputs()[0].shape
        return (input_shape[2], input_shape[3])

    # --------------------------------------------------------------------
    # YOLOX preprocess / postprocess
    # --------------------------------------------------------------------
    def preprocess_image_yolox(self, img):
        """Letterbox-resize to (input_size, input_size), BGR, float32, no /255.

        Pads top-left with 114 (matches YOLOX training pipeline and the
        pipeline-repo reference implementation).
        """
        input_height, input_width = self.input_size
        h, w = img.shape[:2]
        scale = min(input_width / w, input_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized  # top-left padding
        chw = padded.transpose(2, 0, 1).astype(np.float32)  # BGR -> CHW
        return chw[None, ...], img, scale

    def postprocess_output_yolox(
        self,
        raw,
        scale,
        original_width,
        original_height,
        conf_threshold,
        iou_threshold,
    ):
        """Decode YOLOX raw output [1, N, 5+C] into detection dicts.

        Assumes decode_in_inference=True at export time, so xywh is already in
        input-pixel coordinates (no grid/stride decoding needed).
        """
        # raw: (1, N, 5+C) or (N, 5+C)
        if raw.ndim == 3:
            raw = raw[0]
        boxes_xywh = raw[:, 0:4]
        obj_conf = raw[:, 4:5]
        cls_probs = raw[:, 5:]
        scores = obj_conf * cls_probs  # (N, C)
        cls_ids = scores.argmax(axis=1)
        cls_scores = scores.max(axis=1)

        keep = cls_scores > conf_threshold
        if not np.any(keep):
            return []
        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        cls_scores = cls_scores[keep]

        # xywh (centre) -> xyxy
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Per-class NMS via cv2.dnn.NMSBoxes (expects xywh with top-left origin).
        # Offset boxes per-class so NMS effectively runs per-class.
        max_coord = max(original_width, original_height, *self.input_size) + 1
        offsets = cls_ids.astype(np.float32) * float(max_coord)
        boxes_for_nms = boxes_xyxy.copy()
        boxes_for_nms[:, 0] += offsets
        boxes_for_nms[:, 1] += offsets
        boxes_for_nms[:, 2] += offsets
        boxes_for_nms[:, 3] += offsets
        nms_xywh = np.stack(
            [
                boxes_for_nms[:, 0],
                boxes_for_nms[:, 1],
                boxes_for_nms[:, 2] - boxes_for_nms[:, 0],
                boxes_for_nms[:, 3] - boxes_for_nms[:, 1],
            ],
            axis=1,
        ).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_xywh, cls_scores.tolist(), conf_threshold, iou_threshold
        )
        if isinstance(indices, tuple) or len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        detections = []
        for i in indices:
            x1_orig = float(boxes_xyxy[i, 0]) / scale
            y1_orig = float(boxes_xyxy[i, 1]) / scale
            x2_orig = float(boxes_xyxy[i, 2]) / scale
            y2_orig = float(boxes_xyxy[i, 3]) / scale
            # Clip and order
            x1_clip = int(max(0.0, min(original_width, min(x1_orig, x2_orig))))
            y1_clip = int(max(0.0, min(original_height, min(y1_orig, y2_orig))))
            x2_clip = int(max(0.0, min(original_width, max(x1_orig, x2_orig))))
            y2_clip = int(max(0.0, min(original_height, max(y1_orig, y2_orig))))
            detections.append(
                {
                    "class": int(cls_ids[i]),
                    "confidence": float(cls_scores[i]),
                    "x1": x1_clip,
                    "y1": y1_clip,
                    "x2": x2_clip,
                    "y2": y2_clip,
                }
            )
        return detections

    # --------------------------------------------------------------------
    # YOLOX detect()
    # --------------------------------------------------------------------
    def detect(self, frame, confidence_threshold=None):
        """Run YOLOX inference on *frame*.

        ``confidence_threshold`` is accepted for backwards compatibility with
        callers that still pass a value, but it is IGNORED. The active
        threshold always comes from the model's metadata
        (``self.conf_threshold_default``, loaded from
        ``model_metadata.json``). Per-variant calibration is authoritative;
        the previous ``max(user, model)`` hybrid silently ceiling'd Tiny
        (0.15) and S (0.30) with the old default 0.65 and destroyed the
        calibrated operating point.
        """
        del confidence_threshold  # unused, kept in signature for back-compat
        detection_info_list = []
        try:
            processed_image, original_image, scale = self.preprocess_image_yolox(frame)
            original_height, original_width = original_image.shape[:2]
            outputs = self.session.run(None, {self.input_name: processed_image})
            detections = self.postprocess_output_yolox(
                outputs[0],
                scale,
                original_width,
                original_height,
                self.conf_threshold_default,
                self.iou_threshold_default,
            )

            self.inference_error_count = 0

            for detection in detections:
                class_id = detection["class"]
                label = self.class_names.get(str(int(class_id)), "unknown")
                detection_info_list.append(
                    {
                        "class_name": label,
                        "confidence": detection["confidence"],
                        "x1": int(detection["x1"]),
                        "y1": int(detection["y1"]),
                        "x2": int(detection["x2"]),
                        "y2": int(detection["y2"]),
                    }
                )

        except Exception as e:
            self.inference_error_count += 1
            logger.debug(
                f"Error during ONNX inference: {e} (Error count: {self.inference_error_count})"
            )
            if self.inference_error_count >= 3:
                logger.error(
                    "Persistent inference errors encountered. Consider restarting the application."
                )
                return []
            return []

        return detection_info_list

    def exhaustive_detect(self, frame):
        """
        Performs an exhaustive detection using tiling and low thresholds.
        Returns a list of all detections mapped back to the original frame.
        """
        logger.info("Starting exhaustive deep scan (Full + Tiled 0.1)...")
        all_detections = []
        low_conf = 0.1

        # 1. Full frame scan with low confidence
        full_dets = self.detect(frame, low_conf)
        for d in full_dets:
            d["method"] = "full"
        all_detections.extend(full_dets)

        # 2. Tiling (2x2 with overlap)
        h, w = frame.shape[:2]
        # Define 4 tiles with overlap
        # overlap ~20%
        mid_x = w // 2
        mid_y = h // 2

        # Coordinates: x1, y1, x2, y2
        tiles = [
            (0, 0, mid_x + 100, mid_y + 100),  # TL
            (mid_x - 100, 0, w, mid_y + 100),  # TR
            (0, mid_y - 100, mid_x + 100, h),  # BL
            (mid_x - 100, mid_y - 100, w, h),  # BR
        ]

        for tx1, ty1, tx2, ty2 in tiles:
            # Clip to image bounds
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)

            tile_img = frame[ty1:ty2, tx1:tx2]
            if tile_img.size == 0:
                continue

            tile_dets = self.detect(tile_img, low_conf)

            # Map back to original coordinates
            for d in tile_dets:
                d["x1"] += tx1
                d["y1"] += ty1
                d["x2"] += tx1
                d["y2"] += ty1
                d["method"] = "tiled"
                all_detections.append(d)

        # 3. Simple NMS (Non-Maximum Suppression) to remove duplicates
        # We prefer 'tiled' detections if they have higher confidence, but 'full' gives better context.
        # Simple approach: sort by confidence, check IoU.

        keep = []
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        for current in all_detections:
            is_new = True
            cx1, cy1, cx2, cy2 = (
                current["x1"],
                current["y1"],
                current["x2"],
                current["y2"],
            )
            current_area = (cx2 - cx1) * (cy2 - cy1)

            for kept in keep:
                kx1, ky1, kx2, ky2 = kept["x1"], kept["y1"], kept["x2"], kept["y2"]

                # Intersection
                ix1 = max(cx1, kx1)
                iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2)
                iy2 = min(cy2, ky2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    kept_area = (kx2 - kx1) * (ky2 - ky1)
                    union_area = current_area + kept_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > 0.5:  # 50% overlap considered same object
                        is_new = False
                        break

            if is_new:
                keep.append(current)

        logger.info(
            f"Exhaustive scan complete. Found {len(keep)} objects (merged from {len(all_detections)})."
        )
        return keep


# ------------------------------------------------------------------------------
# Detector Class (Modularized)
# ------------------------------------------------------------------------------
class Detector:
    def __init__(self, model_choice="yolo", debug=False):
        """
        Loads the detection model.
        """
        self.debug = debug
        self.model_choice = model_choice.lower()
        if self.model_choice == "yolo":
            self.model = ONNXDetectionModel(debug=debug)
            self.model_id = getattr(self.model, "model_id", "")
        else:
            raise ValueError(f"Unsupported model choice: {self.model_choice}")

    def detect_objects(self, frame, confidence_threshold=0.5, save_threshold=0.8):
        """
        Runs object detection on a frame.
        Returns a tuple: (object_detected, original_frame, detection_info_list)
        """
        original_frame = frame.copy()
        detection_info_list = self.model.detect(frame, confidence_threshold)
        object_detected = any(
            det["confidence"] >= save_threshold for det in detection_info_list
        )
        return object_detected, original_frame, detection_info_list

    def exhaustive_detect(self, frame):
        """Delegates exhaustive detection to the model."""
        return self.model.exhaustive_detect(frame)
