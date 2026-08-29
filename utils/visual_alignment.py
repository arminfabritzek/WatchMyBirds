"""CPU-friendly local-feature alignment for visual PTZ navigation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    """Geometric relationship between a target view and the current view."""

    success: bool
    method: str
    model: str
    reference_keypoints: int
    current_keypoints: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    coverage: float
    reprojection_rmse: float | None
    quality: float
    error_x: float | None
    error_y: float | None
    pixel_dx: float | None
    pixel_dy: float | None
    mapped_x: float | None = None
    mapped_y: float | None = None
    scale: float | None = None
    rotation_deg: float | None = None
    reason: str = ""

    @property
    def error_magnitude(self) -> float | None:
        if self.error_x is None or self.error_y is None:
            return None
        return math.hypot(self.error_x, self.error_y)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["error_magnitude"] = self.error_magnitude
        return payload


class FeatureAligner:
    """Match SIFT/ORB features and estimate a robust image transform."""

    def __init__(
        self,
        *,
        method: str = "auto",
        max_features: int = 2500,
        ratio_threshold: float = 0.75,
        min_matches: int = 12,
        min_inliers: int = 8,
        ransac_threshold_px: float = 4.0,
        max_dimension: int = 1280,
    ) -> None:
        requested = method.strip().lower()
        if requested not in {"auto", "sift", "orb"}:
            raise ValueError("method must be auto, sift, or orb")
        if requested in {"auto", "sift"} and hasattr(cv2, "SIFT_create"):
            self.method = "sift"
            self._detector = cv2.SIFT_create(nfeatures=max_features)
            self._norm = cv2.NORM_L2
        elif requested == "sift":
            raise RuntimeError("This OpenCV build does not provide SIFT")
        else:
            self.method = "orb"
            self._detector = cv2.ORB_create(nfeatures=max_features)
            self._norm = cv2.NORM_HAMMING
            if ratio_threshold == 0.75:
                ratio_threshold = 0.82
        self.ratio_threshold = float(ratio_threshold)
        self.min_matches = max(4, int(min_matches))
        self.min_inliers = max(4, int(min_inliers))
        self.ransac_threshold_px = max(0.5, float(ransac_threshold_px))
        self.max_dimension = max(320, int(max_dimension))

    def align(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        *,
        reference_anchor: tuple[float, float] = (0.5, 0.5),
        debug_path: Path | None = None,
    ) -> AlignmentResult:
        anchor_x, anchor_y = reference_anchor
        if not 0.0 <= anchor_x <= 1.0 or not 0.0 <= anchor_y <= 1.0:
            raise ValueError("reference_anchor coordinates must be between 0 and 1")
        reference_gray, reference_bgr = self._prepare(reference)
        current_gray, current_bgr = self._prepare(current)
        keypoints_ref, descriptors_ref = self._detector.detectAndCompute(
            reference_gray, None
        )
        keypoints_cur, descriptors_cur = self._detector.detectAndCompute(
            current_gray, None
        )
        ref_count = len(keypoints_ref)
        cur_count = len(keypoints_cur)
        if descriptors_ref is None or descriptors_cur is None:
            return self._failure(
                reference_bgr,
                current_bgr,
                ref_count,
                cur_count,
                "not enough image features",
                debug_path,
            )

        matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        pairs = matcher.knnMatch(descriptors_ref, descriptors_cur, k=2)
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.ratio_threshold * second.distance
        ]
        if len(good) < self.min_matches:
            return self._failure(
                reference_bgr,
                current_bgr,
                ref_count,
                cur_count,
                f"only {len(good)} ratio-test matches",
                debug_path,
                good_matches=len(good),
                keypoints_ref=keypoints_ref,
                keypoints_cur=keypoints_cur,
                matches=good,
            )

        points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good])
        points_cur = np.float32([keypoints_cur[m.trainIdx].pt for m in good])
        transform, mask = cv2.findHomography(
            points_ref,
            points_cur,
            cv2.RANSAC,
            self.ransac_threshold_px,
        )
        model = "homography"
        if transform is None or mask is None:
            affine, mask = cv2.estimateAffinePartial2D(
                points_ref,
                points_cur,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold_px,
            )
            if affine is not None:
                transform = np.vstack([affine, [0.0, 0.0, 1.0]])
                model = "affine"
        if transform is None or mask is None:
            return self._failure(
                reference_bgr,
                current_bgr,
                ref_count,
                cur_count,
                "RANSAC could not estimate a stable transform",
                debug_path,
                good_matches=len(good),
                keypoints_ref=keypoints_ref,
                keypoints_cur=keypoints_cur,
                matches=good,
            )

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / len(good)
        coverage = self._coverage(
            points_ref[inlier_mask], reference_gray.shape[1], reference_gray.shape[0]
        )
        rmse = self._reprojection_rmse(
            points_ref[inlier_mask], points_cur[inlier_mask], transform
        )
        quality = self._quality(inliers, inlier_ratio, coverage, rmse)

        ref_h, ref_w = reference_gray.shape[:2]
        cur_h, cur_w = current_gray.shape[:2]
        anchor_ref = np.float32([[[ref_w * anchor_x, ref_h * anchor_y]]])
        mapped_anchor = cv2.perspectiveTransform(anchor_ref, transform)[0, 0]
        dx = float(mapped_anchor[0] - cur_w / 2.0)
        dy = float(mapped_anchor[1] - cur_h / 2.0)
        scale, rotation_deg = self._local_scale_and_rotation(
            transform, ref_w / 2.0, ref_h / 2.0, min(ref_w, ref_h) * 0.1
        )
        finite = np.isfinite(mapped_anchor).all()
        plausible = abs(dx) <= cur_w * 1.5 and abs(dy) <= cur_h * 1.5
        success = inliers >= self.min_inliers and finite and plausible
        reason = "" if success else "transform failed geometry safety checks"
        result = AlignmentResult(
            success=success,
            method=self.method,
            model=model,
            reference_keypoints=ref_count,
            current_keypoints=cur_count,
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            coverage=coverage,
            reprojection_rmse=rmse,
            quality=quality,
            error_x=dx / cur_w if success else None,
            error_y=dy / cur_h if success else None,
            pixel_dx=dx if success else None,
            pixel_dy=dy if success else None,
            mapped_x=float(mapped_anchor[0] / cur_w) if success else None,
            mapped_y=float(mapped_anchor[1] / cur_h) if success else None,
            scale=scale if success else None,
            rotation_deg=rotation_deg if success else None,
            reason=reason,
        )
        self._write_debug(
            reference_bgr,
            current_bgr,
            keypoints_ref,
            keypoints_cur,
            good,
            inlier_mask,
            result,
            debug_path,
        )
        return result

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image is None or image.size == 0:
            raise ValueError("image must not be empty")
        if image.ndim == 2:
            gray = image
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            bgr = image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("image must be grayscale or BGR")
        height, width = gray.shape[:2]
        scale = min(1.0, self.max_dimension / max(height, width))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        return gray, bgr

    @staticmethod
    def _coverage(points: np.ndarray, width: int, height: int) -> float:
        if len(points) < 2:
            return 0.0
        x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
        return min(1.0, (w * h) / max(1.0, width * height))

    @staticmethod
    def _reprojection_rmse(
        source: np.ndarray, target: np.ndarray, transform: np.ndarray
    ) -> float:
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), transform)
        error = projected.reshape(-1, 2) - target
        return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))

    @staticmethod
    def _local_scale_and_rotation(
        transform: np.ndarray, center_x: float, center_y: float, step: float
    ) -> tuple[float, float]:
        samples = np.float32(
            [
                [[center_x, center_y]],
                [[center_x + step, center_y]],
                [[center_x, center_y + step]],
            ]
        ).reshape(1, 3, 2)
        mapped = cv2.perspectiveTransform(samples, transform)[0]
        horizontal = mapped[1] - mapped[0]
        vertical = mapped[2] - mapped[0]
        scale_x = float(np.linalg.norm(horizontal) / step)
        scale_y = float(np.linalg.norm(vertical) / step)
        scale = math.sqrt(max(0.0, scale_x * scale_y))
        rotation_deg = math.degrees(math.atan2(horizontal[1], horizontal[0]))
        return scale, rotation_deg

    @staticmethod
    def _quality(
        inliers: int, inlier_ratio: float, coverage: float, rmse: float
    ) -> float:
        score = (
            0.50 * min(1.0, inlier_ratio)
            + 0.25 * min(1.0, inliers / 40.0)
            + 0.15 * min(1.0, coverage / 0.25)
            + 0.10 * math.exp(-rmse / 5.0)
        )
        return max(0.0, min(1.0, score))

    def _failure(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        ref_count: int,
        cur_count: int,
        reason: str,
        debug_path: Path | None,
        *,
        good_matches: int = 0,
        keypoints_ref: list[Any] | None = None,
        keypoints_cur: list[Any] | None = None,
        matches: list[Any] | None = None,
    ) -> AlignmentResult:
        result = AlignmentResult(
            success=False,
            method=self.method,
            model="none",
            reference_keypoints=ref_count,
            current_keypoints=cur_count,
            good_matches=good_matches,
            inliers=0,
            inlier_ratio=0.0,
            coverage=0.0,
            reprojection_rmse=None,
            quality=0.0,
            error_x=None,
            error_y=None,
            pixel_dx=None,
            pixel_dy=None,
            mapped_x=None,
            mapped_y=None,
            scale=None,
            rotation_deg=None,
            reason=reason,
        )
        self._write_debug(
            reference,
            current,
            keypoints_ref or [],
            keypoints_cur or [],
            matches or [],
            None,
            result,
            debug_path,
        )
        return result

    @staticmethod
    def _write_debug(
        reference: np.ndarray,
        current: np.ndarray,
        keypoints_ref: list[Any],
        keypoints_cur: list[Any],
        matches: list[Any],
        inlier_mask: np.ndarray | None,
        result: AlignmentResult,
        debug_path: Path | None,
    ) -> None:
        if debug_path is None:
            return
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        selected = matches[:100]
        draw_mask = None
        if inlier_mask is not None:
            draw_mask = [int(value) for value in inlier_mask[: len(selected)]]
        canvas = cv2.drawMatches(
            reference,
            keypoints_ref,
            current,
            keypoints_cur,
            selected,
            None,
            matchesMask=draw_mask,
            matchColor=(52, 211, 153),
            singlePointColor=(100, 116, 139),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        if result.mapped_x is not None and result.mapped_y is not None:
            current_x_offset = reference.shape[1]
            target = (
                current_x_offset + round(result.mapped_x * current.shape[1]),
                round(result.mapped_y * current.shape[0]),
            )
            cv2.drawMarker(
                canvas,
                target,
                (0, 191, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
            )
        header = np.full((92, canvas.shape[1], 3), (24, 24, 27), dtype=np.uint8)
        line_1 = (
            f"{result.method.upper()} / {result.model}  quality={result.quality:.3f}  "
            f"inliers={result.inliers}/{result.good_matches}"
        )
        if result.error_x is not None and result.error_y is not None:
            line_2 = (
                f"visual error x={result.error_x:+.4f} y={result.error_y:+.4f}  "
                f"pixels=({result.pixel_dx:+.1f}, {result.pixel_dy:+.1f})  "
                f"scale={result.scale:.4f}"
            )
        else:
            line_2 = result.reason
        cv2.putText(
            header, line_1, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (244, 244, 245), 2
        )
        cv2.putText(
            header, line_2, (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (161, 161, 170), 2
        )
        if not cv2.imwrite(str(debug_path), np.vstack([header, canvas])):
            raise OSError(f"Could not write visual PTZ debug image: {debug_path}")
