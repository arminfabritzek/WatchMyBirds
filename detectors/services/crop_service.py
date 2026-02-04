"""
Crop Service - Image Cropping and Resizing Operations.

Centralizes all crop, resize, and color conversion operations.
Pure image math - no filesystem, database, or network operations.
"""

import cv2
import numpy as np

from utils.image_ops import create_square_crop


class CropService:
    """
    Handles all image cropping and resizing operations.

    Features:
    - Classification crops with margin and padding
    - Edge-shifted thumbnail crops
    - Resize and color conversion utilities

    Constraints:
    - No filesystem operations
    - No database operations
    - No PathManager usage
    - Pure numpy/cv2/image_ops only
    """

    DEFAULT_CLASSIFICATION_SIZE = 512
    DEFAULT_THUMBNAIL_SIZE = 256
    DEFAULT_EXPANSION_PERCENT = 0.10
    DEFAULT_MARGIN_PERCENT = 0.1

    def create_classification_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        size: int = DEFAULT_CLASSIFICATION_SIZE,
        margin_percent: float = DEFAULT_MARGIN_PERCENT,
        to_rgb: bool = True,
    ) -> np.ndarray | None:
        """
        Creates a square crop suitable for species classification.

        Uses centered square crop with margin and optional padding.
        Resizes to target size and converts to RGB if requested.

        Args:
            frame: BGR image (from cv2).
            bbox: Bounding box as (x1, y1, x2, y2) in pixels.
            size: Target output size (square).
            margin_percent: Extra margin around bbox (default 10%).
            to_rgb: Convert from BGR to RGB (default True for classifiers).

        Returns:
            Cropped and resized image as np.ndarray, or None on error.
        """
        try:
            crop = create_square_crop(frame, bbox, margin_percent=margin_percent)

            if crop is None or crop.size == 0:
                return None

            # Resize to target size
            crop_resized = cv2.resize(
                crop,
                (size, size),
                interpolation=cv2.INTER_AREA,
            )

            # Convert BGR to RGB if requested
            if to_rgb:
                return cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            return crop_resized

        except Exception:
            return None

    def create_thumbnail_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        size: int = DEFAULT_THUMBNAIL_SIZE,
        expansion_percent: float = DEFAULT_EXPANSION_PERCENT,
    ) -> np.ndarray | None:
        """
        Creates an edge-shifted square crop for thumbnails.

        Uses edge-shifting instead of padding to maximize visible content.
        When the crop would extend beyond image boundaries, it shifts
        to stay within bounds rather than adding black padding.

        Args:
            frame: BGR image (from cv2).
            bbox: Bounding box as (x1, y1, x2, y2) in pixels.
            size: Target output size (square, default 256).
            expansion_percent: Extra expansion around bbox (default 10%).

        Returns:
            Cropped and resized image as np.ndarray (BGR), or None on error.
        """
        try:
            x1, y1, x2, y2 = bbox
            img_h, img_w = frame.shape[:2]

            # Calculate square side based on bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            side = int(max(bbox_w, bbox_h) * (1 + expansion_percent))

            # Center the square on bbox center
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            sq_x1, sq_y1 = int(cx - side / 2), int(cy - side / 2)
            sq_x2, sq_y2 = sq_x1 + side, sq_y1 + side

            # Edge shift clamping (shift instead of clip)
            if sq_x1 < 0:
                sq_x2 -= sq_x1
                sq_x1 = 0
            if sq_y1 < 0:
                sq_y2 -= sq_y1
                sq_y1 = 0
            if sq_x2 > img_w:
                sq_x1 -= sq_x2 - img_w
                sq_x2 = img_w
            if sq_y2 > img_h:
                sq_y1 -= sq_y2 - img_h
                sq_y2 = img_h

            # Ensure valid crop area
            sq_x1 = max(0, sq_x1)
            sq_y1 = max(0, sq_y1)

            if sq_x2 <= sq_x1 or sq_y2 <= sq_y1:
                return None

            thumb_crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]

            if thumb_crop.size == 0:
                return None

            # Resize to target size
            return cv2.resize(
                thumb_crop,
                (size, size),
                interpolation=cv2.INTER_AREA,
            )

        except Exception:
            return None

    def resize_image(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """
        Resize an image to the given size.

        Args:
            image: Input image.
            size: Target size as (width, height).
            interpolation: cv2 interpolation method (default INTER_AREA).

        Returns:
            Resized image.
        """
        return cv2.resize(image, size, interpolation=interpolation)

    def convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image from BGR to RGB color space.

        Args:
            image: BGR image.

        Returns:
            RGB image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def convert_rgb_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image from RGB to BGR color space.

        Args:
            image: RGB image.

        Returns:
            BGR image.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
