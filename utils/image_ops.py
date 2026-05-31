from pathlib import Path

import cv2
import numpy as np

# Reference resolution used by laplacian_sharpness() to normalise
# scores across crop sizes. Bigger crops have more pixels and
# therefore more raw Laplacian variance even at the same perceptual
# sharpness; downsampling to a fixed long-side makes the score
# comparable across detections of small distant birds and large
# close-ups.
_SHARPNESS_REF_LONG_SIDE = 256


def create_square_crop(image, bbox, margin_percent=0.2, pad_color=(0, 0, 0)):
    """
    Creates a square crop centered on the object defined by bbox, adding padding if necessary
    so that the output is always a full square with the object centered.

    Args:
        image (np.ndarray): The source image.
        bbox (tuple): Bounding box as (x1, y1, x2, y2).
        margin_percent (float): Extra margin percentage to add around the bbox.
        pad_color (tuple): Color for padding (default is black).

    Returns:
        np.ndarray: The square cropped image.
    """
    bx1, by1, bx2, by2 = bbox
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    bbox_width = bx2 - bx1
    bbox_height = by2 - by1
    bbox_side = max(bbox_width, bbox_height)
    new_side = int(bbox_side * (1 + margin_percent))
    desired_x1 = int(cx - new_side / 2)
    desired_y1 = int(cy - new_side / 2)
    desired_x2 = desired_x1 + new_side
    desired_y2 = desired_y1 + new_side
    image_h, image_w = image.shape[:2]
    crop_x1 = max(0, desired_x1)
    crop_y1 = max(0, desired_y1)
    crop_x2 = min(image_w, desired_x2)
    crop_y2 = min(image_h, desired_y2)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    pad_left = crop_x1 - desired_x1
    pad_top = crop_y1 - desired_y1
    pad_right = desired_x2 - crop_x2
    pad_bottom = desired_y2 - crop_y2
    square_crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return square_crop


def generate_preview_thumbnail(
    original_path: str, output_path: str, size: int = 256
) -> bool:
    """
    Generates a center-cropped square preview thumbnail.
    Used for orphan images without detection bounding boxes.

    Args:
        original_path: Absolute path to original image
        output_path: Absolute path to save the thumbnail
        size: Output thumbnail size in pixels (default 256)

    Returns:
        True if successful, False otherwise
    """
    try:
        image = cv2.imread(original_path)
        if image is None:
            return False

        h, w = image.shape[:2]

        # Center crop to square
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        square = image[y1 : y1 + side, x1 : x1 + side]

        # Resize to target size
        thumb = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as WebP
        cv2.imwrite(output_path, thumb, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
        return True
    except Exception:
        return False


def laplacian_sharpness(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on grayscale. Higher = sharper.

    Normalised by downsampling the crop to a fixed long-side
    (`_SHARPNESS_REF_LONG_SIDE` = 256 px) before applying the
    Laplacian. This makes the score comparable across bbox sizes —
    a small distant bird and a large close-up at the same
    perceptual sharpness produce scores within ~15% of each other
    instead of differing by an order of magnitude.

    Args:
        crop_bgr: BGR image as a numpy array (cv2.imread output).
            Can be any size; will be downsampled if larger than the
            reference long-side.

    Returns:
        A non-negative float. Empirical range on bird crops is
        roughly 0–5000; clearly sharp portraits land 800+, motion-
        blurred frames land below 200. The score is *aux signal*,
        not a hard pass/fail threshold — interpretation is up to
        the consumer.

    Edge cases:
        - Empty or zero-size input → returns 0.0.
        - Color crops are converted to grayscale internally.
        - Float vs uint8 input both work; the function casts to
          uint8 grayscale.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    img = crop_bgr
    # Cast to uint8 BEFORE cvtColor; cv2 cvtColor only accepts
    # uint8 / uint16 / float32 — not float64. Doing the cast first
    # also makes the variance reference deterministic.
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert to grayscale if 3+ channels.
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    # Downsample if long side exceeds reference.
    long_side = max(h, w)
    if long_side > _SHARPNESS_REF_LONG_SIDE:
        scale = _SHARPNESS_REF_LONG_SIDE / long_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_brightness(crop_bgr: np.ndarray) -> float:
    """Mean luminance on grayscale, in [0.0, 255.0].

    Companion metric for ``laplacian_sharpness``. A very dark crop
    will always produce low sharpness scores — not because the
    image is blurry, but because there's not enough signal to
    measure. Storing brightness next to sharpness lets downstream
    consumers tell "blur" from "underexposed".

    Args:
        crop_bgr: BGR image as a numpy array.

    Returns:
        Mean grayscale luminance. 0.0 for a black image, ~255.0 for
        a fully white one. Returns 0.0 on empty input.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    img = crop_bgr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.size == 0:
        return 0.0
    return float(np.mean(gray))
