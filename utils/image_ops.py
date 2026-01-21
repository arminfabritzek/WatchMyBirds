import cv2
import numpy as np
from pathlib import Path


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


def generate_preview_thumbnail(original_path: str, output_path: str, size: int = 256) -> bool:
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
        square = image[y1:y1+side, x1:x1+side]
        
        # Resize to target size
        thumb = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WebP
        cv2.imwrite(output_path, thumb, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
        return True
    except Exception:
        return False
