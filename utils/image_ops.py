from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

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


# --------------------------------------------------------------------------
# XMP metadata burn-in for download/export copies.
#
# Hand-authored RDF/XMP packet passed to Pillow's ``Image.save(xmp=...)``.
# No IPTC-IIM, no ExifTool, no pyexiv2 — Pillow only — so the RPi/aarch64
# lane gains no new native dependency (plan: metadata-burn-in-export-copies).
# The packet is written into in-memory COPIES served at download; on-disk
# originals are never touched.
# --------------------------------------------------------------------------

_XMP_NS = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "dwc": "http://rs.tdwg.org/dwc/terms/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "wmb": "https://watchmybirds.app/ns/1.0/",
}


def _rdf_bag(terms: list[str]) -> str:
    """Serialise a keyword list as an ``rdf:Bag`` body (one li per term)."""
    items = "".join(
        f"<rdf:li>{_xml_escape(t)}</rdf:li>" for t in terms if t
    )
    return f"<rdf:Bag>{items}</rdf:Bag>"


def build_xmp_packet(metadata) -> str:
    """Build an RDF/XMP packet string from an ``EventMetadata`` envelope.

    Writes (all optional, emitted only when populated):
      - ``dc:subject``      species keywords (rdf:Bag) — the iNat match key
      - ``dc:description``  composed "Common (Scientific)" caption
      - ``xmp:Rating``      0-5 stars (favorite → 5, else max star rating)
      - ``xmp:Label``       "Favorite" when any detection is starred
      - ``xmp:CreatorTool`` / ``dc:creator`` — provenance in standard fields
      - ``dwc:*``           per-species scientific/genus/class/kingdom
                            (class/kingdom gated on birds only)
      - ``wmb:*``           private provenance (schema version, models,
                            review status, favorite flag, per-detection
                            id/confidence/rating)

    Accepts a ``core.event_metadata.EventMetadata``; typed loosely to keep
    this module free of a core import (image_ops is a leaf utility).
    Returns a UTF-8 XMP packet ready for ``Image.save(xmp=...)``.
    """
    ns_decls = " ".join(f'xmlns:{p}="{uri}"' for p, uri in _XMP_NS.items())

    body_parts: list[str] = []

    keywords = metadata.subject_keywords()
    if keywords:
        body_parts.append(f"<dc:subject>{_rdf_bag(keywords)}</dc:subject>")

    caption = metadata.primary_caption()
    if caption:
        body_parts.append(
            f"<dc:description>{_xml_escape(caption)}</dc:description>"
        )

    # Class A — standard image-wide rating/label that foto tools display.
    # A favorite pins the image to 5 stars + a "Favorite" colour label;
    # otherwise the highest star rating on the frame wins.
    rating = metadata.xmp_rating()
    if rating is not None:
        body_parts.append(f"<xmp:Rating>{int(rating)}</xmp:Rating>")
    label = metadata.xmp_label()
    if label:
        body_parts.append(f"<xmp:Label>{_xml_escape(label)}</xmp:Label>")

    # Class B — provenance in standard fields (CreatorTool, dc:creator).
    if metadata.creator_tool:
        body_parts.append(
            f"<xmp:CreatorTool>{_xml_escape(metadata.creator_tool)}"
            f"</xmp:CreatorTool>"
        )
    if metadata.creator:
        body_parts.append(
            f"<dc:creator><rdf:Seq><rdf:li>{_xml_escape(metadata.creator)}"
            f"</rdf:li></rdf:Seq></dc:creator>"
        )

    for entry in metadata.species:
        if entry.scientific:
            body_parts.append(
                f"<dwc:scientificName>{_xml_escape(entry.scientific)}"
                f"</dwc:scientificName>"
            )
        if entry.genus:
            body_parts.append(
                f"<dwc:genus>{_xml_escape(entry.genus)}</dwc:genus>"
            )
        # Aves/Animalia only for birds; squirrels/martens get no dwc class.
        if entry.dwc_class:
            body_parts.append(
                f"<dwc:class>{_xml_escape(entry.dwc_class)}</dwc:class>"
            )
        if entry.dwc_kingdom:
            body_parts.append(
                f"<dwc:kingdom>{_xml_escape(entry.dwc_kingdom)}</dwc:kingdom>"
            )

    body_parts.append(
        f"<wmb:metadataSchemaVersion>{int(metadata.schema_version)}"
        f"</wmb:metadataSchemaVersion>"
    )
    if metadata.detector_model:
        body_parts.append(
            f"<wmb:detectorModel>{_xml_escape(metadata.detector_model)}"
            f"</wmb:detectorModel>"
        )
    if metadata.classifier_model:
        body_parts.append(
            f"<wmb:classifierModel>{_xml_escape(metadata.classifier_model)}"
            f"</wmb:classifierModel>"
        )
    if metadata.review_status:
        body_parts.append(
            f"<wmb:reviewStatus>{_xml_escape(metadata.review_status)}"
            f"</wmb:reviewStatus>"
        )
    if metadata.is_favorite:
        body_parts.append("<wmb:isFavorite>true</wmb:isFavorite>")
    for entry in metadata.species:
        if entry.detection_id is not None:
            body_parts.append(
                f"<wmb:detectionId>{int(entry.detection_id)}"
                f"</wmb:detectionId>"
            )
        if entry.confidence is not None:
            body_parts.append(
                f"<wmb:confidence>{float(entry.confidence):.4f}"
                f"</wmb:confidence>"
            )
        if entry.rating:
            body_parts.append(
                f"<wmb:rating>{int(entry.rating)}</wmb:rating>"
            )

    body = "".join(body_parts)
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" {ns_decls}>'
        f"{body}"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )


def save_jpeg_copy_with_metadata(src_path: str | Path, xmp_packet: str) -> bytes:
    """Return JPEG bytes of ``src_path`` with ``xmp_packet`` injected.

    Reads the on-disk original, preserves its existing EXIF (ingest-time
    DateTimeOriginal + GPS), strips the camera/maker author chrome that is
    irrelevant for an exported copy, injects the XMP packet, and returns the
    re-encoded bytes. **The source file is never written** — burn-in happens
    only into this in-memory copy.

    Raises ``FileNotFoundError`` if the source is missing; lets other Pillow
    errors propagate so the caller can fall back to serving the raw original.
    """
    import io

    from PIL import Image

    src_path = Path(src_path)
    with Image.open(src_path) as img:
        # Preserve ingest-time EXIF (datetime + GPS) verbatim so iNat still
        # reads observation date/location off the copy.
        exif = img.info.get("exif")
        save_kwargs: dict = {
            "format": "JPEG",
            "quality": "keep",
            "xmp": xmp_packet.encode("utf-8"),
        }
        if exif:
            save_kwargs["exif"] = exif

        buffer = io.BytesIO()
        img.save(buffer, **save_kwargs)
        return buffer.getvalue()
