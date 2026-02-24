"""
Persistence Service - Image and Detection Storage.

Implements PersistenceInterface for saving images, thumbnails, and database records.
Extracts persistence logic from DetectionManager for independent operation.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import cv2

from config import get_config
from detectors.interfaces.persistence import (
    DetectionData,
    DetectionPersistenceResult,
    ImagePersistenceResult,
    PersistenceInterface,
)
from detectors.services.crop_service import CropService
from logging_config import get_logger
from utils.db import (
    get_connection,
    insert_classification,
    insert_detection,
    insert_image,
)
from utils.path_manager import get_path_manager

logger = get_logger(__name__)


def add_exif_metadata(
    image_path: str, capture_time: datetime, location_config: dict | None = None
) -> None:
    """
    Adds DateTimeOriginal and optional GPS EXIF data to an image file.

    Args:
        image_path: Path to the saved JPEG image.
        capture_time: Datetime representing capture time.
        location_config: Optional dict with 'latitude' and 'longitude'.
    """
    try:
        import piexif

        # Format DateTime for EXIF
        exif_dt_str = capture_time.strftime("%Y:%m:%d %H:%M:%S")

        # Prepare EXIF dictionary
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_dt_str
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_dt_str

        # Add GPS Data if available and valid
        if (
            isinstance(location_config, dict)
            and "latitude" in location_config
            and "longitude" in location_config
        ):
            try:
                lat = float(location_config["latitude"])
                lon = float(location_config["longitude"])

                def degrees_to_dms_rational(degrees_float):
                    degrees_float = abs(degrees_float)
                    d = int(degrees_float)
                    m_float = (degrees_float - d) * 60
                    m = int(m_float)
                    s_int = max(0, int((m_float - m) * 60 * 1000))
                    return [(d, 1), (m, 1), (s_int, 1000)]

                gps_lat = degrees_to_dms_rational(lat)
                gps_lon = degrees_to_dms_rational(lon)

                exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = (
                    "N" if lat >= 0 else "S"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = gps_lat
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = (
                    "E" if lon >= 0 else "W"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = gps_lon

                utc_now = datetime.now(UTC)
                exif_dict["GPS"][piexif.GPSIFD.GPSDateStamp] = utc_now.strftime(
                    "%Y:%m:%d"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSTimeStamp] = [
                    (utc_now.hour, 1),
                    (utc_now.minute, 1),
                    (max(0, int(utc_now.second * 1000)), 1000),
                ]
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse GPS data: {e}")

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        logger.debug(f"Added EXIF data to {os.path.basename(image_path)}")

    except FileNotFoundError:
        logger.error(f"EXIF Error: Image not found at {image_path}")
    except Exception as e:
        logger.error(f"EXIF Error: {e}", exc_info=True)


class PersistenceService(PersistenceInterface):
    """
    Handles persistence of images and detection records.

    Features:
    - Saves original JPEG with EXIF metadata
    - Generates optimized WebP derivatives
    - Creates square thumbnails for detections
    - Persists detection and classification records to database
    """

    def __init__(self, output_dir: str | None = None):
        """
        Initialize the persistence service.

        Args:
            output_dir: Base output directory. Uses config if not provided.
        """
        self._config = get_config()
        self._output_dir = output_dir or self._config["OUTPUT_DIR"]
        self._path_mgr = get_path_manager(self._output_dir)
        self._db_conn = get_connection()
        self._crop_service = CropService()

    def close(self) -> None:
        """Close owned resources."""
        try:
            if self._db_conn:
                self._db_conn.close()
        except Exception as e:
            logger.debug(f"Failed to close PersistenceService DB connection: {e}")
        finally:
            self._db_conn = None

    def __del__(self):
        """Best-effort cleanup for interpreter shutdown."""
        try:
            self.close()
        except Exception:
            pass

    def save_image(
        self,
        frame,  # np.ndarray
        capture_time: datetime,
        detector_model_id: str,
        classifier_model_id: str,
        source_id: int,
        location_config: dict | None = None,
        exif_gps_enabled: bool = True,
    ) -> ImagePersistenceResult:
        """
        Saves original and optimized versions of an image.

        Args:
            frame: BGR image to save.
            capture_time: When the image was captured.
            detector_model_id: ID of the detection model.
            classifier_model_id: ID of the classification model.
            source_id: Database ID of the video source.
            location_config: Optional GPS coordinates for EXIF.
            exif_gps_enabled: Whether to include GPS in EXIF.

        Returns:
            ImagePersistenceResult with paths and success status.
        """
        # Generate filenames
        timestamp_str = capture_time.strftime("%Y%m%d_%H%M%S_%f")
        base_filename = f"{timestamp_str}.jpg"
        date_str = capture_time.strftime("%Y-%m-%d")

        # Ensure directory structure
        self._path_mgr.ensure_date_structure(date_str)

        # Get paths
        original_path = self._path_mgr.get_original_path(base_filename)
        optimized_path = self._path_mgr.get_derivative_path(base_filename, "optimized")

        try:
            # Save original JPEG
            if not cv2.imwrite(str(original_path), frame):
                logger.error(f"Failed to save original image: {original_path}")
                return ImagePersistenceResult(success=False)

            # Add EXIF metadata
            if exif_gps_enabled and location_config:
                add_exif_metadata(str(original_path), capture_time, location_config)
            else:
                add_exif_metadata(str(original_path), capture_time, None)

            # Save optimized WebP (resize if large)
            h, w = frame.shape[:2]
            if w > 1920:
                scale = 1920 / w
                new_h = int(h * scale)
                optimized_frame = cv2.resize(frame, (1920, new_h))
            else:
                optimized_frame = frame

            cv2.imwrite(
                str(optimized_path),
                optimized_frame,
                [int(cv2.IMWRITE_WEBP_QUALITY), 80],
            )

            # Insert database record
            insert_image(
                self._db_conn,
                {
                    "filename": base_filename,
                    "timestamp": timestamp_str,
                    "coco_json": None,
                    "downloaded_timestamp": "",
                    "detector_model_id": detector_model_id,
                    "classifier_model_id": classifier_model_id,
                    "source_id": source_id,
                    "content_hash": None,
                },
            )

            return ImagePersistenceResult(
                success=True,
                original_path=original_path,
                optimized_path=optimized_path,
                base_filename=base_filename,
                date_str=date_str,
            )

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return ImagePersistenceResult(success=False)

    def save_detection(
        self,
        image_filename: str,
        detection: DetectionData,
        frame,  # np.ndarray
        detector_model_id: str,
        classifier_model_id: str,
        crop_index: int,
    ) -> DetectionPersistenceResult:
        """
        Saves a detection with thumbnail and database record.

        Args:
            image_filename: Filename of the parent image.
            detection: Detection data to persist.
            frame: Original frame for thumbnail generation.
            detector_model_id: ID of the detection model.
            classifier_model_id: ID of the classification model.
            crop_index: Index of this detection (for thumbnail naming).

        Returns:
            DetectionPersistenceResult with database ID and paths.
        """
        # Generate thumbnail filename
        base_name_no_ext = os.path.splitext(image_filename)[0]
        thumb_filename = f"{base_name_no_ext}_crop_{crop_index}.webp"
        thumb_path = self._path_mgr.get_derivative_path(thumb_filename, "thumb")

        # Generate thumbnail
        x1, y1, x2, y2 = detection.bbox
        self.generate_thumbnail(frame, detection.bbox, thumb_path)

        # Normalize bbox to 0-1 range
        img_h, img_w = frame.shape[:2]
        bbox_x = x1 / img_w
        bbox_y = y1 / img_h
        bbox_w = (x2 - x1) / img_w
        bbox_h = (y2 - y1) / img_h

        created_at_iso = datetime.now(UTC).isoformat()

        try:
            det_id = insert_detection(
                self._db_conn,
                {
                    "image_filename": image_filename,
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "od_class_name": detection.class_name,
                    "od_confidence": detection.confidence,
                    "od_model_id": detector_model_id,
                    "created_at": created_at_iso,
                    "score": detection.score,
                    "agreement_score": detection.agreement_score,
                    "detector_model_name": self._config.get(
                        "DETECTOR_MODEL_CHOICE", ""
                    ),
                    "detector_model_version": detector_model_id,
                    "classifier_model_name": "classifier",
                    "classifier_model_version": classifier_model_id,
                    "thumbnail_path": thumb_filename,
                    "frame_width": img_w,
                    "frame_height": img_h,
                },
            )

            # Insert classification if available
            if detection.cls_class_name:
                insert_classification(
                    self._db_conn,
                    {
                        "detection_id": det_id,
                        "cls_class_name": detection.cls_class_name,
                        "cls_confidence": detection.cls_confidence,
                        "cls_model_id": classifier_model_id,
                        "created_at": created_at_iso,
                    },
                )

            return DetectionPersistenceResult(
                success=True,
                detection_id=det_id,
                thumbnail_path=thumb_path,
                thumbnail_filename=thumb_filename,
            )

        except Exception as e:
            logger.error(f"Error saving detection: {e}")
            return DetectionPersistenceResult(success=False)

    def generate_thumbnail(
        self,
        frame,  # np.ndarray
        bbox: tuple[int, int, int, int],
        output_path: Path,
        size: int = 256,
    ) -> bool:
        """
        Generates a square thumbnail from a detection bbox.

        Uses edge-shifted square crop via CropService.

        Args:
            frame: Original frame.
            bbox: Bounding box as (x1, y1, x2, y2).
            output_path: Where to save the thumbnail.
            size: Target thumbnail size (square).

        Returns:
            True if thumbnail was generated successfully.
        """
        try:
            # Create thumbnail crop via CropService
            thumb_img = self._crop_service.create_thumbnail_crop(
                frame=frame,
                bbox=bbox,
                size=size,
            )

            if thumb_img is None:
                return False

            # Save as WebP
            cv2.imwrite(
                str(output_path),
                thumb_img,
                [int(cv2.IMWRITE_WEBP_QUALITY), 80],
            )
            return True

        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return False
