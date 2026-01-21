
import os
import hashlib
import time
import cv2
from utils.image_ops import create_square_crop
import piexif
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from config import get_config
from utils.db import (
    get_connection,
    insert_image,
    insert_detection,
    insert_classification,
    check_image_exists_by_hash,
)
from utils.path_manager import get_path_manager
from detectors.detector import Detector
from detectors.classifier import ImageClassifier

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("FolderIngest")

config = get_config()

SAVE_RESOLUTION_CROP = 512

def calculate_sha256(filepath: str) -> str:
    """Calculates the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to avoid memory issues with large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_original_exif(filepath: str) -> Optional[bytes]:
    """
    Extracts EXIF data from the original image file.
    Returns the EXIF bytes if found, None otherwise.
    Used to preserve original EXIF/GPS data during manual ingest.
    """
    try:
        exif_dict = piexif.load(filepath)
        # Check if there's any meaningful EXIF data
        if exif_dict["0th"] or exif_dict["Exif"] or exif_dict["GPS"]:
            return piexif.dump(exif_dict)
    except Exception as e:
        logger.debug(f"No EXIF data found in {os.path.basename(filepath)}: {e}")
    return None

def get_image_creation_date(filepath: str) -> datetime:
    """
    Determines the creation timestamp of an image.
    Priority: EXIF DateTimeOriginal > File Modification Time > Now
    """
    try:
        # 1. Try EXIF
        exif_dict = piexif.load(filepath)
        if piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
            dt_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
            # Format is typically "YYYY:MM:DD HH:MM:SS"
            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass  # Fallback

    # 2. Try File Modification Time
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime)
    except Exception:
        pass

    # 3. Fallback to Now
    return datetime.now()



def ingest_folder(folder_path: str, source_id: int, move_files: bool = False):
    """
    Recursively scans and ingests images from a folder.
    Args:
        folder_path (str): The root folder to ingest from.
        source_id (int): The ID of the source.
        move_files (bool): If True, moves files to 'processed', 'skipped', 'error' subfolders.
    """
    logger.info(f"Starting ingest from: {folder_path} (Source ID: {source_id}, Move: {move_files})")
    
    detector = Detector(model_choice=config["DETECTOR_MODEL_CHOICE"], debug=False)
    # Check if detector loaded correctly
    det_model_id = getattr(detector, "model_id", "") or "unknown"
    # Basic attempt to split version if model_id looks like "model_v1.onnx" or similar, 
    # but strictly speaking we treat the ID as version for now as requested.
    # The requirement says: "Keine Ableitung aus Dateipfaden... Strings explizit übergeben"
    # Since we can't change Detector class interface easily without checking detectors/*,
    # we use the loaded `model_id` as version and the configured name as name.
    det_model_name = config["DETECTOR_MODEL_CHOICE"]
    det_model_version = det_model_id

    classifier = ImageClassifier()
    cls_model_id = getattr(classifier, "model_id", "unknown")
    cls_model_name = "classifier" # Default name
    cls_model_version = cls_model_id
    
    conn = get_connection()
    
    count_ingested = 0
    count_skipped = 0
    count_errors = 0

    for root, dirnames, files in os.walk(folder_path):
        # Exclude system folders from recursion
        dirnames[:] = [d for d in dirnames if d not in ("processed", "skipped", "error")]

        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(root, filename)
                try:
                    status = ingest_file(conn, filepath, source_id, detector, classifier, 
                                         det_meta=(det_model_name, det_model_version), 
                                         cls_meta=(cls_model_name, cls_model_version))
                    
                    if move_files:
                        _handle_file_move(root, filename, status)

                    if status == "ingested":
                        count_ingested += 1
                    elif status == "skipped":
                        count_skipped += 1
                except Exception as e:
                    logger.error(f"Error ingesting {filepath}: {e}", exc_info=True)
                    if move_files:
                        _handle_file_move(root, filename, "error")
                    count_errors += 1
    
    logger.info(f"Ingest complete. Ingested: {count_ingested}, Skipped: {count_skipped}, Errors: {count_errors}")

def _handle_file_move(root: str, filename: str, status: str):
    """
    Moves file to appropriate subdirectory within the root.
    """
    import shutil
    
    # Define Target Folder
    timestamp_folder = datetime.now().strftime("%Y%m%d")
    
    if status == "ingested":
        target_dir = os.path.join(config.get("OUTPUT_DIR", root), "processed", timestamp_folder)
    elif status == "skipped":
        target_dir = os.path.join(root, "skipped", timestamp_folder)
    else: # error
        target_dir = os.path.join(root, "error")
        
    os.makedirs(target_dir, exist_ok=True)
    
    src_path = os.path.join(root, filename)
    dst_path = os.path.join(target_dir, filename)
    
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        logger.error(f"Failed to move file {filename} to {dst_path}: {e}") 

def ingest_file(conn, filepath: str, source_id: int, detector, classifier, 
                det_meta: Tuple[str, str] = ("unknown", "unknown"), 
                cls_meta: Tuple[str, str] = ("unknown", "unknown")) -> str:
    """
    Ingests a single file. 
    Returns: 'ingested', 'skipped', 'error' (though error usually raises exception)
    """
    det_name, det_version = det_meta
    cls_name, cls_version = cls_meta
    # 1. Calculate Hash & Check Idempotency
    content_hash = calculate_sha256(filepath)
    if check_image_exists_by_hash(conn, content_hash):
        logger.debug(f"Skipping duplicate (hash match): {os.path.basename(filepath)}")
        return "skipped"

    # 2. Extract original EXIF data (before loading with cv2 which strips it)
    original_exif_bytes = extract_original_exif(filepath)
    
    # 3. Load Image
    image = cv2.imread(filepath)
    if image is None:
        logger.error(f"Failed to load image (cv2 returned None): {filepath}")
        raise ValueError("Failed to load image") # Raise to trigger error handling/move

    # 4. Determine Timestamp
    creation_dt = get_image_creation_date(filepath)
    timestamp_str = creation_dt.strftime("%Y%m%d_%H%M%S")
    
    # 5. Run Detection
    object_detected, _, detection_info_list = detector.detect_objects(
        image,
        confidence_threshold=config["CONFIDENCE_THRESHOLD_DETECTION"],
        save_threshold=config["SAVE_THRESHOLD"]
    )

    created_at_iso = datetime.now(timezone.utc).isoformat()
    output_dir = config["OUTPUT_DIR"]
    pm = get_path_manager(output_dir)
    # Ensure directory structure exists for this date
    date_str_iso = f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}"
    pm.ensure_date_structure(date_str_iso)

    # 6. Pipeline Logic
    
    # CASE A: Detection Found
    if object_detected and detection_info_list:
        # Identify Best
        best_det = max(detection_info_list, key=lambda d: d["confidence"])
        best_class = best_det["class_name"].replace(" ", "_")
        
        base_name = f"{timestamp_str}.jpg"
        
        original_name = base_name
        optimized_name = f"{timestamp_str}.webp"
        
        # Use PathManager for paths
        original_path = str(pm.get_original_path(original_name))
        optimized_path = str(pm.get_derivative_path(optimized_name, "optimized"))
        
        # Save Images
        cv2.imwrite(original_path, image)
        
        if image.shape[1] > 800:
            optimized_frame = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
            cv2.imwrite(optimized_path, optimized_frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
        else:
            cv2.imwrite(optimized_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
        
        # Preserve original EXIF/GPS data if present
        if original_exif_bytes:
            try:
                piexif.insert(original_exif_bytes, original_path)
                piexif.insert(original_exif_bytes, optimized_path)
                logger.debug(f"Preserved original EXIF data for {original_name}")
            except Exception as e:
                logger.warning(f"Failed to preserve EXIF data: {e}")


        
        # PASS 1: Process all detections, classify, save crops, collect enriched data
        enriched_detections = []
        
        for i, det in enumerate(detection_info_list, start=1):
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            bbox_tuple = (x1, y1, x2, y2)
            cls_name = None
            cls_conf = 0.0

            # Crop & Classify
            crop = create_square_crop(image, bbox_tuple, margin_percent=0.1)
            if crop is not None and crop.size > 0:
                crop_resized = cv2.resize(crop, (SAVE_RESOLUTION_CROP, SAVE_RESOLUTION_CROP))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                # Classifier predict
                _, _, cls_name, cls_conf = classifier.predict_from_image(crop_rgb)
                
            # --- GENERATE SQUARE THUMBNAIL (Match Live Stream: _crop_{i}) ---
            thumb_filename = f"{timestamp_str}_crop_{i}.webp"
            thumb_path = str(pm.get_derivative_path(thumb_filename, "thumb"))
            TARGET_THUMB_SIZE = 256
            EXPANSION_PERCENT = 0.10
            
            try:
                # 1. Calculate square side from bbox + expansion
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                side = int(max(bbox_w, bbox_h) * (1 + EXPANSION_PERCENT))
                
                # 2. Center the square on the bbox center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                sq_x1 = int(cx - side / 2)
                sq_y1 = int(cy - side / 2)
                sq_x2 = sq_x1 + side
                sq_y2 = sq_y1 + side
                
                # 3. Clamp to image boundaries by SHIFTING (not shrinking or padding)
                img_h, img_w = image.shape[:2]
                if sq_x1 < 0:
                    sq_x2 -= sq_x1  # shift right
                    sq_x1 = 0
                if sq_y1 < 0:
                    sq_y2 -= sq_y1  # shift down
                    sq_y1 = 0
                if sq_x2 > img_w:
                    sq_x1 -= (sq_x2 - img_w)  # shift left
                    sq_x2 = img_w
                if sq_y2 > img_h:
                    sq_y1 -= (sq_y2 - img_h)  # shift up
                    sq_y2 = img_h
                
                # Final clamp
                sq_x1 = max(0, sq_x1)
                sq_y1 = max(0, sq_y1)
                sq_x2 = min(img_w, sq_x2)
                sq_y2 = min(img_h, sq_y2)
                
                # 4. Crop and Resize
                if sq_x2 > sq_x1 and sq_y2 > sq_y1:
                    square_crop = image[sq_y1:sq_y2, sq_x1:sq_x2]
                    if square_crop.size > 0:
                        thumb_img = cv2.resize(square_crop, (TARGET_THUMB_SIZE, TARGET_THUMB_SIZE), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(thumb_path, thumb_img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                else:
                    thumb_filename = None
            except Exception as e:
                logger.error(f"Error generating thumbnail {thumb_filename}: {e}")
                thumb_filename = None

            # Store enriched detection data
            enriched_detections.append({
                "bbox": (x1, y1, x2, y2),
                "od_class_name": det["class_name"],
                "od_confidence": det["confidence"],
                "cls_class_name": cls_name,
                "cls_confidence": cls_conf,
                "thumb_filename": thumb_filename
            })

        # Calculate Primary Scores & Aggregates
        best_score_val = 0.0
        for enriched_det in enriched_detections:
            od = enriched_det["od_confidence"]
            cls = enriched_det["cls_confidence"]
            # Formula: 0.5 * od + 0.5 * cls
            # If cls is 0.0 (no class found/threshold), effectively it lowers the score, which is correct.
            # However, if cls is missing/None, we should treat it as od (fallback).
            # Here cls_conf is 0.0 if not found, so let's check for that.
            if cls > 0:
                 score = 0.5 * od + 0.5 * cls
            else:
                 score = od
            
            if cls > 0:
                 score = 0.5 * od + 0.5 * cls
                 agreement = min(od, cls)
            else:
                 score = od
                 agreement = od
            
            enriched_det["score"] = score
            enriched_det["agreement_score"] = agreement # New field
            
            if score > best_score_val:
                best_score_val = score
            
            # Logging according to verification request (DEBUG only)
            logger.debug(
                f"Ingest Logic: od={od:.3f}, cls={cls:.3f} -> score={score:.3f}, agreement={agreement:.3f} "
                f"[Models: od={det_name}:{det_version}, cls={cls_name}:{cls_version}]"
            )

        # INSERT PARENT IMAGE RECORD FIRST
        insert_image(conn, {
            "filename": original_name,
            "timestamp": timestamp_str,
            "original_name": original_name,
            "optimized_name": optimized_name,
            "coco_json": "{}", # Empty for ingest
            "source_id": source_id,
            "content_hash": content_hash,
            "detector_model_id": getattr(detector, "model_id", "unknown"),
            "classifier_model_id": getattr(classifier, "model_id", "unknown"),
        })
        
        # PASS 2: Now insert child detection and classification records
        for enriched_det in enriched_detections:
            x1, y1, x2, y2 = enriched_det["bbox"]
            
            # DB: Insert Detection
            det_id = insert_detection(conn, {
                "image_filename": original_name,
                "image_timestamp": timestamp_str,
                "bbox_x": x1 / image.shape[1],
                "bbox_y": y1 / image.shape[0],
                "bbox_w": (x2 - x1) / image.shape[1],
                "bbox_h": (y2 - y1) / image.shape[0],
                "od_class_name": enriched_det["od_class_name"],
                "od_confidence": enriched_det["od_confidence"],
                "od_model_id": getattr(detector, "model_id", "unknown"),
                "created_at": created_at_iso,
                "score": enriched_det["score"],
                "agreement_score": enriched_det["agreement_score"],
                "detector_model_name": det_name,
                "detector_model_version": det_version,
                "classifier_model_name": cls_name,
                "classifier_model_version": cls_version,
                "thumbnail_path": enriched_det["thumb_filename"]
            })

            # DB: Insert Classification
            if enriched_det["cls_class_name"]:
                insert_classification(conn, {
                    "detection_id": det_id,
                    "cls_class_name": enriched_det["cls_class_name"],
                    "cls_confidence": enriched_det["cls_confidence"],
                    "cls_model_id": getattr(classifier, "model_id", "unknown"),
                    "created_at": created_at_iso
                })
        
        logger.info(f"Ingested {filepath} -> {original_name}")

    # CASE B: No Detection
    else:
        # Save file to maintain Integrity? 
        # Plan decision: We need to store it to mark hash as processed.
        # We will save it with a generic name so the DB entry points to something valid.
        
        name_no_det = f"{timestamp_str}_nodetection.jpg"
        save_path = str(pm.get_original_path(name_no_det))
        cv2.imwrite(save_path, image)
        
        # Preserve original EXIF/GPS data if present
        if original_exif_bytes:
            try:
                piexif.insert(original_exif_bytes, save_path)
                logger.debug(f"Preserved original EXIF data for {name_no_det}")
            except Exception as e:
                logger.warning(f"Failed to preserve EXIF data: {e}")
        
        insert_image(conn, {
            "filename": name_no_det,
            "timestamp": timestamp_str,
            "original_name": name_no_det,
            "optimized_name": name_no_det.replace(".jpg", ".webp"),
            "source_id": source_id,
            "content_hash": content_hash,
            "coco_json": "{}",
            "detector_model_id": getattr(detector, "model_id", "unknown"),
            "classifier_model_id": getattr(classifier, "model_id", "unknown"),
        })
        logger.info(f"Ingested (No Detection) {filepath}")

    return "ingested"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python utils/ingest.py <folder_path> <source_id>")
        sys.exit(1)
    
    ingest_folder(sys.argv[1], int(sys.argv[2]))
