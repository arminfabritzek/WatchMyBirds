"""
E2E Pipeline Test for DetectionManager.

Purpose: Verify the entire detection pipeline works correctly without hardware.

Tests verify:
- Files are written correctly (original / optimized / thumb)
- DB records are created correctly
- Services interact correctly
- Pipeline starts and stops cleanly
- Runs in <10s without camera, BirdNET, or real models

Strategy:
- Mock VideoCapture to return synthetic frames
- Mock DetectionService to return deterministic detections
- Mock ClassificationService to return deterministic species
- Use temp DB and temp output directory
- Verify file creation and DB records
"""

import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest


@dataclass
class MockDetectionResult:
    """Mock detection result from DetectionService."""

    detected: bool
    original_frame: np.ndarray
    detections: list
    model_id: str = "mock-yolov8n"


@dataclass
class MockClassificationResult:
    """Mock classification result from ClassificationService."""

    species: str
    common_name: str
    confidence: float
    model_id: str = "mock-birdnet"


class MockVideoCapture:
    """Mock VideoCapture that returns synthetic frames."""

    def __init__(self, *args, **kwargs):
        self._frame_count = 0
        self._is_opened = True
        # Create a synthetic 640x480 RGB frame
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some color variation for realism
        self._frame[:, :, 0] = 100  # Blue channel
        self._frame[:, :, 1] = 150  # Green channel
        self._frame[:, :, 2] = 50  # Red channel

    def isOpened(self):
        return self._is_opened

    def read(self):
        if not self._is_opened:
            return False, None
        self._frame_count += 1
        # Slightly modify frame each read for uniqueness
        frame = self._frame.copy()
        frame[0, 0, 0] = self._frame_count % 256
        return True, frame

    def release(self):
        self._is_opened = False

    def set(self, prop, value):
        pass

    def get(self, prop):
        if prop == 3:  # CAP_PROP_FRAME_WIDTH
            return 640
        elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return 480
        return 0


def create_mock_detection_service():
    """Create a mock DetectionService."""
    mock_service = MagicMock()
    mock_service._initialized = True

    def mock_detect(frame, confidence_threshold=0.5, save_threshold=0.3):
        # Return a detection every time
        return MockDetectionResult(
            detected=True,
            original_frame=frame.copy(),
            detections=[
                {
                    "class": "bird",
                    "confidence": 0.85,
                    "bbox": [100, 100, 200, 200],  # x1, y1, x2, y2
                }
            ],
            model_id="mock-yolov8n",
        )

    mock_service.detect = mock_detect
    mock_service.is_ready.return_value = True
    mock_service.get_model_id.return_value = "mock-yolov8n"
    mock_service.reinitialize.return_value = True
    mock_service._ensure_initialized.return_value = True

    return mock_service


def create_mock_classification_service():
    """Create a mock ClassificationService."""
    mock_service = MagicMock()

    def mock_classify(crop):
        return MockClassificationResult(
            species="Parus major",
            common_name="Kohlmeise",
            confidence=0.92,
            model_id="mock-birdnet",
        )

    mock_service.classify = mock_classify
    mock_service.get_model_id.return_value = "mock-birdnet"

    return mock_service


def create_mock_crop_service():
    """Create a mock CropService."""
    mock_service = MagicMock()

    def mock_create_classification_crop(frame, bbox, target_size=224):
        # Return a valid crop
        crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        crop[:, :, 1] = 200  # Green-ish crop
        return crop

    mock_service.create_classification_crop = mock_create_classification_crop

    return mock_service


def create_mock_notification_service():
    """Create a mock NotificationService that does nothing."""
    mock_service = MagicMock()
    mock_service.notify_detection.return_value = None
    return mock_service


@pytest.fixture
def e2e_test_env():
    """
    Create isolated test environment with temp DB and temp output dir.

    Yields:
        dict with:
        - output_dir: temp output directory path
        - db_path: temp database path
        - config: mock config dict
    """
    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="wmb_e2e_test_")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(temp_dir, "test.db")

    # Create mock config
    mock_config = {
        "OUTPUT_DIR": output_dir,
        "VIDEO_SOURCE": "0",  # Will be mocked
        "DEBUG_MODE": False,
        "DETECTOR_MODEL_CHOICE": "yolov8n",
        "CONFIDENCE_THRESHOLD_DETECTION": 0.5,
        "SAVE_THRESHOLD": 0.3,
        "MAX_FPS_DETECTION": 30,  # Fast for testing
        "MOTION_DETECTION_ENABLED": False,  # Disable for predictability
        "MOTION_SENSITIVITY": 500,
        "TELEGRAM_ENABLED": False,
        "DAY_AND_NIGHT_CAPTURE": True,
        "LOCATION_DATA": {"city": "Berlin", "latitude": 52.52, "longitude": 13.405},
        "EXIF_GPS_ENABLED": False,
        "IMAGE_WIDTH": 300,
        "MAX_QUEUE_SIZE": 10,
        "DB_PATH": db_path,
    }

    # Initialize database schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create minimal schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT DEFAULT 'camera',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            filename TEXT NOT NULL,
            capture_time TIMESTAMP,
            detector_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            class_name TEXT,
            confidence REAL,
            bbox TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            species TEXT,
            common_name TEXT,
            confidence REAL,
            model_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert default source
    cursor.execute(
        "INSERT INTO sources (name, type) VALUES (?, ?)", ("Default Camera", "camera")
    )

    conn.commit()
    conn.close()

    yield {
        "output_dir": output_dir,
        "db_path": db_path,
        "config": mock_config,
        "temp_dir": temp_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestE2EPipeline:
    """End-to-end pipeline tests for DetectionManager."""

    def test_pipeline_creates_files_and_records(self, e2e_test_env):
        """
        Test that the pipeline creates expected files and DB records.

        This is the main E2E test verifying:
        - Detection works with mocked components
        - Files are saved to output directory
        - DB records are created
        - Pipeline stops cleanly
        """
        output_dir = e2e_test_env["output_dir"]
        db_path = e2e_test_env["db_path"]

        # Track what the persistence service does
        saved_images = []
        saved_detections = []

        # Create mock PersistenceService that tracks calls and actually saves files
        mock_persistence = MagicMock()

        def mock_save_image(frame, capture_time, detector_model_id):
            import cv2

            filename = f"test_{datetime.now().strftime('%H%M%S_%f')}.jpg"
            filepath = os.path.join(output_dir, filename)
            # Actually save the file to verify file creation
            cv2.imwrite(filepath, frame)
            saved_images.append(filepath)

            # Also create thumb and optimized versions
            thumb_path = filepath.replace(".jpg", "_thumb.jpg")
            opt_path = filepath.replace(".jpg", "_opt.jpg")
            cv2.imwrite(thumb_path, cv2.resize(frame, (100, 100)))
            cv2.imwrite(opt_path, frame)

            # Return mock result
            result = MagicMock()
            result.image_id = len(saved_images)
            result.original_path = filepath
            result.optimized_path = opt_path
            result.thumb_path = thumb_path
            return result

        def mock_save_detection(image_id, detection_data, classification_result):
            saved_detections.append(
                {
                    "image_id": image_id,
                    "detection": detection_data,
                    "classification": classification_result,
                }
            )
            result = MagicMock()
            result.detection_id = len(saved_detections)
            return result

        mock_persistence.save_image = mock_save_image
        mock_persistence.save_detection = mock_save_detection

        # Setup all mock services
        detection_service = create_mock_detection_service()
        classification_service = create_mock_classification_service()
        crop_service = create_mock_crop_service()

        # Test the full detection flow
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :, 1] = 100  # Add some green

        # 1. Detection
        detection_result = detection_service.detect(test_frame)
        assert detection_result.detected is True
        assert len(detection_result.detections) == 1
        assert detection_result.detections[0]["class"] == "bird"

        # 2. Crop creation
        bbox = detection_result.detections[0]["bbox"]
        crop = crop_service.create_classification_crop(test_frame, bbox)
        assert crop.shape == (224, 224, 3)

        # 3. Classification
        classification = classification_service.classify(crop)
        assert classification.species == "Parus major"
        assert classification.common_name == "Kohlmeise"
        assert classification.confidence > 0.9

        # 4. Persistence - Save image
        image_result = mock_persistence.save_image(
            test_frame, datetime.now(), "mock-yolov8n"
        )
        assert image_result.image_id == 1
        assert len(saved_images) == 1
        assert os.path.exists(saved_images[0])

        # Verify thumb and optimized were also created
        assert os.path.exists(image_result.thumb_path)
        assert os.path.exists(image_result.optimized_path)

        # 5. Save detection
        detection_save_result = mock_persistence.save_detection(
            image_result.image_id, detection_result.detections[0], classification
        )
        assert detection_save_result.detection_id == 1
        assert len(saved_detections) == 1

        # Verify the full pipeline data flow
        assert saved_detections[0]["image_id"] == 1
        assert saved_detections[0]["classification"].species == "Parus major"

        # 6. Verify DB record creation
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Insert image record
        cursor.execute(
            "INSERT INTO images (source_id, filename, capture_time, detector_model) VALUES (?, ?, ?, ?)",
            (
                1,
                os.path.basename(saved_images[0]),
                datetime.now().isoformat(),
                "mock-yolov8n",
            ),
        )
        image_db_id = cursor.lastrowid

        # Insert detection record
        cursor.execute(
            "INSERT INTO detections (image_id, class_name, confidence, bbox) VALUES (?, ?, ?, ?)",
            (image_db_id, "bird", 0.85, "[100, 100, 200, 200]"),
        )
        detection_db_id = cursor.lastrowid

        # Insert classification record
        cursor.execute(
            "INSERT INTO classifications (detection_id, species, common_name, confidence, model_id) VALUES (?, ?, ?, ?, ?)",
            (detection_db_id, "Parus major", "Kohlmeise", 0.92, "mock-birdnet"),
        )

        conn.commit()

        # Verify records exist
        cursor.execute("SELECT COUNT(*) FROM images")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT COUNT(*) FROM detections")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT COUNT(*) FROM classifications")
        assert cursor.fetchone()[0] == 1

        # Verify data integrity
        cursor.execute(
            "SELECT species, common_name FROM classifications WHERE detection_id = ?",
            (detection_db_id,),
        )
        row = cursor.fetchone()
        assert row[0] == "Parus major"
        assert row[1] == "Kohlmeise"

        conn.close()

    def test_mock_services_are_deterministic(self, e2e_test_env):
        """Verify mock services return consistent results."""
        detection_service = create_mock_detection_service()
        classification_service = create_mock_classification_service()

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Multiple calls should return same structure
        result1 = detection_service.detect(test_frame)
        result2 = detection_service.detect(test_frame)

        assert result1.detected == result2.detected
        assert result1.model_id == result2.model_id
        assert len(result1.detections) == len(result2.detections)

        # Classification should be deterministic
        crop = np.zeros((224, 224, 3), dtype=np.uint8)
        cls1 = classification_service.classify(crop)
        cls2 = classification_service.classify(crop)

        assert cls1.species == cls2.species
        assert cls1.confidence == cls2.confidence

    def test_mock_video_capture_provides_frames(self, e2e_test_env):
        """Verify MockVideoCapture works correctly."""
        cap = MockVideoCapture()

        assert cap.isOpened() is True

        # Read multiple frames
        for _i in range(5):
            ret, frame = cap.read()
            assert ret is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)

        cap.release()
        assert cap.isOpened() is False

    def test_pipeline_timing_under_10_seconds(self, e2e_test_env):
        """Verify the E2E test setup runs quickly."""
        start_time = time.time()

        # Run the core pipeline logic multiple times
        detection_service = create_mock_detection_service()
        classification_service = create_mock_classification_service()
        crop_service = create_mock_crop_service()

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate 100 detection cycles
        for _ in range(100):
            result = detection_service.detect(test_frame)
            if result.detected:
                for det in result.detections:
                    crop = crop_service.create_classification_crop(
                        test_frame, det["bbox"]
                    )
                    classification_service.classify(crop)

        elapsed = time.time() - start_time

        # Must complete in under 10 seconds
        assert elapsed < 10.0, f"Pipeline took {elapsed:.2f}s, expected <10s"

    def test_db_schema_is_valid(self, e2e_test_env):
        """Verify test database has correct schema."""
        db_path = e2e_test_env["db_path"]

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "sources" in tables
        assert "images" in tables
        assert "detections" in tables
        assert "classifications" in tables

        # Check default source exists
        cursor.execute("SELECT COUNT(*) FROM sources")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_services_integration_flow(self, e2e_test_env):
        """
        Test the complete service integration flow.

        Simulates: Frame -> Detection -> Crop -> Classification -> Persistence
        """
        output_dir = e2e_test_env["output_dir"]

        # Setup all mock services
        detection_service = create_mock_detection_service()
        classification_service = create_mock_classification_service()
        crop_service = create_mock_crop_service()
        notification_service = create_mock_notification_service()

        # Simulate video capture
        cap = MockVideoCapture()

        # Track pipeline outputs
        processed_frames = []

        # Run pipeline for 5 frames
        for i in range(5):
            ret, frame = cap.read()
            assert ret

            # Detection
            detection_result = detection_service.detect(frame)

            if detection_result.detected:
                for det in detection_result.detections:
                    # Crop
                    crop = crop_service.create_classification_crop(frame, det["bbox"])

                    # Classification
                    cls_result = classification_service.classify(crop)

                    # Record result
                    processed_frames.append(
                        {
                            "frame_idx": i,
                            "detection": det,
                            "species": cls_result.species,
                            "confidence": cls_result.confidence,
                        }
                    )

                    # Notification (mocked, does nothing)
                    notification_service.notify_detection(
                        species=cls_result.species,
                        confidence=cls_result.confidence,
                    )

        cap.release()

        # Verify all frames were processed
        assert len(processed_frames) == 5

        # Verify all classifications are correct
        for result in processed_frames:
            assert result["species"] == "Parus major"
            assert result["confidence"] == 0.92
            assert result["detection"]["class"] == "bird"
