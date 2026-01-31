# tests/test_motion_detector.py
import cv2
import numpy as np

from detectors.motion_detector import MotionDetector


def test_no_motion_identical_frames():
    detector = MotionDetector(sensitivity=500)

    # Create black frame
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

    # First frame initializes
    assert not detector.detect(frame1)

    # Second frame identical -> No motion
    assert not detector.detect(frame2)


def test_motion_detected():
    detector = MotionDetector(sensitivity=100)

    # Frame 1: Black
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 2: White square in middle
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (40, 40), (60, 60), (255, 255, 255), -1)

    detector.detect(frame1)

    # Should detect 20x20=400 pixels > 100 sensitivity
    assert detector.detect(frame2)


def test_sensitivity_threshold():
    # High sensitivity threshold
    detector = MotionDetector(sensitivity=1000)

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Small change (10x10 = 100 pixels)
    cv2.rectangle(frame2, (45, 45), (55, 55), (255, 255, 255), -1)

    detector.detect(frame1)

    # Should NOT detect because 100 < 1000
    assert not detector.detect(frame2)


def test_noise_filtered():
    # Testing that slight noise (GaussianBlur) is handled
    detector = MotionDetector(sensitivity=500)

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()

    # Add single pixel noise
    frame2[50, 50] = [50, 50, 50]

    detector.detect(frame1)
    assert not detector.detect(frame2)
