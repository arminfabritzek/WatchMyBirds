from detectors.detection_manager import DetectionManager


def test_deep_scan_gate_toggles_paused_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    manager = DetectionManager()

    assert manager.paused is False
    assert manager.is_deep_scan_active() is False

    manager.enter_deep_scan_mode()

    assert manager.paused is True
    assert manager.is_deep_scan_active() is True

    manager.exit_deep_scan_mode()

    assert manager.paused is False
    assert manager.is_deep_scan_active() is False


def test_deep_scan_gate_preserves_preexisting_pause_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    manager = DetectionManager()
    manager.paused = True

    manager.enter_deep_scan_mode()
    manager.exit_deep_scan_mode()

    assert manager.paused is True
    assert manager.is_deep_scan_active() is False


def test_deep_scan_gate_is_reference_counted(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    manager = DetectionManager()

    manager.enter_deep_scan_mode()
    manager.enter_deep_scan_mode()
    manager.exit_deep_scan_mode()

    assert manager.paused is True
    assert manager.is_deep_scan_active() is True

    manager.exit_deep_scan_mode()

    assert manager.paused is False
    assert manager.is_deep_scan_active() is False


def test_run_exhaustive_scan_delegates_to_detection_service(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    manager = DetectionManager()
    expected = [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "confidence": 0.9}]

    manager.detection_service.exhaustive_detect = lambda frame: expected
    manager.detection_service.get_model_id = lambda: "detector_v_test"
    manager.detector_model_id = ""

    result = manager.run_exhaustive_scan(frame=object())

    assert result == expected
    assert manager.detector_model_id == "detector_v_test"
