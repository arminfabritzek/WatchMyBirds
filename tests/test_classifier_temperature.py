import numpy as np

from detectors.classifier import ImageClassifier
from detectors.cls_config import ClsDecisionConfig


class _FakeInput:
    name = "input"


class _FakeOrtSession:
    def __init__(self, logits):
        self._logits = np.asarray(logits, dtype=np.float32)

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, _ort_inputs):
        return [self._logits]


def test_predict_from_image_applies_temperature_scaled_softmax(monkeypatch):
    clf = ImageClassifier()
    monkeypatch.setattr(clf, "_ensure_initialized", lambda: None)
    clf.ort_session = _FakeOrtSession([[2.0, 1.0]])
    clf.classes = ["Parus_major", "Parus_caeruleus"]
    clf.CLASSIFIER_IMAGE_SIZE = 2
    clf.mean = np.zeros((3, 1, 1), dtype=np.float32)
    clf.std = np.ones((3, 1, 1), dtype=np.float32)
    clf.decision_config = ClsDecisionConfig(
        species_threshold=0.65,
        genus_threshold=0.55,
        genus_map={
            "Parus_major": "Parus",
            "Parus_caeruleus": "Parus",
        },
        genus_pairs=frozenset(["Parus"]),
        temperature=2.0,
    )

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    _, top_k_confidences, top1_class_name, top1_confidence = clf.predict_from_image(image)

    expected_top1 = np.exp(1.0) / (np.exp(1.0) + np.exp(0.5))
    assert top1_class_name == "Parus_major"
    assert top1_confidence == np.float32(expected_top1)
    assert top_k_confidences[0] == np.float32(expected_top1)
    # With T=2.0 the top-1 stays below the species threshold, so the
    # decision layer must not behave like the uncalibrated softmax path.
    assert clf.last_decision["level"] == "genus"
    assert clf.last_decision["prob"] > top1_confidence
