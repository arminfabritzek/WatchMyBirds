"""Successful answer and trash requests invalidate analytics only."""

from unittest.mock import patch

import pytest

from core.human_label_core import HumanLabelError
from tests.labeling_helpers import post
from tests.test_human_labels_blueprint import labeling_case as _labeling_case

case = _labeling_case
from web.services import cache_service


@pytest.fixture(autouse=True)
def reset_cache():
    cache_service.clear()
    yield
    cache_service.clear()


@pytest.mark.parametrize("action", ["answer", "trash"])
def test_mutations_invalidate_analytics_cache(case, action):
    client, filename, did = case
    cache_service.cached("analytics.summary", 60, lambda: "cached")
    cache_service.cached("gallery.daily", 60, lambda: "preserved")
    if action == "answer":
        response = post(
            client,
            "/api/labels/answer",
            dict(detection_ids=[did], species_identity="unknown"),
        )
    else:
        response = post(client, "/api/trash/move", dict(detection_ids=[did]))
    assert response.status_code == 200
    assert cache_service.size() == 1
    assert cache_service.cached("gallery.daily", 60, lambda: "new") == "preserved"


def test_failed_answer_keeps_cache(case):
    client, _, did = case
    cache_service.cached("analytics.summary", 60, lambda: "cached")
    with patch(
        "web.services.human_label_service.apply_answer_request",
        side_effect=HumanLabelError("answer rejected"),
    ):
        assert (
            post(
                client,
                "/api/labels/answer",
                dict(detection_ids=[did], species_identity="unknown"),
            ).status_code
            == 409
        )
    assert cache_service.cached("analytics.summary", 60, lambda: "new") == "cached"
