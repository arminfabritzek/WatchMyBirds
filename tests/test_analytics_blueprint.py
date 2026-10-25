from __future__ import annotations

import pytest
from flask import Flask

from tests.test_surface_routes_local_db import local_db_app as local_db_app


@pytest.mark.parametrize(
    "path",
    [
        "/analytics/diagnostic",
        "/api/analytics/summary",
        "/api/analytics/time-of-day",
        "/api/analytics/species-activity",
        "/api/analytics/visits",
        "/api/analytics/event-intelligence",
        "/api/analytics/simulation",
        "/api/analytics/diversity",
        "/api/analytics/species-pca",
        "/api/analytics/species-table",
        "/api/analytics/quality-metrics",
    ],
)
def test_removed_diagnostic_analytics_returns_404(
    local_db_app: Flask, path: str
) -> None:
    with local_db_app.test_client() as client:
        assert client.get(path).status_code == 404
