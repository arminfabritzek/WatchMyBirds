from __future__ import annotations

from flask.testing import FlaskClient
from werkzeug.test import TestResponse


def prepare_export(client: FlaskClient, token: str = "test-csrf-token") -> str:
    response = client.post(
        "/api/canonical-dataset/download", headers={"X-CSRF-Token": token}
    )
    assert response.status_code == 202, response.get_data(as_text=True)
    client.application.extensions["training_export_jobs"].wait(10)
    status = client.get(response.json["status_url"]).json
    assert status["status"] == "complete", status
    assert status["parts"], status
    return str(status["parts"][0]["export_id"])


def download_export(
    client: FlaskClient, token: str = "test-csrf-token"
) -> TestResponse:
    identifier = prepare_export(client, token)
    return client.get(
        f"/api/canonical-dataset/exports/{identifier}/download", buffered=True
    )
