"""Removing the star from several photos in one step."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from core import gallery_core
from tests.test_favorites_only_filter import local_db_app as local_db_app
from tests.test_favorites_page import album as album
from tests.test_favorites_page import owner as owner
from tests.test_favorites_page import tiles
from tests.test_favorites_page import visitor as visitor
from utils.db import closing_connection

ROUTE = "/api/moderation/bulk/unfavorite"


@pytest.fixture(autouse=True)
def fresh_gallery_cache() -> Iterator[None]:
    gallery_core.invalidate_cache()
    yield
    gallery_core.invalidate_cache()


def send(client: FlaskClient, payload: Any, route: str = ROUTE) -> TestResponse:
    with client.session_transaction() as session:
        session["_csrf_token"] = "test-csrf-token"
    return client.post(route, json=payload, headers={"X-CSRF-Token": "test-csrf-token"})


def starred() -> set[int]:
    with closing_connection() as conn:
        rows = conn.execute(
            "SELECT detection_id FROM detections WHERE COALESCE(is_favorite, 0) != 0"
        )
        return {int(row[0]) for row in rows}


def trace(detection_id: int) -> tuple[int, str | None, str | None]:
    """What a star change leaves on a photo: state, who set it, when."""
    with closing_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(is_favorite, 0), rating_source, species_updated_at"
            " FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
    return int(row[0]), row[1], row[2]


def test_the_owner_removes_the_star_from_the_selected_photos(
    album: dict[str, int], owner: FlaskClient
) -> None:
    response = send(owner, {"detection_ids": [album["tit_old"], album["magpie"]]})

    assert response.status_code == 200
    assert response.get_json() == {"status": "success", "removed_favorites": 2}
    remaining = owner.get("/favorites").get_data(as_text=True)
    assert tiles(remaining) == [album["tit_new"], album["blackbird"]]


def test_a_photo_without_a_star_is_left_as_it_was(
    album: dict[str, int], owner: FlaskClient
) -> None:
    before = trace(album["plain"])

    response = send(owner, {"detection_ids": [album["plain"], album["magpie"]]})

    assert response.get_json()["removed_favorites"] == 1
    assert trace(album["plain"]) == before


def test_a_photo_that_is_gone_does_not_stop_the_rest(
    album: dict[str, int], owner: FlaskClient
) -> None:
    response = send(owner, {"detection_ids": [987654, album["magpie"]]})

    assert response.status_code == 200
    assert response.get_json()["removed_favorites"] == 1
    assert album["magpie"] not in starred()


def test_the_removal_leaves_the_same_trace_as_a_single_star_click(
    album: dict[str, int], owner: FlaskClient
) -> None:
    before = trace(album["magpie"])
    send(owner, {"detection_id": album["tit_old"]}, "/api/detections/favorite")

    send(owner, {"detection_ids": [album["magpie"]]})

    single, several = trace(album["tit_old"]), trace(album["magpie"])
    assert several[:2] == single[:2] == (0, "manual")
    assert several[2] != before[2]
    assert abs(
        datetime.fromisoformat(several[2]) - datetime.fromisoformat(single[2])
    ) < timedelta(seconds=30)


def test_pages_that_remember_photos_show_the_change_at_once(
    album: dict[str, int], owner: FlaskClient
) -> None:
    def remembered_star() -> bool:
        return next(
            bool(photo.get("is_favorite"))
            for photo in gallery_core.get_captured_detections()
            if photo["detection_id"] == album["magpie"]
        )

    assert remembered_star()

    send(owner, {"detection_ids": [album["magpie"]]})

    assert not remembered_star()


def test_a_visitor_cannot_remove_stars(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    before = starred()

    response = send(visitor, {"detection_ids": [album["magpie"]]})

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    assert starred() == before


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"detection_ids": []},
        {"detection_ids": "12"},
        {"detection_ids": ["12"]},
        {"detection_ids": [0]},
        {"detection_ids": [True]},
        {"detection_ids": [1.5]},
        {"detection_ids": list(range(1, 1002))},
        ["12"],
    ],
)
def test_a_selection_that_is_no_list_of_photo_numbers_is_refused(
    album: dict[str, int], owner: FlaskClient, payload: Any
) -> None:
    before = starred()

    response = send(owner, payload)

    assert response.status_code == 400
    assert response.get_json() == {
        "status": "error",
        "message": "Choose between 1 and 1000 photos",
    }
    assert starred() == before
