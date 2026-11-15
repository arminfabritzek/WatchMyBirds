"""The Favorites page: every starred photo of the station in one place."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import Flask
from flask.testing import FlaskClient

from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from tests.test_favorites_only_filter import _seed_detection
from tests.test_favorites_only_filter import local_db_app as local_db_app
from utils.db import closing_connection
from web import view_helpers

TILE = re.compile(r'<div class="wm-tile[^"]*" id="detection-(\d+)"')
OPTION = re.compile(r'<option value="([^"]*)"( selected)?>([^<]*)</option>')


def photo(species: str, day: str, *, star: bool = True, trashed: bool = False) -> int:
    stamp = f"{day}_120000"
    with closing_connection() as conn:
        return _seed_detection(
            conn,
            filename=f"{stamp}_{species}.jpg",
            timestamp=stamp,
            species=species,
            is_favorite=int(star),
            detection_status="rejected" if trashed else "active",
        )


@pytest.fixture
def album(local_db_app: Flask) -> dict[str, int]:
    """Four starred photos of three species, one plain photo, one in the trash."""
    return {
        "tit_old": photo("Parus_major", "20260101"),
        "blackbird": photo("Turdus_merula", "20260102"),
        "magpie": photo("Pica_pica", "20260103"),
        "tit_new": photo("Parus_major", "20260104"),
        "plain": photo("Erithacus_rubecula", "20260105", star=False),
        "trashed": photo("Parus_major", "20260106", trashed=True),
    }


@pytest.fixture
def visitor(local_db_app: Flask) -> FlaskClient:
    return local_db_app.test_client()


@pytest.fixture
def owner(local_db_app: Flask) -> FlaskClient:
    client = local_db_app.test_client()
    with client.session_transaction() as session:
        session["authenticated"] = True
    return client


def tiles(body: str) -> list[int]:
    return [int(found) for found in TILE.findall(body)]


def by_displayed_name(*species: str) -> list[str]:
    return sorted(species, key=lambda key: view_helpers.get_common_name(key).casefold())


def test_every_starred_photo_is_listed_newest_first(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    response = visitor.get("/favorites")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert tiles(body) == [
        album["tit_new"],
        album["magpie"],
        album["blackbird"],
        album["tit_old"],
    ]
    assert "Page 1 of 1 (4 photos)" in " ".join(body.split())


def test_species_list_offers_only_starred_species_by_displayed_name(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    body = visitor.get("/favorites").get_data(as_text=True)
    species = body.split('name="species"')[1].split("</select>")[0]

    expected = by_displayed_name("Parus_major", "Pica_pica", "Turdus_merula")
    assert OPTION.findall(species) == [
        ("", " selected", "All species"),
        *[(key, "", view_helpers.get_common_name(key)) for key in expected],
    ]


def test_one_species_can_be_picked(album: dict[str, int], visitor: FlaskClient) -> None:
    body = visitor.get("/favorites?species=Parus_major").get_data(as_text=True)

    assert tiles(body) == [album["tit_new"], album["tit_old"]]
    assert '<option value="Parus_major" selected>' in body
    assert "Page 1 of 1 (2 photos)" in " ".join(body.split())


def test_sorting_by_species_follows_the_displayed_name(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    order = by_displayed_name("Parus_major", "Pica_pica", "Turdus_merula")
    # Otherwise this could not tell a displayed-name order from a key order.
    assert order != sorted(order)
    newest_first = {
        "Parus_major": [album["tit_new"], album["tit_old"]],
        "Pica_pica": [album["magpie"]],
        "Turdus_merula": [album["blackbird"]],
    }

    body = visitor.get("/favorites?sort=species").get_data(as_text=True)

    assert tiles(body) == [found for key in order for found in newest_first[key]]
    assert '<option value="species" selected>' in body


def test_a_species_without_stars_falls_back_to_all_photos(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    response = visitor.get("/favorites?species=Erithacus_rubecula&sort=species")

    assert response.status_code == 302
    target = urlsplit(response.headers["Location"])
    assert target.path == "/favorites"
    assert parse_qs(target.query) == {"sort": ["species"]}


def test_an_unknown_sort_value_shows_the_newest_first(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    response = visitor.get("/favorites?sort=anything")

    assert response.status_code == 200
    assert tiles(response.get_data(as_text=True))[0] == album["tit_new"]


def test_further_pages_keep_the_filter_and_the_sorting(
    album: dict[str, int], visitor: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("web.blueprints.gallery.PAGE_SIZE", 1)

    first = visitor.get("/favorites?species=Parus_major&sort=species")
    body = first.get_data(as_text=True)

    assert tiles(body) == [album["tit_new"]]
    assert "Page 1 of 2 (2 photos)" in " ".join(body.split())
    pager = body.split('aria-label="Favorites pagination"')[1].split("</nav>")[0]
    links = [
        urlsplit(href.replace("&amp;", "&"))
        for href in re.findall(r'href="([^"]+)"', pager)
    ]
    assert links, "no link to the second page"
    for link in links:
        assert link.path == "/favorites"
        assert parse_qs(link.query) == {
            "page": ["2"],
            "species": ["Parus_major"],
            "sort": ["species"],
        }

    second = visitor.get("/favorites?species=Parus_major&sort=species&page=2")
    assert tiles(second.get_data(as_text=True)) == [album["tit_old"]]


def test_a_page_past_the_end_shows_the_last_page(
    album: dict[str, int], visitor: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("web.blueprints.gallery.PAGE_SIZE", 3)

    body = visitor.get("/favorites?page=99").get_data(as_text=True)

    assert tiles(body) == [album["tit_old"]]
    assert "Page 2 of 2 (4 photos)" in " ".join(body.split())


def test_a_starred_bird_of_unknown_species_is_listed(
    local_db_app: Flask, visitor: FlaskClient
) -> None:
    detection_id = photo("Parus_major", "20260107")
    with closing_connection() as conn:
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename="20260107_120000_Parus_major.jpg",
                detection_id=detection_id,
                species_identity="unknown",
                species_key=None,
            ),
            LabelProvenance(
                "test", "test", "normal_correction", "watchmybirds_ui", "test"
            ),
        )

    body = visitor.get("/favorites").get_data(as_text=True)

    assert tiles(body) == [detection_id]
    unknown = view_helpers.get_common_name(view_helpers.UNKNOWN_SPECIES_KEY)
    assert (
        f'<option value="{view_helpers.UNKNOWN_SPECIES_KEY}">{unknown}</option>' in body
    )
    picked = visitor.get(f"/favorites?species={view_helpers.UNKNOWN_SPECIES_KEY}")
    assert tiles(picked.get_data(as_text=True)) == [detection_id]


def test_each_photo_opens_in_the_viewer_with_its_file_name(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    body = visitor.get("/favorites").get_data(as_text=True)

    for name in ("tit_new", "magpie", "blackbird", "tit_old"):
        assert f'data-bs-target="#modal-favorites-{album[name]}"' in body
        assert f'id="modal-favorites-{album[name]}"' in body
    # The viewer posts corrections by file name; an empty one is refused.
    assert 'data-filename="20260104_120000_Parus_major.jpg"' in body
    assert 'data-filename=""' not in body


def test_a_visitor_can_look_but_neither_change_nor_collect(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    body = visitor.get("/favorites").get_data(as_text=True)

    assert "Batch Edit" not in body
    assert "inlineEditFooter" not in body
    assert "Download Selected" not in body
    stars = re.findall(r'<button[^>]*data-action="favorite"[^>]*>', body)
    assert len(stars) >= 4
    assert all('data-login-required="true"' in star for star in stars)


def test_the_signed_in_owner_can_select_photos_for_download(
    album: dict[str, int], owner: FlaskClient
) -> None:
    body = owner.get("/favorites").get_data(as_text=True)

    assert "Batch Edit" in body
    assert 'id="inlineEditFooter"' in body
    assert "Download Selected" in body
    selectable = re.findall(
        r'<div class="wm-tile[^"]*" id="detection-\d+"[^>]*data-batch-target="detection"',
        body,
    )
    assert len(selectable) == 4
    stars = re.findall(r'<button[^>]*data-action="favorite"[^>]*>', body)
    assert stars and not any("data-login-required" in star for star in stars)


def test_the_menu_leads_to_the_favorites_and_marks_them(
    album: dict[str, int], visitor: FlaskClient
) -> None:
    link = re.compile(r'<a class="([^"]*)"\s+href="/favorites">\s*Favorites\s*</a>')

    elsewhere = link.search(visitor.get("/species").get_data(as_text=True))
    here = link.search(visitor.get("/favorites").get_data(as_text=True))

    assert elsewhere and "active" not in elsewhere.group(1).split()
    assert here and "active" in here.group(1).split()


def test_an_empty_album_says_how_photos_get_there(
    local_db_app: Flask, visitor: FlaskClient
) -> None:
    photo("Parus_major", "20260101", star=False)

    response = visitor.get("/favorites")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert tiles(body) == []
    assert "No favorite photos yet" in body
    assert 'name="species"' not in body
