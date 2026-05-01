from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from utils.daily_report import (
    _fetch_species_best_photos,
    build_production_collage,
    build_report_mobile_album,
    build_report_variant_previews,
    render_species_photo_caption,
    render_text_report,
    send_report_variant_previews,
    send_species_best_photos_album,
)


def test_render_text_report_formats_polished_sections():
    message = render_text_report(
        report_date="2026-04-15",
        total_events=245,
        species_count=5,
        top_species_name="Cyanistes_caeruleus",
        top_species_count=7,
    )

    assert "WatchMyBirds" in message
    assert "15.04.2026" in message
    assert "<b>245</b> Events" in message
    assert "<b>5</b> Arten" in message
    assert "Cyanistes caeruleus" in message
    assert "(7x)" in message
    assert "System" not in message
    assert "Dashboard" not in message


def test_render_text_report_handles_no_species():
    message = render_text_report(
        report_date="2026-04-15",
        total_events=0,
        species_count=0,
        top_species_name="—",
        top_species_count=0,
    )

    assert "Keine Arten erkannt." in message
    assert "<b>0</b> Events" in message
    assert "<b>0</b> Arten" in message


def test_render_species_photo_caption_uses_singular_and_escapes_html():
    caption = render_species_photo_caption("Meise & <Co>", 1)

    assert "<b>Meise &amp; &lt;Co&gt;</b>" in caption
    assert "1 Sichtung heute" in caption
    assert "🖼️" not in caption


def test_build_report_variant_previews_renders_local_files():
    with TemporaryDirectory() as tmpdir:
        image_paths = []
        for idx in range(3):
            img = np.full((720, 960, 3), (30 + idx * 20, 90 + idx * 10, 160 - idx * 15), dtype=np.uint8)
            cv2.rectangle(img, (260, 180), (700, 560), (220, 220, 220), thickness=-1)
            path = Path(tmpdir) / f"photo-{idx}.jpg"
            cv2.imwrite(str(path), img)
            image_paths.append(str(path))

        species_photos = [
            {
                "species": f"Species_{idx}",
                "count": idx + 2,
                "best_photo_path": image_paths[idx],
                "bbox_x": 0.27,
                "bbox_y": 0.25,
                "bbox_w": 0.46,
                "bbox_h": 0.5,
            }
            for idx in range(3)
        ]

        variants = build_report_variant_previews(
            species_photos,
            {"Species_0": "Blaumeise", "Species_1": "Kohlmeise", "Species_2": "Rotkehlchen"},
            report_date="2026-04-15",
            output_dir=tmpdir,
        )

        assert len(variants) == 5
        assert [variant["name"] for variant in variants] == [
            "Variante A · Zoom-Collage",
            "Variante B · Vollbild plus Crop",
            "Variante C · Crop-Story Board",
            "Variante D · Drei Zoom-Stufen",
            "Variante E · Weite Kontext-Collage",
        ]
        for variant in variants:
            assert Path(variant["photo_path"]).is_file()
            assert variant["caption"].startswith("<b>Variante ")


def test_send_report_variant_previews_sends_intro_and_each_variant():
    variants = [
        {"caption": "<b>Variante A</b>", "photo_path": "/tmp/a.jpg"},
        {"caption": "<b>Variante B</b>", "photo_path": "/tmp/b.jpg"},
    ]

    with patch("utils.daily_report.send_telegram_message") as mock_send:
        mock_send.side_effect = [[{"ok": True}], [{"ok": True}], [{"ok": True}]]
        responses = send_report_variant_previews(variants)

    assert len(responses) == 3
    assert mock_send.call_count == 3
    assert mock_send.call_args_list[0].args[0].startswith("<b>Abendbericht Varianten-Test</b>")
    assert mock_send.call_args_list[1].kwargs["photo_path"] == "/tmp/a.jpg"
    assert mock_send.call_args_list[2].kwargs["photo_path"] == "/tmp/b.jpg"


def test_send_species_best_photos_album_chunks_media_groups():
    with patch("utils.daily_report.send_telegram_media_group") as mock_send:
        mock_send.side_effect = [[{"ok": True}], [{"ok": True}]]

        species_photos = [
            {
                "species": f"Species_{idx}",
                "count": idx + 1,
                "best_photo_path": f"/tmp/photo-{idx}.jpg",
            }
            for idx in range(11)
        ]

        responses = send_species_best_photos_album(species_photos, {})

    assert responses == [{"ok": True}, {"ok": True}]
    assert mock_send.call_count == 2
    first_chunk = mock_send.call_args_list[0].args[0]
    second_chunk = mock_send.call_args_list[1].args[0]
    assert len(first_chunk) == 10
    assert len(second_chunk) == 1
    assert first_chunk[0]["caption"].startswith("<b>Species 0</b>")


def test_build_production_collage_creates_single_image():
    with TemporaryDirectory() as tmpdir:
        image_paths = []
        for idx in range(4):
            img = np.full((720, 960, 3), (30 + idx * 20, 90 + idx * 10, 160 - idx * 15), dtype=np.uint8)
            cv2.rectangle(img, (260, 180), (700, 560), (220, 220, 220), thickness=-1)
            path = Path(tmpdir) / f"photo-{idx}.jpg"
            cv2.imwrite(str(path), img)
            image_paths.append(str(path))

        species_photos = [
            {
                "species": f"Species_{idx}",
                "count": idx + 2,
                "best_photo_path": image_paths[idx],
                "bbox_x": 0.27,
                "bbox_y": 0.25,
                "bbox_w": 0.46,
                "bbox_h": 0.5,
            }
            for idx in range(4)
        ]

        collage = build_production_collage(
            species_photos,
            {"Species_0": "Blaumeise", "Species_1": "Kohlmeise",
             "Species_2": "Rotkehlchen", "Species_3": "Amsel"},
            report_date="2026-04-15",
            output_dir=tmpdir,
        )

        assert collage is not None
        assert Path(collage["photo_path"]).is_file()
        assert "Abendbericht" in collage["caption"]
        assert "15.04.2026" in collage["caption"]


# Phantom-species filter coverage. Each test seeds a small in-memory-ish
# detections table on tmp_path, then asserts what survives the report
# query. The seed data mirrors the real-world failure modes captured from
# the RPi snapshot (catalog-orphan genus fallbacks; unconfirmed low-score
# rows; legitimate confirmed rows that should pass).

@pytest.fixture
def report_db(tmp_path, monkeypatch):
    """Create a fresh sqlite DB with the schema needed for report queries.

    Returns the open connection. Caller seeds rows and calls
    _fetch_species_best_photos directly.
    """
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())

    from utils.db.connection import closing_connection

    with closing_connection() as conn:
        # Seed a stub image so the FK constraint on detections.image_filename
        # is satisfied. The date prefix has to match what the report query
        # filters on (YYYYMMDD format).
        conn.execute(
            "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
            ("20260430_120000_stub.jpg", "20260430_120000"),
        )
        conn.commit()
        yield conn


def _seed_detection(
    conn,
    *,
    species: str,
    decision_state: str,
    score: float,
    image_filename: str = "20260430_120000_stub.jpg",
) -> int:
    """Insert a detection with a specific species (via od_class_name fallback)
    and decision_state. Returns the detection_id."""
    cur = conn.execute(
        """
        INSERT INTO detections
            (image_filename, od_class_name, od_confidence, score,
             decision_state, status, bbox_x, bbox_y, bbox_w, bbox_h,
             bbox_quality)
        VALUES (?, ?, ?, ?, ?, 'active', 0.1, 0.1, 0.2, 0.2, 0.8)
        """,
        (image_filename, species, score, score, decision_state),
    )
    return cur.lastrowid


def test_build_report_mobile_album_appends_collector_card_last():
    """Album builder must finish every report with a single collector
    card whose caption identifies it as 'Sammelkarte'. That's the
    deck-of-cards finisher the operator opens last in the Telegram
    media group."""
    with TemporaryDirectory() as tmpdir:
        img = np.full((600, 800, 3), (40, 90, 60), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (600, 450), (220, 220, 220), thickness=-1)
        photo_paths = []
        for i in range(3):
            p = Path(tmpdir) / f"photo-{i}.jpg"
            cv2.imwrite(str(p), img)
            photo_paths.append(str(p))

        species_photos = [
            {
                "species": s,
                "count": c,
                "best_photo_path": photo_paths[i],
                "bbox_x": 0.30, "bbox_y": 0.25, "bbox_w": 0.45, "bbox_h": 0.55,
            }
            for i, (s, c) in enumerate(
                [
                    ("Columba_palumbus", 7),
                    ("Cyanistes_caeruleus", 4),
                    ("Parus_major", 12),
                ]
            )
        ]
        common_names = {
            "Columba_palumbus": "Ringeltaube",
            "Cyanistes_caeruleus": "Blaumeise",
            "Parus_major": "Kohlmeise",
        }

        tiles = build_report_mobile_album(
            species_photos,
            common_names,
            report_date="2026-04-30",
            output_dir=tmpdir,
        )

        # 3 species + 1 collector card.
        assert len(tiles) == 4
        # The collector card sits last and is labelled in the caption so
        # downstream surfaces (and the operator) can identify it.
        assert "Sammelkarte" in tiles[-1]["caption"]
        # And it carries the daily species count.
        assert "3 Arten" in tiles[-1]["caption"]
        # Earlier tiles must NOT carry the Sammelkarte label.
        for tile in tiles[:-1]:
            assert "Sammelkarte" not in tile["caption"]
        # Every tile, including the collector card, is 1080x1080.
        for tile in tiles:
            rendered = cv2.imread(tile["photo_path"])
            assert rendered is not None
            assert rendered.shape == (1080, 1080, 3)


def test_build_report_mobile_album_renders_square_1080_cards():
    """Every mobile tile must be a 1080x1080 standalone post-ready square.
    Asserts the achievement-card layout contract end-to-end via the
    public album builder, not the internal renderer.

    With one species in, the album returns 2 tiles: the species card
    and the closing collector card.
    """
    with TemporaryDirectory() as tmpdir:
        img = np.full((600, 800, 3), (40, 90, 60), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (600, 450), (220, 220, 220), thickness=-1)
        path = Path(tmpdir) / "photo-0.jpg"
        cv2.imwrite(str(path), img)

        species_photos = [
            {
                "species": "Columba_palumbus",
                "count": 5,
                "best_photo_path": str(path),
                "bbox_x": 0.30, "bbox_y": 0.25, "bbox_w": 0.45, "bbox_h": 0.55,
            },
        ]

        tiles = build_report_mobile_album(
            species_photos,
            {"Columba_palumbus": "Ringeltaube"},
            report_date="2026-04-30",
            output_dir=tmpdir,
        )

        # 1 species card + 1 collector card.
        assert len(tiles) == 2
        for tile in tiles:
            rendered = cv2.imread(tile["photo_path"])
            assert rendered is not None
            assert rendered.shape == (1080, 1080, 3)


def test_build_report_mobile_album_pluralises_count_and_drops_pagination():
    """Mobile tile captions should read '1 Sichtung' / 'N Sichtungen' and
    must NOT carry the legacy '1/N' pagination tag."""
    with TemporaryDirectory() as tmpdir:
        image_paths = []
        for idx in range(2):
            img = np.full((600, 800, 3), (40, 90, 60), dtype=np.uint8)
            cv2.rectangle(img, (200, 150), (600, 450), (220, 220, 220), thickness=-1)
            path = Path(tmpdir) / f"photo-{idx}.jpg"
            cv2.imwrite(str(path), img)
            image_paths.append(str(path))

        species_photos = [
            {
                "species": "Columba_palumbus",
                "count": 7,
                "best_photo_path": image_paths[0],
                "bbox_x": 0.27, "bbox_y": 0.25, "bbox_w": 0.46, "bbox_h": 0.5,
            },
            {
                "species": "Dendrocopos_major",
                "count": 1,
                "best_photo_path": image_paths[1],
                "bbox_x": 0.27, "bbox_y": 0.25, "bbox_w": 0.46, "bbox_h": 0.5,
            },
        ]

        tiles = build_report_mobile_album(
            species_photos,
            {"Columba_palumbus": "Ringeltaube", "Dendrocopos_major": "Buntspecht"},
            report_date="2026-04-30",
            output_dir=tmpdir,
        )

    # 2 species cards + 1 collector card.
    assert len(tiles) == 3
    # Plural for count > 1, singular for count == 1.
    assert "7 Sichtungen" in tiles[0]["caption"]
    assert "1 Sichtung" in tiles[1]["caption"]
    assert "1 Sichtungen" not in tiles[1]["caption"]  # no plural-on-one
    # Collector card sits last with its own caption shape.
    assert "Sammelkarte" in tiles[-1]["caption"]
    # The "1/2" pagination tag is gone from species cards — Telegram
    # album order replaces it.
    for tile in tiles[:-1]:
        assert "1/2" not in tile["caption"]
        assert "2/2" not in tile["caption"]


def test_fetch_species_best_photos_drops_unconfirmed(report_db, tmp_path):
    """Detections with decision_state != 'confirmed' must not reach Telegram.

    This is the gallery-parity gate: the gallery already hides these rows;
    the daily report previously did not, which surfaced phantom species.
    """
    # Confirmed -> should appear
    _seed_detection(
        report_db, species="Parus_major", decision_state="confirmed", score=0.95
    )
    # Unconfirmed (uncertain / unknown) -> must NOT appear
    _seed_detection(
        report_db, species="Pyrrhula_pyrrhula", decision_state="uncertain", score=0.65
    )
    _seed_detection(
        report_db, species="Sitta_europaea", decision_state="unknown", score=0.45
    )
    report_db.commit()

    # Image file existence check is bypassed by patching get_path_manager —
    # we only care which species survive the SQL+catalog filter, not whether
    # the JPEG is on disk.
    with patch("utils.daily_report.get_path_manager") as mock_pm:
        mock_pm.return_value.get_original_path.return_value = (
            tmp_path / "fake.jpg"
        )
        # Make os.path.isfile return True for the fake path so we don't drop
        # results on the disk check.
        with patch("utils.daily_report.os.path.isfile", return_value=True):
            results = _fetch_species_best_photos(report_db, "2026-04-30")

    species_in_report = {r["species"] for r in results}
    assert species_in_report == {"Parus_major"}


def test_fetch_species_best_photos_drops_catalog_orphans(report_db, tmp_path):
    """Classifier genus-fallbacks (Phoenicurus_sp., Passer_sp.) and stale
    class names must not reach Telegram even when decision_state=confirmed.

    The whitelist comes from common_names + extended_species + the non-bird
    OD class set. Anything else is dropped at the Python pass.
    """
    # Confirmed valid -> survives
    _seed_detection(
        report_db, species="Cyanistes_caeruleus", decision_state="confirmed", score=0.9
    )
    # Confirmed catalog-orphan -> dropped
    _seed_detection(
        report_db, species="Phoenicurus_sp.", decision_state="confirmed", score=0.7
    )
    _seed_detection(
        report_db, species="Passer_sp.", decision_state="confirmed", score=0.7
    )
    # Confirmed nonsense (typo / stale class) -> dropped
    _seed_detection(
        report_db, species="not_a_real_species", decision_state="confirmed", score=0.7
    )
    # Confirmed non-bird OD class (cat) -> survives, OD label IS the species
    _seed_detection(
        report_db, species="cat", decision_state="confirmed", score=0.85
    )
    report_db.commit()

    with patch("utils.daily_report.get_path_manager") as mock_pm:
        mock_pm.return_value.get_original_path.return_value = (
            tmp_path / "fake.jpg"
        )
        with patch("utils.daily_report.os.path.isfile", return_value=True):
            results = _fetch_species_best_photos(report_db, "2026-04-30")

    species_in_report = {r["species"] for r in results}
    assert species_in_report == {"Cyanistes_caeruleus", "cat"}


def test_fetch_species_best_photos_respects_min_confirmed_observations(
    report_db, tmp_path, monkeypatch
):
    """TELEGRAM_MIN_CONFIRMED_OBSERVATIONS=2 drops species with a single
    confirmed sighting in the report window.

    Operators raise this threshold when they want even tighter rarity
    filtering than 'one confirmed frame'. The default is 1 so this stays
    a deliberate operator choice.
    """
    # 3 confirmed Parus -> survives at threshold 2
    for _ in range(3):
        _seed_detection(
            report_db, species="Parus_major", decision_state="confirmed", score=0.9
        )
    # 1 confirmed Garrulus -> dropped at threshold 2
    _seed_detection(
        report_db, species="Garrulus_glandarius", decision_state="confirmed", score=0.95
    )
    report_db.commit()

    # Override the threshold via the live config that _fetch_species_best_photos
    # reads. We monkeypatch the get_config call site, not the global, so we
    # don't pollute neighbouring tests.
    real_get_config = __import__("config").get_config

    def stricter_config():
        cfg = dict(real_get_config())
        cfg["TELEGRAM_MIN_CONFIRMED_OBSERVATIONS"] = 2
        return cfg

    monkeypatch.setattr("utils.daily_report.get_config", stricter_config)

    with patch("utils.daily_report.get_path_manager") as mock_pm:
        mock_pm.return_value.get_original_path.return_value = (
            tmp_path / "fake.jpg"
        )
        with patch("utils.daily_report.os.path.isfile", return_value=True):
            results = _fetch_species_best_photos(report_db, "2026-04-30")

    species_in_report = {r["species"] for r in results}
    assert species_in_report == {"Parus_major"}
    assert "Garrulus_glandarius" not in species_in_report
