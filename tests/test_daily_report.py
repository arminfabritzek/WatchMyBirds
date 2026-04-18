from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from utils.daily_report import (
    build_production_collage,
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
