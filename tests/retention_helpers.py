"""Shared helpers for retention tests.

Builds a *realistic* on-disk + DB layout that matches what
PersistenceService actually writes in production, so tests exercise the
real artifact names instead of fabricated ones:

- original:   originals/<date>/<stem>.jpg
- optimized:  derivatives/optimized/<date>/<stem>.webp
- thumb:      derivatives/thumbs/<date>/<stem>_crop_<N>.webp   (per detection)
- preview:    derivatives/thumbs/<date>/<stem>_preview.webp    (orphan fallback)

and stores ``detections.thumbnail_path = '<stem>_crop_<N>.webp'`` the way
PersistenceService does. Tests describe the artifacts they want; they do
not duplicate path math or insert_detection internals.
"""

from __future__ import annotations

from pathlib import Path

from utils.db.detections import insert_detection


def date_folder(filename: str) -> str:
    return f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"


def seed_image(
    conn,
    filename: str,
    output_dir: Path,
    *,
    review_status: str = "confirmed_bird",
    original_present: int = 1,
    original_deleted_at: str | None = None,
    detections: int = 1,
    favorite: bool = False,
    write_original: bool = True,
    write_optimized: bool = True,
    write_thumbs: bool = True,
    write_preview: bool = False,
    manual_species_override: str | None = None,
    raw_species_name: str | None = None,
    species_source: str | None = None,
    decision_level: str | None = None,
    orig_bytes: int = 1000,
) -> dict:
    """Create one image with production-shaped artifacts + DB rows.

    Returns a dict of the absolute Paths written, keyed:
    ``original``, ``optimized``, ``thumbs`` (list), ``preview``.
    """
    folder = date_folder(filename)
    stem = filename.rsplit(".", 1)[0]
    paths: dict = {"thumbs": []}

    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status, "
        "original_present, original_deleted_at) VALUES (?, ?, ?, ?, ?)",
        (filename, filename[:15], review_status, original_present, original_deleted_at),
    )

    if write_original:
        d = output_dir / "originals" / folder
        d.mkdir(parents=True, exist_ok=True)
        p = d / filename
        p.write_bytes(b"o" * orig_bytes)
        paths["original"] = p

    if write_optimized:
        d = output_dir / "derivatives" / "optimized" / folder
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stem}.webp"
        p.write_bytes(b"optimized-bytes")
        paths["optimized"] = p

    thumbs_dir = output_dir / "derivatives" / "thumbs" / folder
    for i in range(1, detections + 1):
        thumb_name = f"{stem}_crop_{i}.webp"
        det_id = insert_detection(
            conn,
            {
                "image_filename": filename,
                "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.2, "bbox_h": 0.2,
                "od_class_name": "bird", "od_confidence": 0.9,
                "thumbnail_path": thumb_name,
                "manual_species_override": manual_species_override,
                "raw_species_name": raw_species_name,
                "species_source": species_source,
                "decision_level": decision_level,
            },
        )
        conn.execute(
            "UPDATE detections SET status='active', is_favorite=?, rating_source=? "
            "WHERE detection_id=?",
            (1 if favorite else 0, "manual" if favorite else None, det_id),
        )
        if write_thumbs:
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            tp = thumbs_dir / thumb_name
            tp.write_bytes(b"thumb-bytes")
            paths["thumbs"].append(tp)

    if write_preview:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        pp = thumbs_dir / f"{stem}_preview.webp"
        pp.write_bytes(b"preview-bytes")
        paths["preview"] = pp

    conn.commit()
    return paths
