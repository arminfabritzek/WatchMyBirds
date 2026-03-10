from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_subgallery_all_observation_images_support_thumb_toggle():
    content = (_project_root() / "templates" / "subgallery.html").read_text(
        encoding="utf-8"
    )

    assert 'data-thumb-src="{{ obs.cover_detection.display_url }}"' in content
    assert 'data-full-src="{{ obs.cover_detection.full_url }}"' in content
    assert 'data-thumb-src="{{ det.display_url }}"' in content
    assert 'data-full-src="{{ det.full_url }}"' in content
