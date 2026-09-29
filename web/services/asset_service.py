"""Content versions for packaged browser assets."""

import hashlib
from pathlib import Path


def editor_version(assets_folder: str) -> str:
    return hashlib.sha256(
        (Path(assets_folder) / "js/bird_editor.js").read_bytes()
    ).hexdigest()[:16]
