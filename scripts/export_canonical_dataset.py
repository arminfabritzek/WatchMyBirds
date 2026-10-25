#!/usr/bin/env python3
"""Write the canonical bundle to an explicitly chosen transfer path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_config  # noqa: E402
from utils.db import connection as db_connection  # noqa: E402
from utils.path_manager import PathManager  # noqa: E402
from web.services.canonical_dataset_service import (  # noqa: E402
    create_export,
    get_or_create_labeling_installation_id,
    original_resolver,
    write_canonical_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record waiting training photos and write their canonical ZIP."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Explicit ZIP destination, including filename.",
    )
    args = parser.parse_args()

    path_manager = PathManager(str(get_config()["OUTPUT_DIR"]))
    with db_connection.closing_connection() as conn:
        export_id, bundle = create_export(
            conn,
            paths=path_manager,
            labeling_installation_id=get_or_create_labeling_installation_id(
                str(path_manager.base_dir)
            ),
        )
    destination = write_canonical_bundle(
        bundle,
        path_resolver=original_resolver(path_manager),
        destination=args.destination,
    )
    print(f"{export_id} {bundle.bundle_id} {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
