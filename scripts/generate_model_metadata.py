#!/usr/bin/env python3
"""Generate ``model_metadata.json`` from a ``*_model_config.yaml``.

Thin CLI wrapper around :mod:`utils.model_metadata_generator`. The
actual conversion logic lives in that module so the Flask app can
import it directly at runtime (``web/blueprints/api_v1.py``) without
the Docker image having to ship this ``scripts/`` directory.

Usage:
    python scripts/generate_model_metadata.py \\
        <path-to-model_config.yaml> \\
        <path-to-output-model_metadata.json>

Or:
    python scripts/generate_model_metadata.py \\
        --model-dir /opt/app/data/models/object_detection

When ``--model-dir`` is given the script reads ``latest_models.json``
to pick the active variant, derives the matching ``_model_config.yaml``
from the weights filename, and writes ``model_metadata.json`` next to
it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script standalone (outside a proper module context)
# by adding the repo root to sys.path so ``utils.model_metadata_generator``
# resolves. The deploy script invokes this file as an external process.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from utils.model_metadata_generator import config_to_metadata, resolve_active_yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "yaml_path",
        nargs="?",
        help="Path to an *_model_config.yaml (mutually exclusive with --model-dir)",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Where to write model_metadata.json (mutually exclusive with --model-dir)",
    )
    parser.add_argument(
        "--model-dir",
        help="Model directory with latest_models.json; picks active default automatically",
    )
    args = parser.parse_args()

    if args.model_dir and (args.yaml_path or args.output_path):
        parser.error("--model-dir cannot be combined with positional arguments")
    if not args.model_dir and not (args.yaml_path and args.output_path):
        parser.error(
            "Need either --model-dir OR both positional arguments "
            "(yaml_path, output_path)"
        )

    if args.model_dir:
        yaml_path, output_path = resolve_active_yaml(Path(args.model_dir))
    else:
        yaml_path = Path(args.yaml_path)
        output_path = Path(args.output_path)

    config = yaml.safe_load(yaml_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(
            f"{yaml_path}: top-level YAML must be a mapping, got {type(config).__name__}"
        )

    metadata = config_to_metadata(config, source_yaml_name=yaml_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {output_path}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
