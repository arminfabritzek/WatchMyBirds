"""One-shot CLI to score every unscored detection with sharpness/brightness.

Runs the same job the nightly hub does, but from the command line.
Useful for first-time backfill against an existing DB, or after a
restore.

Usage:
    .venv/bin/python scripts/backfill_sharpness.py [--batch N]

The script honours the OUTPUT_DIR / WMB_DB_PATH / WMB_CROPS_ROOT env
vars the rest of the app uses. Interrupt with Ctrl-C — the job
checks the stop event between batches and exits cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

# Allow running from anywhere: insert the repo root on sys.path so
# imports like `from web.services...` resolve. The repo root is the
# parent of this file's directory (scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web.services.nightly_jobs.sharpness_job import SharpnessJob  # noqa: E402

logger = logging.getLogger("backfill_sharpness")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill detections.sharpness_score / crop_brightness."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Batch size for the inner loop (default: 100).",
    )
    parser.add_argument(
        "--reason",
        default="backfill CLI",
        help="Reason string for the log line.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.environ["WMB_SHARPNESS_BATCH_SIZE"] = str(args.batch)
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.warning("Signal %d received — stopping job cleanly", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    job = SharpnessJob()
    rc = job.run(stop_event, reason=args.reason)
    logger.info("backfill done, rc=%s", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
