# ------------------------------------------------------------------------------
# Backup Utilities for WatchMyBirds
# utils/backup.py
# ------------------------------------------------------------------------------
"""
Streaming backup archive generation for migration.
Implements tar.gz streaming without local archive file.
"""

import logging
import sqlite3
import tarfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path

from utils.db import _get_db_path as get_db_path
from utils.path_manager import get_path_manager
from utils.settings import get_settings_path


logger = logging.getLogger(__name__)


def get_backup_stats() -> dict:
    """
    Returns statistics about data to be backed up.

    Returns:
        dict: {
            "db_size_bytes": int,
            "db_size_mb": float,
            "originals_count": int,
            "originals_size_bytes": int,
            "originals_size_mb": float,
            "derivatives_count": int,
            "derivatives_size_bytes": int,
            "derivatives_size_mb": float,
            "settings_exists": bool,
            "settings_size_bytes": int
        }
    """
    pm = get_path_manager()
    stats = {
        "db_size_bytes": 0,
        "db_size_mb": 0.0,
        "originals_count": 0,
        "originals_size_bytes": 0,
        "originals_size_mb": 0.0,
        "derivatives_count": 0,
        "derivatives_size_bytes": 0,
        "derivatives_size_mb": 0.0,
        "settings_exists": False,
        "settings_size_bytes": 0,
    }

    # DB Size
    try:
        db_path = Path(get_db_path())
        if db_path.exists():
            stats["db_size_bytes"] = db_path.stat().st_size
            stats["db_size_mb"] = round(stats["db_size_bytes"] / (1024 * 1024), 2)
    except Exception as e:
        logger.warning(f"Could not get DB size: {e}")

    # Originals
    try:
        if pm.originals_dir.exists():
            for f in pm.originals_dir.rglob("*"):
                if f.is_file():
                    stats["originals_count"] += 1
                    stats["originals_size_bytes"] += f.stat().st_size
            stats["originals_size_mb"] = round(
                stats["originals_size_bytes"] / (1024 * 1024), 2
            )
    except Exception as e:
        logger.warning(f"Could not scan originals: {e}")

    # Derivatives
    try:
        if pm.derivatives_dir.exists():
            for f in pm.derivatives_dir.rglob("*"):
                if f.is_file():
                    stats["derivatives_count"] += 1
                    stats["derivatives_size_bytes"] += f.stat().st_size
            stats["derivatives_size_mb"] = round(
                stats["derivatives_size_bytes"] / (1024 * 1024), 2
            )
    except Exception as e:
        logger.warning(f"Could not scan derivatives: {e}")

    # Settings
    try:
        settings_path = Path(get_settings_path())
        if settings_path.exists():
            stats["settings_exists"] = True
            stats["settings_size_bytes"] = settings_path.stat().st_size
    except Exception as e:
        logger.warning(f"Could not check settings: {e}")

    return stats


def _create_db_snapshot(tmp_path: Path) -> bool:
    """
    Creates a consistent SQLite backup using the backup API.

    Args:
        tmp_path: Path for the temporary DB copy.

    Returns:
        bool: True if successful.
    """
    try:
        source_path = get_db_path()
        if not Path(source_path).exists():
            logger.error("Source DB does not exist.")
            return False

        # Use SQLite backup API for consistency
        source_conn = sqlite3.connect(source_path)
        dest_conn = sqlite3.connect(str(tmp_path))

        source_conn.backup(dest_conn)

        dest_conn.close()
        source_conn.close()

        logger.info(f"DB snapshot created at {tmp_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create DB snapshot: {e}")
        return False


def stream_backup_archive(
    include_db: bool = True,
    include_originals: bool = True,
    include_derivatives: bool = False,
    include_settings: bool = True,
) -> Generator[bytes, None, None]:
    """
    Streams a tar.gz backup archive.

    This generator yields chunks of the archive as they are created,
    without writing a full archive to disk.

    Args:
        include_db: Include images.db
        include_originals: Include originals/ folder
        include_derivatives: Include derivatives/ folder
        include_settings: Include settings.yaml

    Yields:
        bytes: Chunks of the tar.gz archive
    """
    pm = get_path_manager()
    tmp_db_path = None

    try:
        # Create temp DB snapshot if needed
        if include_db:
            tmp_db_path = pm.get_backup_tmp_db_path()
            if not _create_db_snapshot(tmp_db_path):
                raise RuntimeError("Failed to create DB snapshot")

        # Create streaming tar
        buffer = BytesIO()

        with tarfile.open(fileobj=buffer, mode="w|gz") as tar:
            # Add DB
            if include_db and tmp_db_path and tmp_db_path.exists():
                tar.add(str(tmp_db_path), arcname="images.db")
                logger.debug("Added DB to archive")

                # Yield current buffer
                buffer.seek(0)
                data = buffer.read()
                if data:
                    yield data
                buffer.seek(0)
                buffer.truncate(0)

            # Add settings
            if include_settings:
                settings_path = Path(get_settings_path())
                if settings_path.exists():
                    tar.add(str(settings_path), arcname="settings.yaml")
                    logger.debug("Added settings.yaml to archive")

                    buffer.seek(0)
                    data = buffer.read()
                    if data:
                        yield data
                    buffer.seek(0)
                    buffer.truncate(0)

            # Add originals
            if include_originals and pm.originals_dir.exists():
                file_count = 0
                for filepath in pm.originals_dir.rglob("*"):
                    if filepath.is_file():
                        arcname = f"originals/{filepath.relative_to(pm.originals_dir)}"
                        tar.add(str(filepath), arcname=arcname)
                        file_count += 1

                        # Yield every 10 files to keep memory low
                        if file_count % 10 == 0:
                            buffer.seek(0)
                            data = buffer.read()
                            if data:
                                yield data
                            buffer.seek(0)
                            buffer.truncate(0)

                logger.debug(f"Added {file_count} original files to archive")

                # Final yield for originals
                buffer.seek(0)
                data = buffer.read()
                if data:
                    yield data
                buffer.seek(0)
                buffer.truncate(0)

            # Add derivatives
            if include_derivatives and pm.derivatives_dir.exists():
                file_count = 0
                for filepath in pm.derivatives_dir.rglob("*"):
                    if filepath.is_file():
                        arcname = (
                            f"derivatives/{filepath.relative_to(pm.derivatives_dir)}"
                        )
                        tar.add(str(filepath), arcname=arcname)
                        file_count += 1

                        if file_count % 10 == 0:
                            buffer.seek(0)
                            data = buffer.read()
                            if data:
                                yield data
                            buffer.seek(0)
                            buffer.truncate(0)

                logger.debug(f"Added {file_count} derivative files to archive")

        # Final buffer contents (tar footer)
        buffer.seek(0)
        final_data = buffer.read()
        if final_data:
            yield final_data

        logger.info("Backup archive streaming complete")

    except Exception as e:
        logger.error(f"Error streaming backup: {e}")
        raise

    finally:
        # Cleanup temp DB
        if tmp_db_path and tmp_db_path.exists():
            try:
                tmp_db_path.unlink()
                logger.debug(f"Cleaned up temp DB: {tmp_db_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp DB: {e}")
