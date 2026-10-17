"""Measured storage and existing retention protections; never deletes artifacts."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core import training_retention
from core.canonical_dataset import build_canonical_dataset, read_media_size
from core.deletion_plan import deletion_preview, plan_deletions
from core.retention_core import activation_date, settings_for_plan
from utils.db import storage as storage_db
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)

Progress = Callable[[dict[str, Any]], None]
CATEGORIES = (
    "originals",
    "optimized",
    "thumbnails",
    "other_derivatives",
    "sqlite",
    "wal",
    "shm",
    "backups",
    "archives",
    "temporary_exports",
    "restore_temporary",
    "other",
)


def _category(relative: Path) -> str:
    parts = relative.parts
    name = relative.name
    if name in {"images.db", "images.db-wal", "images.db-shm"} and len(parts) == 1:
        return {"images.db": "sqlite", "images.db-wal": "wal", "images.db-shm": "shm"}[
            name
        ]
    if len(parts) == 1 and name.startswith(("images.pre-", "images.db.pre-")):
        return "backups"
    if parts[0] in {"backup", "backups", "backup_before_restore", "code-sync-backups"}:
        return "backups"
    if parts[0] == "originals":
        return "originals"
    if parts[0] == "derivatives":
        return {"optimized": "optimized", "thumbs": "thumbnails"}.get(
            parts[1] if len(parts) > 1 else "", "other_derivatives"
        )
    if parts[0] == "restore_tmp":
        return "restore_temporary"
    if parts[0].startswith("canonical-bundle-"):
        return "temporary_exports"
    if parts[0] in {"archives", "exports"} or name.lower().endswith(
        (".zip", ".tar", ".tar.gz", ".tgz")
    ):
        return "archives"
    return "other"


def _metric() -> dict[str, int]:
    return dict(files=0, bytes=0, allocated_bytes=0)


def _error(errors: list[dict[str, str]], scope: str, kind: str) -> None:
    if len(errors) < 20:
        errors.append(dict(scope=scope, kind=kind))


def unmounted_automount(
    root: Path, mountinfo: Path = Path("/proc/self/mountinfo")
) -> bool:
    # Statting an inactive systemd automount can initiate a slow mount attempt.
    # Read the kernel's mount table before touching optional storage instead.
    try:
        lines = mountinfo.read_text().splitlines()
    except OSError:
        return False
    covering = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        fields = before.split()
        if not separator or len(fields) < 5:
            continue
        mount = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
        kind = after.split()[0]
        if root.absolute().is_relative_to(mount):
            covering.append((len(mount.parts), kind != "autofs", kind))
    return bool(covering and max(covering)[2] == "autofs")


def inventory(
    roots: list[tuple[str, Path, str | None]],
    progress: Progress,
) -> dict[str, Any]:
    categories = {key: _metric() for key in CATEGORIES}
    volumes: dict[int, dict[str, Any]] = {}
    seen_links: set[tuple[int, int]] = set()
    errors: list[dict[str, str]] = []
    scopes = []
    digest = hashlib.sha256()
    done = skipped = output_bytes = 0
    allocated_available = True
    for scope, root, forced_category in roots:
        if unmounted_automount(root):
            scopes.append(
                dict(scope=scope, state="unavailable", reason="inactive_automount")
            )
            continue
        try:
            root_stat = root.stat()
            usage = shutil.disk_usage(root)
            volumes.setdefault(
                root_stat.st_dev,
                dict(
                    scope=scope,
                    capacity_bytes=usage.total,
                    free_bytes=usage.free,
                    used_bytes=usage.used,
                    measured_bytes=0,
                ),
            )
        except OSError as exc:
            scopes.append(dict(scope=scope, state="unavailable"))
            if scope == "output":
                _error(errors, scope, type(exc).__name__)
            continue
        scopes.append(dict(scope=scope, state="measured", device=root_stat.st_dev))

        def walk_error(exc: OSError, label: str = scope) -> None:
            _error(errors, label, type(exc).__name__)

        for directory, folders, files in os.walk(
            root, followlinks=False, onerror=walk_error
        ):
            directory_path = Path(directory)
            safe_folders = []
            for name in sorted(folders):
                if (directory_path / name).is_symlink():
                    skipped += 1
                else:
                    safe_folders.append(name)
            folders[:] = safe_folders
            for name in sorted(files):
                path = directory_path / name
                try:
                    info = path.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        skipped += 1
                        continue
                except OSError as exc:
                    _error(errors, scope, type(exc).__name__)
                    continue
                category = forced_category or _category(path.relative_to(root))
                item = categories[category]
                item["files"] += 1
                item["bytes"] += info.st_size
                if scope == "output":
                    output_bytes += info.st_size
                volumes[root_stat.st_dev]["measured_bytes"] += info.st_size
                if not hasattr(info, "st_blocks"):
                    allocated_available = False
                else:
                    identity = (info.st_dev, info.st_ino)
                    if info.st_nlink <= 1 or identity not in seen_links:
                        item["allocated_bytes"] += info.st_blocks * 512
                        if info.st_nlink > 1:
                            if len(seen_links) < 200000:
                                seen_links.add(identity)
                            else:
                                allocated_available = False
                if category not in {"sqlite", "wal", "shm"}:
                    digest.update(
                        str(
                            (
                                scope,
                                str(path.relative_to(root)),
                                info.st_ino,
                                info.st_size,
                                info.st_mtime_ns,
                                info.st_ctime_ns,
                            )
                        ).encode()
                    )
                done += 1
                if done % 256 == 0:
                    progress(dict(phase="files", files_done=done))
    return dict(
        categories=categories,
        volumes=list(volumes.values()),
        scopes=scopes,
        output_bytes=output_bytes,
        logical_bytes=sum(v["bytes"] for v in categories.values()),
        allocated_bytes=sum(v["allocated_bytes"] for v in categories.values())
        if allocated_available
        else None,
        allocated_status="available" if allocated_available else "unavailable",
        signature=digest.hexdigest(),
        errors=errors,
        skipped_entries=skipped,
    )


def scan_roots(
    path_mgr: PathManager, *, temp_root: Path, backup_root: Path
) -> list[tuple[str, Path, str | None]]:
    output = path_mgr.base_dir.resolve()
    roots: list[tuple[str, Path, str | None]] = [("output", output, None)]
    covered = [output]
    # Only known application export directories are inspected in the shared temp root.
    try:
        exports = sorted(temp_root.glob("canonical-bundle-*"))
    except OSError:
        exports = []
    extra = [("usb_backup", backup_root, "backups")]
    extra.extend(
        (f"temporary_export_{index}", p, "temporary_exports")
        for index, p in enumerate(exports)
    )
    extra.extend(
        (
            f"sibling_{name}",
            output.parent / name,
            "backups" if name != "archives" else "archives",
        )
        for name in ("backups", "code-sync-backups", "archives")
        if (output.parent / name).exists()
    )
    for label, root, category in extra:
        if unmounted_automount(root):
            roots.append((label, root, category))
            continue
        try:
            if root.is_symlink():
                continue
            resolved = root.resolve()
        except OSError:
            roots.append((label, root, category))
            continue
        if any(resolved == path or resolved.is_relative_to(path) for path in covered):
            continue
        roots.append((label, resolved, category))
        covered.append(resolved)
    return roots


def _original(path_mgr: PathManager, filename: str) -> Path | None:
    candidate = path_mgr.get_original_path(filename)
    if (
        candidate.is_symlink()
        or path_mgr.originals_dir.is_symlink()
        or candidate.parent.is_symlink()
    ):
        return None
    return path_mgr.contained_path(candidate, path_mgr.originals_dir)


def _add(target: dict[str, int], size: int, allocated: int) -> None:
    target["files"] += 1
    target["bytes"] += size
    target["unshared_allocated_bytes"] += allocated


def protection_summary(
    conn: Any,
    path_mgr: PathManager,
    settings: dict[str, Any],
    now: datetime,
    progress: Progress,
) -> dict[str, Any]:
    keys = (
        "disabled",
        "too_recent",
        "training_evidence",
        "favorite",
        "presence_marker_conflict",
        "waiting_for_training",
        "unconfirmed_export",
        "training_ready",
        "training_unresolved",
        "historical_facts",
    )
    reasons = {key: dict(files=0, bytes=0, unshared_allocated_bytes=0) for key in keys}
    eligible = dict(files=0, bytes=0, unshared_allocated_bytes=0)
    potential = dict(eligible)
    protected = dict(eligible)
    overlaps: Counter[str] = Counter()
    missing = retired = unsafe = done = accounted_bytes = shared = 0
    errors: list[dict[str, str]] = []
    storage_db.prepare_canonical_views(conn)
    for batch in storage_db.image_batches(conn):
        filenames = [str(row["filename"]) for row in batch]
        storage_db.select_images(conn, filenames)
        sources = training_retention.protection_sources_any(
            conn,
            filenames,
            media_exists=lambda name: training_retention.original_resolver(path_mgr)(
                name
            ).is_file(),
        )
        planned = {
            photo.filename: photo
            for photo in plan_deletions(
                conn,
                settings_for_plan(settings),
                now,
                paths=path_mgr,
                filenames=filenames,
                automatic_deletion_enabled_at=activation_date(settings),
            )
        }
        history = storage_db.historical_images(conn)
        existing: dict[str, tuple[Path, os.stat_result]] = {}
        for image_row in batch:
            name = str(image_row["filename"])
            path = _original(path_mgr, name)
            if path is None:
                unsafe += 1
                continue
            try:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    unsafe += 1
                    continue
                existing[name] = (path, info)
            except FileNotFoundError:
                if image_row["original_present"]:
                    missing += 1
                else:
                    retired += 1
            except OSError as exc:
                _error(errors, "originals", type(exc).__name__)
        bundle = build_canonical_dataset(
            conn,
            labeling_installation_id="",
            media_exists=lambda name, files=existing: name in files,
            media_size=lambda name, files=existing: (
                read_media_size(files[name][0]) if name in files else None
            ),
        )
        ready = {
            str(row["image_filename"])
            for row in bundle.manifest
            if row["decision"] == "included"
            and row["view"] in {"od_positive", "od_negative", "cls_positive"}
        }
        unresolved = {
            str(row["image_filename"])
            for row in bundle.manifest
            if row["decision"] == "included" and row["view"] == "unresolved"
        }
        for row in batch:
            name = str(row["filename"])
            done += 1
            if name not in existing:
                continue
            _, info = existing[name]
            size = info.st_size
            allocated = (
                info.st_blocks * 512
                if hasattr(info, "st_blocks") and info.st_nlink == 1
                else 0
            )
            accounted_bytes += size
            shared += int(info.st_nlink > 1)
            original = next(
                (item for item in planned[name].files if item.category == "original"),
                None,
            )
            due = original is not None and original.decision.due(now)
            blocked = []
            if activation_date(settings) is None:
                blocked.append("disabled")
            if not row["original_present"]:
                blocked.append("presence_marker_conflict")
            if original and original.decision.deadline and not due:
                blocked.append("too_recent")
            if original and original.decision.rule == 3:
                blocked.append("training_evidence")
            if original and original.decision.rule == 2:
                blocked.append("favorite")
            labels = set(blocked) | sources.get(name, set())
            if name in ready:
                labels.add("training_ready")
            if name in unresolved or (
                sources.get(name, set())
                & {"waiting_for_training", "unconfirmed_export"}
                and name not in ready
            ):
                labels.add("training_unresolved")
            if name in history:
                labels.add("historical_facts")
            for label in labels:
                _add(reasons[label], size, allocated)
            _add(
                eligible if due and activation_date(settings) else protected,
                size,
                allocated,
            )
            if due:
                _add(potential, size, allocated)
            if len(blocked) > 1:
                overlaps[" + ".join(sorted(blocked))] += 1
        progress(dict(phase="protection", images_done=done))
    return dict(
        reasons=reasons,
        eligible=eligible,
        eligible_if_enabled=potential,
        protected=protected,
        overlap_groups=dict(overlaps),
        missing_originals=missing,
        retired_originals=retired,
        unsafe_originals=unsafe,
        accounted_original_bytes=accounted_bytes,
        shared_originals=shared,
        errors=errors,
    )


def inspect_storage(
    path_mgr: PathManager,
    settings: dict[str, Any],
    *,
    progress: Progress = lambda _value: None,
    now: datetime | None = None,
    temp_root: Path | None = None,
    backup_root: Path = Path("/mnt/wmb-backup/snapshots"),
) -> dict[str, Any]:
    began = datetime.now(UTC)
    now = now or began
    progress(dict(phase="locating_storage"))
    roots = scan_roots(
        path_mgr,
        temp_root=temp_root or Path(tempfile.gettempdir()),
        backup_root=backup_root,
    )
    first = inventory(roots, progress)
    progress(dict(phase="database"))
    database, protection, preview, error = {}, {}, {}, None
    changed_db = False
    try:
        with closing(storage_db.open_readonly(path_mgr.get_database_path())) as conn:
            version = conn.execute("PRAGMA main.data_version").fetchone()[0]
            database = storage_db.database_summary(conn)
            progress(dict(images_total=database["records"].get("images", 0)))
            preview = deletion_preview(
                conn,
                settings_for_plan(settings),
                now,
                paths=path_mgr,
                automatic_deletion_enabled_at=activation_date(settings),
            )
            preview["enabled"] = activation_date(settings) is not None
            preview["enabled_at"] = (
                settings.get("AUTOMATIC_DELETION_ENABLED_AT")
                if preview["enabled"]
                else None
            )
            protection = protection_summary(conn, path_mgr, settings, now, progress)
            conn.rollback()
            changed_db = (
                conn.execute("PRAGMA main.data_version").fetchone()[0] != version
            )
    except Exception as exc:
        logger.exception("Storage database diagnostics failed")
        error = type(exc).__name__
    progress(dict(phase="verify_files"))
    last = inventory(
        scan_roots(
            path_mgr,
            temp_root=temp_root or Path(tempfile.gettempdir()),
            backup_root=backup_root,
        ),
        progress,
    )
    changed = first["signature"] != last["signature"]
    errors = first["errors"] + last["errors"] + protection.get("errors", [])
    partial = bool(
        errors
        or error
        or changed
        or changed_db
        or last["skipped_entries"]
        or protection.get("unsafe_originals")
        or preview.get("errors")
    )
    if protection:
        protection["unattributed_original_bytes"] = max(
            0,
            last["categories"]["originals"]["bytes"]
            - protection["accounted_original_bytes"],
        )
        protection["current_estimate"] = not partial
    return dict(
        schema_version=1,
        started_at=began.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        status="partial" if partial else "complete",
        automatic_deletion_enabled=activation_date(settings) is not None,
        inventory=last,
        database=database,
        protections=protection,
        deletion_preview=preview,
        files_changed=changed,
        database_changed=changed_db,
        database_error=error,
        errors=errors[:20],
        temporary_export_lifecycle="unavailable",
        collection_mode="manual",
        atomic_filesystem_snapshot=False,
    )
