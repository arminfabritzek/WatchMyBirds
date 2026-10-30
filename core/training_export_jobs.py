"""Durable, bounded preparation of canonical training ZIP parts."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from core.canonical_archive import stream_canonical_bundle
from core.canonical_dataset import (
    CanonicalDataset,
    build_canonical_dataset,
    read_media_size,
)
from core.human_label_core import get_or_create_labeling_installation_id
from core.training_export import (
    _selection_filenames,
    answer_revisions,
    load_export,
    original_resolver,
    waiting_revisions,
)
from utils.db.connection import closing_connection
from utils.db.storage import image_batches
from utils.path_manager import PathManager

logger = logging.getLogger(__name__)
ACTIVE = ("queued", "running", "paused", "cancelling")


class ExportCancelled(Exception):
    pass


class ExportSpaceUnavailable(Exception):
    pass


def _stamp() -> str:
    return datetime.now(UTC).isoformat()


class TrainingExportJobs:
    def __init__(
        self,
        output_dir: str,
        *,
        photos_per_part: int = 1000,
        bytes_per_part: int = 1024**3,
        staging_bytes: int = 2 * 1024**3,
        reserve_bytes: int = 256 * 1024**2,
    ) -> None:
        self.paths = PathManager(output_dir)
        self.photos_per_part = photos_per_part
        self.bytes_per_part = bytes_per_part
        self.staging_bytes = staging_bytes
        self.reserve_bytes = reserve_bytes
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        with closing_connection():
            pass
        self._ensure_worker()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.paths.get_database_path().resolve().as_uri() + "?mode=rw",
            uri=True,
            timeout=15,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _root(self) -> Path:
        root = self.paths.get_training_export_dir()
        checked = self.paths.contained_path(root, self.paths.base_dir)
        if checked != self.paths.base_dir.resolve() / root.name:
            raise ValueError("Unsafe export directory")
        checked.mkdir(mode=0o700, parents=True, exist_ok=True)
        return checked

    def _file(self, job_id: str, name: str) -> Path:
        directory = self.paths.get_training_export_dir(job_id)
        root = self._root()
        checked = self.paths.contained_path(directory, root)
        if checked != root / job_id or not job_id:
            raise ValueError("Unsafe export job directory")
        checked.mkdir(mode=0o700, exist_ok=True)
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Invalid export artifact")
        path = self.paths.contained_path(checked / name, checked)
        if path != checked / name:
            raise ValueError("Unsafe export artifact")
        return path

    def start(
        self, selection: set[str] | None = None, *, rebuild_export_id: str | None = None
    ) -> str:
        job_id = uuid.uuid4().hex
        with self.connection() as conn:
            if rebuild_export_id is not None:
                row = conn.execute(
                    "SELECT length(bundle_json) FROM labeling_exports WHERE export_id=?",
                    (rebuild_export_id,),
                ).fetchone()
                if row is None or row[0] > 32 * 1024**2:
                    raise ValueError(
                        "This export cannot be rebuilt within the part limit"
                    )
            if selection is not None:
                names = _selection_filenames(conn, selection)
                resolve = original_resolver(self.paths)
                revisions = waiting_revisions(
                    conn,
                    filenames=names,
                    media_exists=lambda name: resolve(name).is_file(),
                )
                if (
                    not selection
                    or {r["selection_key"] for r in revisions} & selection != selection
                ):
                    raise ValueError("Select current answers waiting for training")
            try:
                conn.execute(
                    """INSERT INTO training_export_jobs
                    (job_id,created_at,updated_at,status,selection_json,phase) VALUES (?,?,?,'queued',?,?)""",
                    (
                        job_id,
                        _stamp(),
                        _stamp(),
                        json.dumps({"rebuild": rebuild_export_id})
                        if rebuild_export_id
                        else (
                            json.dumps(sorted(selection))
                            if selection is not None
                            else None
                        ),
                        "rebuild" if rebuild_export_id else "queued",
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "Another export is already preparing; finish or cancel it first"
                ) from exc
        self._ensure_worker()
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_export_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError("Export job not found")
            job = {key: row[key] for key in row.keys() if key != "selection_json"}
            job["is_rebuild"] = isinstance(
                json.loads(row["selection_json"] or "null"), dict
            )
            if job["phase"] == "snapshot":
                progress_path = self._file(job_id, "progress.json")
                try:
                    with progress_path.open() as progress:
                        job.update(json.loads(progress.read(1024)))
                except (FileNotFoundError, ValueError):
                    pass
            if job["status"] in {"queued", "running", "cancelling"}:
                self._ensure_worker()
            job["parts"] = [
                dict(part)
                for part in conn.execute(
                    """SELECT p.*, e.received_at, COUNT(m.object_id) AS answer_count
                FROM training_export_parts p JOIN labeling_exports e USING(export_id)
                LEFT JOIN labeling_export_members m USING(export_id)
                WHERE (p.job_id=? OR p.artifact_job_id=?) AND (p.available=1 OR e.received_at IS NULL)
                GROUP BY p.export_id ORDER BY p.part_number LIMIT 20""",
                    (job_id, job_id),
                )
            ]
            job["part_count"] = conn.execute(
                "SELECT COUNT(*) FROM training_export_parts WHERE job_id=? OR artifact_job_id=?",
                (job_id, job_id),
            ).fetchone()[0]
            return job

    def recent(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT job_id FROM training_export_jobs ORDER BY created_at DESC LIMIT 5"
                )
            ]
        return [self.status(job_id) for job_id in ids]

    def cancel(self, job_id: str) -> None:
        with self.connection() as conn:
            result = conn.execute(
                "UPDATE training_export_jobs SET status='cancelling',updated_at=? WHERE job_id=? AND status IN ('queued','running','paused')",
                (_stamp(), job_id),
            )
            if not result.rowcount:
                raise ValueError("This export is no longer preparing")
        self._ensure_worker()

    def retry(self, job_id: str) -> None:
        with self.connection() as conn:
            try:
                changed = conn.execute(
                    "UPDATE training_export_jobs SET status='queued',error=NULL,updated_at=? WHERE job_id=? AND status IN ('failed','paused','cancelled')",
                    (_stamp(), job_id),
                ).rowcount
            except sqlite3.IntegrityError as exc:
                raise ValueError("Another export is already preparing") from exc
            if not changed:
                raise ValueError("This export cannot be resumed")
        self._ensure_worker()

    def _update(self, job_id: str, **values: Any) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE training_export_jobs SET "
                + ",".join(f"{key}=?" for key in values)
                + ",updated_at=? WHERE job_id=?",
                (*values.values(), _stamp(), job_id),
            )

    def _check(self, job_id: str, required_bytes: int = 0) -> None:
        if self._stop.is_set():
            raise ExportCancelled
        with self.connection() as conn:
            row = conn.execute(
                "SELECT status FROM training_export_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None or row[0] == "cancelling":
            raise ExportCancelled
        if shutil.disk_usage(self._root()).free < self.reserve_bytes + required_bytes:
            raise ExportSpaceUnavailable

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="training-export", daemon=True
                )
                self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._lock:
                thread = self._thread
            if thread is None:
                return
            remaining = (
                max(0, deadline - time.monotonic()) if deadline is not None else None
            )
            thread.join(remaining)
            if deadline is not None and time.monotonic() >= deadline:
                return

    def close(self) -> None:
        self._stop.set()
        self.wait(5)

    def _run(self) -> None:
        owned = False
        healthy = True
        try:
            lock_fd = os.open(
                self._root() / ".worker.lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(lock_fd, "ab") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                owned = True
                self.cleanup()
                with self.connection() as conn:
                    row = conn.execute(
                        "SELECT job_id FROM training_export_jobs WHERE status IN ('queued','running','cancelling') ORDER BY created_at LIMIT 1"
                    ).fetchone()
                if row is None:
                    return
                job_id = row[0]
                try:
                    self._check(job_id)
                    self._update(job_id, status="running", error=None)
                    self._prepare(job_id)
                except ExportSpaceUnavailable:
                    self._update(
                        job_id,
                        status="paused",
                        error="More staging space is needed. Download and confirm prepared parts, remove their local ZIPs, then Resume.",
                    )
                except ExportCancelled:
                    self._update(
                        job_id,
                        status="queued" if self._stop.is_set() else "cancelled",
                        error=None,
                    )
                except Exception as error:
                    logger.exception("Training export preparation failed")
                    no_space = isinstance(error, OSError) and error.errno in {
                        errno.ENOSPC,
                        errno.EDQUOT,
                    }
                    self._update(
                        job_id,
                        status="paused" if no_space else "failed",
                        error="Disk full. Free space, then Resume; completed parts remain available."
                        if no_space
                        else "Export preparation failed; see local logs. Retry keeps the same snapshot.",
                    )
                finally:
                    self._remove(job_id, "archive.partial")
                    self._remove(job_id, "progress.partial")
                    for archive in self._file(job_id, "snapshot.db").parent.glob(
                        "*.zip"
                    ):
                        with self.connection() as conn:
                            published = conn.execute(
                                "SELECT 1 FROM training_export_parts WHERE (job_id=? OR artifact_job_id=?) AND export_id=?",
                                (job_id, job_id, archive.stem),
                            ).fetchone()
                        if not published:
                            self._remove(job_id, archive.name)
        except Exception:
            healthy = False
            logger.exception("Training export worker could not start")
        finally:
            with self._lock:
                self._thread = None
                if owned and healthy and not self._stop.is_set():
                    try:
                        with self.connection() as conn:
                            pending = conn.execute(
                                "SELECT 1 FROM training_export_jobs WHERE status IN ('queued','running','cancelling') LIMIT 1"
                            ).fetchone()
                    except sqlite3.Error:
                        logger.info("Export worker stopped after database replacement")
                        pending = None
                else:
                    pending = None
            if pending:
                self._ensure_worker()

    def _snapshot(self, job_id: str) -> Path:
        snapshot = self._file(job_id, "snapshot.db")
        with self.connection() as conn:
            row = conn.execute(
                "SELECT snapshot_ready FROM training_export_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row[0]:
                if not snapshot.is_file():
                    raise ValueError("The export snapshot is unavailable")
                return snapshot
            estimate = (
                conn.execute("PRAGMA page_count").fetchone()[0]
                * conn.execute("PRAGMA page_size").fetchone()[0]
            )
            self._check(job_id, estimate + 4 * 1024**2)
            self._update(job_id, phase="snapshot", bytes_done=0, bytes_total=estimate)
            for suffix in ("", "-wal", "-shm"):
                self._remove(job_id, "snapshot.db" + suffix)
            conn.execute("BEGIN")
            with closing(sqlite3.connect(snapshot)) as target:
                last_progress = 0.0
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]

                def progress(status: int, remaining: int, total: int) -> None:
                    nonlocal last_progress
                    if not remaining or time.monotonic() - last_progress > 0.5:
                        self._check(job_id)
                        # Avoid progress writes growing the WAL while backup pins a read view.
                        pending = self._file(job_id, "progress.partial")
                        self._remove(job_id, "progress.partial")
                        with pending.open("x") as output:
                            json.dump(
                                dict(
                                    bytes_done=(total - remaining) * page_size,
                                    bytes_total=total * page_size,
                                ),
                                output,
                            )
                        pending.replace(self._file(job_id, "progress.json"))
                        last_progress = time.monotonic()

                conn.backup(target, pages=256, progress=progress)
                target.execute(
                    "CREATE TABLE export_candidates(filename TEXT PRIMARY KEY, bytes INTEGER NOT NULL)"
                )
                target.commit()
            self._update(job_id, snapshot_ready=1, phase="selecting")
            self._remove(job_id, "progress.json")
        return snapshot

    def _prepare(self, job_id: str) -> None:
        with self.connection() as conn:
            payload = conn.execute(
                "SELECT selection_json FROM training_export_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()[0]
        if payload and isinstance(json.loads(payload), dict):
            self._rebuild(job_id, json.loads(payload)["rebuild"])
            return
        snapshot_path = self._snapshot(job_id)
        with closing(sqlite3.connect(snapshot_path)) as snapshot:
            snapshot.row_factory = sqlite3.Row
            with self.connection() as live:
                job = dict(
                    live.execute(
                        "SELECT * FROM training_export_jobs WHERE job_id=?", (job_id,)
                    ).fetchone()
                )
            resolve = original_resolver(self.paths)
            if job["phase"] in {"selecting", "snapshot", "queued"}:
                snapshot.execute("DELETE FROM export_candidates")
                selection = (
                    set(json.loads(job["selection_json"]))
                    if job["selection_json"] is not None
                    else None
                )
                selected_names = (
                    _selection_filenames(snapshot, selection)
                    if selection is not None
                    else None
                )
                self._update(
                    job_id,
                    processed=0,
                    total=len(selected_names)
                    if selected_names is not None
                    else snapshot.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                    bytes_done=0,
                    bytes_total=0,
                    missing=0,
                )
                scanned = missing = 0
                for batch in (
                    image_batches(snapshot)
                    if selected_names is None
                    else (
                        list(
                            {"filename": name}
                            for name in selected_names[start : start + 200]
                        )
                        for start in range(0, len(selected_names), 200)
                    )
                ):
                    self._check(job_id)
                    names = [row["filename"] for row in batch]
                    revisions = waiting_revisions(
                        snapshot, filenames=names, media_exists=lambda _: True
                    )
                    candidates = {
                        str(row["image_filename"])
                        for row in revisions
                        if selection is None or row["selection_key"] in selection
                    }
                    for name in sorted(candidates):
                        try:
                            info = resolve(name).stat()
                        except FileNotFoundError:
                            missing += 1
                            continue
                        if not stat.S_ISREG(info.st_mode):
                            missing += 1
                            continue
                        snapshot.execute(
                            "INSERT INTO export_candidates VALUES (?,?)",
                            (name, info.st_size),
                        )
                    snapshot.commit()
                    scanned += len(batch)
                    self._update(job_id, processed=scanned, missing=missing)
                count = snapshot.execute(
                    "SELECT COUNT(*) FROM export_candidates"
                ).fetchone()[0]
                self._update(job_id, phase="archives", total=count, processed=0)
            installation_id = get_or_create_labeling_installation_id(
                str(self.paths.base_dir)
            )
            while True:
                self._check(job_id)
                with self.connection() as live:
                    job = dict(
                        live.execute(
                            "SELECT * FROM training_export_jobs WHERE job_id=?",
                            (job_id,),
                        ).fetchone()
                    )
                    staged = live.execute(
                        "SELECT COALESCE(SUM(archive_bytes),0) FROM training_export_parts WHERE available=1"
                    ).fetchone()[0]
                    part_number = live.execute(
                        "SELECT COUNT(*)+1 FROM training_export_parts WHERE job_id=?",
                        (job_id,),
                    ).fetchone()[0]
                candidates = snapshot.execute(
                    "SELECT * FROM export_candidates WHERE filename>? ORDER BY filename LIMIT ?",
                    (job["cursor"], self.photos_per_part),
                ).fetchall()
                if not candidates:
                    break
                names: list[str] = []
                estimate = 0
                records = 0
                for row in candidates:
                    size = max(0, row["bytes"])
                    if size + 4 * 1024**2 > self.staging_bytes:
                        raise ValueError("One photo exceeds the export staging limit")
                    name = row["filename"]
                    count = snapshot.execute(
                        "SELECT COUNT(*) FROM human_label_facts f JOIN label_subjects s USING(subject_id) WHERE s.image_filename=?",
                        (name,),
                    ).fetchone()[0]
                    count += snapshot.execute(
                        "SELECT COUNT(*) FROM detections WHERE image_filename=?",
                        (name,),
                    ).fetchone()[0]
                    count += snapshot.execute(
                        "SELECT COUNT(*) FROM manual_object_revisions r JOIN manual_objects m USING(manual_object_id) WHERE m.image_filename=?",
                        (name,),
                    ).fetchone()[0]
                    if count > 20000:
                        raise ValueError("One photo exceeds the export metadata limit")
                    if names and (
                        estimate + size > self.bytes_per_part or records + count > 20000
                    ):
                        break
                    names.append(name)
                    estimate += size
                    records += count
                if staged + estimate + 4 * 1024**2 > self.staging_bytes:
                    raise ExportSpaceUnavailable
                self._check(job_id, estimate + 4 * 1024**2)
                bundle = build_canonical_dataset(
                    snapshot,
                    labeling_installation_id=installation_id,
                    filenames=set(names),
                    media_exists=lambda name: resolve(name).is_file(),
                    media_size=lambda name: read_media_size(resolve(name)),
                )
                members = answer_revisions(snapshot, sorted(bundle.media_filenames))
                bundle.snapshot.update(export_job_id=job_id, part_number=part_number)
                export_id = uuid.uuid4().hex
                archive, metadata = self._write_archive(
                    job_id, bundle, resolve, export_id
                )
                with self.connection() as live:
                    live.execute("BEGIN IMMEDIATE")
                    state = live.execute(
                        "SELECT status FROM training_export_jobs WHERE job_id=?",
                        (job_id,),
                    ).fetchone()[0]
                    if state == "cancelling":
                        raise ExportCancelled
                    number = live.execute(
                        "SELECT COUNT(*)+1 FROM training_export_parts WHERE job_id=?",
                        (job_id,),
                    ).fetchone()[0]
                    live.execute(
                        "INSERT INTO labeling_exports(export_id,created_at,bundle_json) VALUES (?,?,?)",
                        (
                            export_id,
                            _stamp(),
                            metadata,
                        ),
                    )
                    live.executemany(
                        "INSERT INTO labeling_export_members VALUES (?,?,?,?,?)",
                        [
                            (
                                export_id,
                                m["image_filename"],
                                m["kind"],
                                m["object_id"],
                                m["revision_json"],
                            )
                            for m in members
                        ],
                    )
                    live.execute(
                        "INSERT INTO training_export_parts(export_id,job_id,part_number,archive_bytes,created_at,available) VALUES (?,?,?,?,?,1)",
                        (export_id, job_id, number, archive.stat().st_size, _stamp()),
                    )
                    live.execute(
                        "UPDATE training_export_jobs SET cursor=?,processed=processed+?,missing=missing+?,updated_at=? WHERE job_id=?",
                        (
                            names[-1],
                            len(names),
                            len(names) - len(bundle.media_filenames),
                            _stamp(),
                            job_id,
                        ),
                    )
            self._update(job_id, status="complete", phase="complete", snapshot_ready=0)
        self._remove(job_id, "snapshot.db")

    def _rebuild(self, job_id: str, export_id: str) -> None:
        with self.connection() as conn:
            bundle, media_map = load_export(conn, export_id)
            available = conn.execute(
                "SELECT available FROM training_export_parts WHERE export_id=?",
                (export_id,),
            ).fetchone()
        if available is not None and available[0]:
            raise ValueError("This ZIP is already available")
        resolve = original_resolver(self.paths)
        self._update(job_id, total=len(bundle.media_filenames), phase="rebuild")
        archive, _ = self._write_archive(
            job_id, bundle, lambda name: resolve(media_map.get(name, name)), export_id
        )
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if (
                conn.execute(
                    "SELECT status FROM training_export_jobs WHERE job_id=?", (job_id,)
                ).fetchone()[0]
                == "cancelling"
            ):
                raise ExportCancelled
            conn.execute(
                """INSERT INTO training_export_parts
                (export_id,job_id,part_number,archive_bytes,created_at,available,artifact_job_id)
                VALUES (?,?,1,?,?,1,?) ON CONFLICT(export_id) DO UPDATE SET
                available=1,artifact_job_id=excluded.artifact_job_id,archive_bytes=excluded.archive_bytes,created_at=excluded.created_at""",
                (export_id, job_id, archive.stat().st_size, _stamp(), job_id),
            )
            conn.execute(
                "UPDATE training_export_jobs SET status='complete',phase='complete',processed=total,updated_at=? WHERE job_id=?",
                (_stamp(), job_id),
            )

    def _write_archive(
        self,
        job_id: str,
        bundle: CanonicalDataset,
        resolve: Callable[[str], Path],
        export_id: str,
    ) -> tuple[Path, str]:
        metadata = json.dumps(asdict(bundle), sort_keys=True, separators=(",", ":"))
        metadata_bytes = len(metadata.encode())
        if metadata_bytes > 32 * 1024**2:
            raise ValueError("Export metadata exceeds the per-part limit")
        estimate = (
            sum(resolve(name).stat().st_size for name in bundle.media_filenames)
            + metadata_bytes
            + 4 * 1024**2
        )
        if estimate > self.staging_bytes:
            raise ValueError("A photo exceeds the export staging limit")
        with self.connection() as conn:
            staged = conn.execute(
                "SELECT COALESCE(SUM(archive_bytes),0) FROM training_export_parts WHERE available=1"
            ).fetchone()[0]
        if staged + estimate > self.staging_bytes:
            raise ExportSpaceUnavailable
        self._check(job_id, estimate)
        self._update(job_id, bytes_done=0, bytes_total=estimate)
        partial = self._file(job_id, "archive.partial")
        self._remove(job_id, "archive.partial")
        last_check = 0.0
        with partial.open("xb") as output:
            for chunk in stream_canonical_bundle(bundle, path_resolver=resolve):
                output.write(chunk)
                if time.monotonic() - last_check > 0.25:
                    self._check(job_id)
                    self._update(job_id, bytes_done=output.tell())
                    last_check = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        archive = self._file(job_id, export_id + ".zip")
        partial.replace(archive)
        self._update(
            job_id,
            bytes_done=archive.stat().st_size,
            bytes_total=archive.stat().st_size,
        )
        return archive, metadata

    def open_download(self, export_id: str) -> tuple[Path, BinaryIO] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(artifact_job_id,job_id) AS job_id,available FROM training_export_parts WHERE export_id=?",
                (export_id,),
            ).fetchone()
        if row is None:
            return None
        if not row["available"]:
            raise FileNotFoundError("The prepared ZIP has expired or was removed")
        path = self._file(row["job_id"], export_id + ".zip")
        try:
            handle = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
        except FileNotFoundError:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE training_export_parts SET available=0 WHERE export_id=?",
                    (export_id,),
                )
            raise
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            handle.close()
            raise ValueError("Export artifact is not a regular file")
        fcntl.flock(handle, fcntl.LOCK_SH)
        return path, handle

    def _remove(self, job_id: str, name: str) -> bool:
        path = self._file(job_id, name)
        try:
            descriptor = os.open(
                path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                handle = os.open(
                    path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                try:
                    if not stat.S_ISREG(os.fstat(handle).st_mode):
                        raise ValueError("Export artifact is not a regular file")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return False
                    os.unlink(path.name, dir_fd=descriptor)
                finally:
                    os.close(handle)
            finally:
                os.close(descriptor)
        except FileNotFoundError:
            pass
        return True

    def remove_archive(self, export_id: str) -> None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(p.artifact_job_id,p.job_id) AS job_id,e.received_at FROM training_export_parts p JOIN labeling_exports e USING(export_id) WHERE export_id=?",
                (export_id,),
            ).fetchone()
            if row is None or row["received_at"] is None:
                raise ValueError("Confirm receipt before removing a local ZIP")
            if not self._remove(row["job_id"], export_id + ".zip"):
                raise ValueError("This ZIP is currently being downloaded")
            conn.execute(
                "UPDATE training_export_parts SET available=0 WHERE export_id=?",
                (export_id,),
            )

    def cleanup(self) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        with self.connection() as conn:
            parts = conn.execute(
                "SELECT * FROM training_export_parts WHERE available=1",
            )
            for part in parts:
                job_id = part["artifact_job_id"] or part["job_id"]
                name = part["export_id"] + ".zip"
                path = self._file(job_id, name)
                if not path.is_file() or (
                    part["created_at"] < cutoff and self._remove(job_id, name)
                ):
                    conn.execute(
                        "UPDATE training_export_parts SET available=0 WHERE export_id=?",
                        (part["export_id"],),
                    )
            jobs = conn.execute(
                "SELECT job_id FROM training_export_jobs WHERE status IN ('failed','cancelled','complete','paused') AND updated_at<?",
                (cutoff,),
            ).fetchall()
            for job in jobs:
                for name in (
                    "archive.partial",
                    "progress.partial",
                    "progress.json",
                    "snapshot.db",
                    "snapshot.db-wal",
                    "snapshot.db-shm",
                ):
                    self._remove(job["job_id"], name)
                conn.execute(
                    "UPDATE training_export_jobs SET snapshot_ready=0,status='expired' WHERE job_id=? AND status!='complete'",
                    (job["job_id"],),
                )
