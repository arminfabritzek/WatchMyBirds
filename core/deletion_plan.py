"""Read-only, ordered photo retention decisions for preview and execution."""

from __future__ import annotations

import sqlite3
import stat
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from core.storage_forecast import StorageForecast
from core.training_export import answer_revisions, list_exports
from core.training_retention import protection_sources_any
from utils.db.storage import image_batches
from utils.path_manager import PathManager


@dataclass(frozen=True)
class DeletionSettings:
    old_bird_photos: Literal["keep_previews", "delete_all"] = "keep_previews"
    original_days: int = 30
    preview_days: int = 180
    reserve_bytes: int = 256 * 1024**2
    trash_days: int = 30
    other_days: int = 14

    def __post_init__(self) -> None:
        if self.old_bird_photos not in {"keep_previews", "delete_all"}:
            raise ValueError("Unknown old bird photos option")
        if (
            min(
                self.original_days,
                self.preview_days,
                self.reserve_bytes,
                self.trash_days,
                self.other_days,
            )
            < 0
            or self.preview_days < self.original_days
        ):
            raise ValueError("Invalid deletion deadlines or reserve")


DEFAULT_SETTINGS = DeletionSettings()


@dataclass(frozen=True)
class Action:
    action: Literal["keep", "delete"]
    rule: int
    reason: str
    deadline: datetime | None = None

    def due(self, now: datetime) -> bool:
        return (
            self.action == "delete"
            and self.deadline is not None
            and self.deadline <= now
        )


@dataclass(frozen=True)
class FilePlan:
    reference: str
    category: str
    present: bool
    size_bytes: int
    decision: Action
    error: str | None = None


@dataclass(frozen=True)
class RowPlan:
    table: str
    key: int | str
    decision: Action


@dataclass(frozen=True)
class PhotoPlan:
    filename: str
    objects: dict[str, Action]
    files: list[FilePlan]
    rows: list[RowPlan]
    training_sources: tuple[str, ...]
    captured_at: datetime | None = None


def _time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.strptime(str(value)[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        try:
            result = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    return (
        result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    )


def _expire(rule: int, reason: str, base: datetime | None, days: int) -> Action:
    if base is None:
        return Action("keep", rule, reason + ": date unavailable")
    return Action("delete", rule, reason, base + timedelta(days=days))


def _strongest(actions: list[Action]) -> Action:
    kept = [item for item in actions if item.action == "keep"]
    if kept:
        return min(kept, key=lambda item: item.rule)
    return max(actions, key=lambda item: (item.deadline, -item.rule))


def _rule(obj: dict[str, Any], sources: set[str], answered: bool) -> tuple[int, str]:
    if obj.get("status") == "rejected":
        return 1, "Trash"
    if obj.get("is_favorite"):
        return 2, "Favorite"
    if sources:
        return 3, ", ".join(sorted(sources))
    if answered:
        return 4, "Answered"
    if obj.get("admitted"):
        return 5, "Admitted bird"
    return 6, "Open case or model non-bird"


def _actions(
    obj: dict[str, Any],
    sources: set[str],
    answered: bool,
    received: datetime | None,
    captured: datetime | None,
    settings: DeletionSettings,
    *,
    automatic_deletion_enabled_at: datetime,
    original_missing: bool = False,
    answered_at: datetime | None = None,
    pressure_stage: int = 0,
    now: datetime | None = None,
) -> tuple[Action, Action, Action]:
    rule, reason = _rule(obj, sources, answered)
    keep = Action("keep", rule, reason)
    if rule in {2, 3}:
        return keep, keep, keep
    if pressure_stage and rule != 1:
        early = Action("delete", rule, reason + ": low free space", now)
        if rule == 6:
            return early, early, early
        return (
            early if pressure_stage >= 2 else keep,
            early
            if pressure_stage >= 3 and settings.old_bird_photos == "delete_all"
            else keep,
            keep,
        )
    if rule in {1, 6}:
        base = _time(obj.get("trashed_at")) if rule == 1 else captured
        if rule == 6 and base is not None:
            base = max(base, automatic_deletion_enabled_at)
        action = _expire(
            rule,
            reason,
            base,
            settings.trash_days if rule == 1 else settings.other_days,
        )
        return action, action, action
    base = received if rule == 4 else captured
    if rule == 4 and base is None and original_missing:
        base = answered_at
    if rule == 4 and obj.get("presence") == "absent":
        action = _expire(rule, "Answered no bird", base, settings.other_days)
        return action, action, keep
    original = _expire(rule, reason, base, settings.original_days)
    derivative = (
        keep
        if settings.old_bird_photos == "keep_previews"
        else _expire(rule, reason, base, settings.preview_days)
    )
    return original, derivative, keep


def _file(
    paths: PathManager, candidate: Path, category: str, decision: Action
) -> FilePlan:
    reference = candidate.relative_to(paths.base_dir).as_posix()
    root = paths.originals_dir if category == "original" else paths.derivatives_dir
    checked = paths.contained_path(candidate, root)
    if (
        checked is None
        or checked == root.resolve()
        or not checked.is_relative_to(paths.base_dir.resolve())
        or checked != paths.base_dir.resolve() / reference
    ):
        return FilePlan(
            reference,
            category,
            False,
            0,
            replace(decision, action="keep", deadline=None, reason="Unsafe media path"),
            "unsafe_path",
        )
    try:
        info = checked.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Not a regular media file")
        return FilePlan(reference, category, True, info.st_size, decision)
    except FileNotFoundError:
        return FilePlan(reference, category, False, 0, decision)
    except (OSError, ValueError):
        return FilePlan(
            reference,
            category,
            False,
            0,
            replace(decision, action="keep", deadline=None, reason="Media unavailable"),
            "unavailable",
        )


def _dependent_rows(conn: sqlite3.Connection, rows: list[RowPlan]) -> list[RowPlan]:
    dependencies = {
        "images": (
            ("rescan_proposals", "proposal_id", "image_filename"),
            ("bird_add_requests", "request_id", "image_filename"),
        ),
        "manual_objects": (
            ("manual_object_requests", "request_id", "manual_object_id"),
        ),
        "detections": (
            ("station_event_reviews", "review_id", "anchor_detection_id"),
            ("station_event_review_members", "rowid", "detection_id"),
        ),
        "station_event_reviews": (
            ("station_event_review_members", "rowid", "review_id"),
        ),
    }
    children: dict[tuple[str, int | str], RowPlan] = {}
    pending = list(rows)
    while pending:
        parent = pending.pop()
        for table, key, foreign in dependencies.get(parent.table, ()):
            for child in conn.execute(
                f"SELECT {key} FROM {table} WHERE {foreign}=?", (parent.key,)
            ):
                identifier = (table, child[0])
                row = RowPlan(table, child[0], parent.decision)
                previous = children.get(identifier)
                # A membership is removed when either of its parents expires.
                if previous and (
                    not row.decision.deadline
                    or previous.decision.deadline
                    and previous.decision.deadline <= row.decision.deadline
                ):
                    continue
                children[identifier] = row
                pending.append(row)
    return list(children.values())


def _batch_crop_names(paths: PathManager, names: list[str]) -> dict[str, set[str]]:
    """Discover unreferenced crops once per directory for this bounded batch.

    Keep the inventory local to a planner batch: executor rechecks still discover
    files anew under each photo's write lock. Only selected stems are retained.
    Individual paths are checked again by _file before entering a deletion plan.
    """
    directories: dict[Path, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    found: dict[str, set[str]] = defaultdict(set)
    for name in names:
        if Path(name).name != name:
            continue
        directory = paths.get_derivative_path(name).parent
        if paths.contained_path(directory, paths.base_dir) != (
            paths.base_dir.resolve() / directory.relative_to(paths.base_dir)
        ):
            continue
        directories[directory][Path(name).stem].append(name)
    for directory, stems in directories.items():
        for path in directory.glob("*_crop_*.webp"):
            # Check each separator, preserving stems that themselves contain
            # '_crop_' and glob metacharacters without interpreting them.
            marker = path.name.find("_crop_")
            while marker >= 0:
                for name in stems.get(path.name[:marker], ()):
                    found[name].add(path.name)
                marker = path.name.find("_crop_", marker + 1)
    return found


def plan_deletions(
    conn: sqlite3.Connection,
    settings: DeletionSettings,
    now: datetime,
    *,
    paths: PathManager,
    batch_size: int = 200,
    filenames: list[str] | None = None,
    automatic_deletion_enabled_at: datetime | None = None,
    pressure_stage: int = 0,
) -> Iterator[PhotoPlan]:
    """Yield complete photos in bounded batches, without SQL or filesystem writes.

    Delete actions include future deadlines; callers use ``decision.due(now)``.
    Missing files retain their proposed action so row cleanup remains possible.
    Export records are independent immutable evidence and are never targeted.
    Without an activation date, preview as if deletion were enabled now.
    """
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    if pressure_stage not in {0, 1, 2, 3}:
        raise ValueError("Unknown pressure stage")
    enabled_at = automatic_deletion_enabled_at or now
    if enabled_at.tzinfo is None:
        raise ValueError("automatic_deletion_enabled_at must include a timezone")
    for batch in image_batches(conn, batch_size, filenames=filenames):
        names = [str(row["filename"]) for row in batch]
        marks = ",".join("?" for _ in names)
        originals = {
            name: _file(
                paths,
                paths.get_original_path(name),
                "original",
                Action("keep", 6, "Unassessed"),
            )
            for name in names
            if Path(name).name == name
        }
        sources = protection_sources_any(
            conn,
            names,
            media_exists=lambda name, files=originals: (
                name in files and files[name].present
            ),
        )
        revisions = answer_revisions(conn, names)
        receipts: dict[str, datetime | None] = {}
        for revision in revisions:
            row = conn.execute(
                """SELECT MAX(e.received_at) AS received_at FROM labeling_export_members m
                JOIN labeling_exports e USING(export_id)
                WHERE m.kind=? AND m.object_id=? AND m.revision_json=?
                AND e.received_at IS NOT NULL""",
                (revision["kind"], revision["object_id"], revision["revision_json"]),
            ).fetchone()
            receipts[str(revision["selection_key"])] = _time(row["received_at"])
        objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in conn.execute(
            f"""SELECT o.object_key, o.detection_id, o.manual_object_id,
            o.image_filename, o.status, o.is_favorite, o.admitted,
            o.presence, o.answered_at, o.thumbnail_path, d.trashed_at
            FROM current_observations o
            LEFT JOIN detections d ON d.detection_id=o.detection_id
            WHERE o.image_filename IN ({marks}) ORDER BY o.object_key""",
            names,
        ):
            objects[row["image_filename"]].append(dict(row))
        subjects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in conn.execute(
            f"SELECT * FROM label_subjects WHERE image_filename IN ({marks})", names
        ):
            subjects[row["image_filename"]].append(dict(row))
        current = {
            row["subject_id"]: dict(row)
            for row in conn.execute(
                f"""SELECT subject_id, MAX(created_at) AS answered_at,
                MAX(CASE WHEN fact_type='bird_presence' THEN answer_value END) AS presence
                FROM current_human_label_facts WHERE image_filename IN ({marks}) GROUP BY subject_id""",
                names,
            )
        }
        crop_names = _batch_crop_names(paths, names)
        for image in batch:
            name = image["filename"]
            captured = _time(image["timestamp"])
            original_file = originals.get(name)
            original_missing = (
                original_file is not None
                and not original_file.present
                and original_file.error is None
            )
            decisions: dict[str, tuple[Action, Action, Action]] = {}
            object_subjects = {
                s["detection_id"]: s
                for s in subjects[name]
                if s["detection_id"] is not None
            }
            for obj in objects[name]:
                key = obj["object_key"]
                subject = object_subjects.get(obj["detection_id"])
                fact = current.get(subject["subject_id"], {}) if subject else {}
                answered = (
                    obj["manual_object_id"] is not None
                    and obj["status"] == "active"
                    or subject is not None
                    and subject["subject_id"] in current
                )
                if obj["manual_object_id"] is not None and obj["status"] != "active":
                    answered = False
                decisions[key] = _actions(
                    obj,
                    sources.get(name, set()),
                    answered,
                    receipts.get(key),
                    captured,
                    settings,
                    automatic_deletion_enabled_at=enabled_at,
                    original_missing=original_missing,
                    answered_at=_time(
                        fact.get("answered_at") or obj.get("answered_at")
                    ),
                    pressure_stage=pressure_stage,
                    now=now,
                )
            for subject in subjects[name]:
                if subject["detection_id"] is None:
                    key = f"image:{subject['subject_id']}"
                    fact = current.get(subject["subject_id"], {})
                    decisions[key] = _actions(
                        fact,
                        sources.get(name, set()),
                        bool(fact),
                        receipts.get(key),
                        captured,
                        settings,
                        automatic_deletion_enabled_at=enabled_at,
                        original_missing=original_missing,
                        answered_at=_time(fact.get("answered_at")),
                        pressure_stage=pressure_stage,
                        now=now,
                    )
            if not decisions:
                decisions["image"] = _actions(
                    {},
                    sources.get(name, set()),
                    False,
                    None,
                    captured,
                    settings,
                    automatic_deletion_enabled_at=enabled_at,
                    pressure_stage=pressure_stage,
                    now=now,
                )
            original, derivative, image_row = (
                _strongest([items[index] for items in decisions.values()])
                for index in range(3)
            )
            rows = [RowPlan("images", name, image_row)]
            for obj in objects[name]:
                decision = decisions[obj["object_key"]][2]
                table, identifier = (
                    ("detections", obj["detection_id"])
                    if obj["detection_id"] is not None
                    else ("manual_objects", obj["manual_object_id"])
                )
                rows.append(RowPlan(table, identifier, decision))
                child_table, foreign_key, primary_key = (
                    ("classifications", "detection_id", "classification_id")
                    if table == "detections"
                    else (
                        "manual_object_revisions",
                        "manual_object_id",
                        "manual_revision_id",
                    )
                )
                for child in conn.execute(
                    f"SELECT {primary_key} FROM {child_table} WHERE {foreign_key}=?",
                    (identifier,),
                ):
                    rows.append(RowPlan(child_table, child[0], decision))
            for subject in subjects[name]:
                key = (
                    f"detection:{subject['detection_id']}"
                    if subject["detection_id"] is not None
                    else f"image:{subject['subject_id']}"
                )
                decision = decisions[key][2]
                rows.append(RowPlan("label_subjects", subject["subject_id"], decision))
                for fact in conn.execute(
                    "SELECT fact_id FROM human_label_facts WHERE subject_id=?",
                    (subject["subject_id"],),
                ):
                    rows.append(RowPlan("human_label_facts", fact[0], decision))
            rows.extend(_dependent_rows(conn, rows))
            if Path(name).name != name:
                rows = [
                    replace(
                        row,
                        decision=Action("keep", row.decision.rule, "Unsafe media path"),
                    )
                    for row in rows
                ]
                yield PhotoPlan(
                    name,
                    {key: item[0] for key, item in decisions.items()},
                    [],
                    rows,
                    tuple(sorted(sources.get(name, set()))),
                    captured,
                )
                continue
            files = [
                _file(paths, paths.get_original_path(name), "original", original),
                _file(
                    paths,
                    paths.get_derivative_path(name, "optimized"),
                    "optimized",
                    derivative,
                ),
                _file(paths, paths.get_derivative_path(name), "thumbnail", derivative),
                _file(paths, paths.get_preview_thumb_path(name), "preview", derivative),
            ]
            thumb_names = {
                obj["thumbnail_path"] or f"{Path(name).stem}_crop_1.webp"
                for obj in objects[name]
                if obj["detection_id"] is not None
            }
            thumb_names.update(crop_names.get(name, ()))
            for thumb in sorted(thumb_names):
                action = derivative
                if Path(thumb).name != thumb:
                    files.append(
                        FilePlan(
                            thumb,
                            "thumbnail",
                            False,
                            0,
                            Action("keep", derivative.rule, "Unsafe media path"),
                            "unsafe_path",
                        )
                    )
                    continue
                if conn.execute(
                    "SELECT 1 FROM detections WHERE thumbnail_path=? AND image_filename!=? LIMIT 1",
                    (thumb, name),
                ).fetchone():
                    action = Action(
                        "keep", derivative.rule, "Shared thumbnail reference"
                    )
                files.append(
                    _file(paths, paths.get_derivative_path(thumb), "thumbnail", action)
                )
            files = list({item.reference: item for item in files}.values())
            yield PhotoPlan(
                name,
                {key: item[0] for key, item in decisions.items()},
                files,
                rows,
                tuple(sorted(sources.get(name, set()))),
                captured,
            )


def deletion_preview(
    conn: sqlite3.Connection,
    settings: DeletionSettings,
    now: datetime,
    *,
    paths: PathManager,
    automatic_deletion_enabled_at: datetime | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    classes: dict[str, dict[str, int]] = defaultdict(lambda: dict(files=0, bytes=0))
    waiting: dict[str, dict[str, int]] = defaultdict(lambda: dict(photos=0, bytes=0))
    upcoming: list[dict[str, Any]] = []
    due_files = due_bytes = due_rows = errors = unknown_deadlines = 0
    fortnight_files = fortnight_bytes = fortnight_photos = 0
    horizon = now + timedelta(days=14)
    forecast = StorageForecast(now, enabled=automatic_deletion_enabled_at is not None)
    processed = 0
    if progress_callback:
        progress_callback(dict(images_done=processed))
    for photo in plan_deletions(
        conn,
        settings,
        now,
        paths=paths,
        automatic_deletion_enabled_at=automatic_deletion_enabled_at,
    ):
        forecast.add(photo)
        present_bytes = sum(item.size_bytes for item in photo.files)
        for source in photo.training_sources:
            waiting[source]["photos"] += 1
            waiting[source]["bytes"] += present_bytes
        unknown_deadlines += sum(
            ": date unavailable" in item.decision.reason
            for item in photo.files
            if item.present
        )
        due_rows += sum(row.decision.due(now) for row in photo.rows)
        upcoming_files = [
            item for item in photo.files if item.present and item.decision.due(horizon)
        ]
        fortnight_photos += bool(upcoming_files) or any(
            row.decision.due(horizon) for row in photo.rows
        )
        fortnight_files += len(upcoming_files)
        fortnight_bytes += sum(item.size_bytes for item in upcoming_files)
        for item in photo.files:
            errors += bool(item.error)
            if not item.present:
                continue
            key = str(item.decision.rule)
            classes[key]["files"] += 1
            classes[key]["bytes"] += item.size_bytes
            if item.decision.due(now):
                due_files += 1
                due_bytes += item.size_bytes
            if item.decision.deadline is not None:
                upcoming.append(
                    dict(
                        filename=photo.filename,
                        reference=item.reference,
                        category=item.category,
                        rule=item.decision.rule,
                        deadline=item.decision.deadline.isoformat(),
                        bytes=item.size_bytes,
                        due=item.decision.due(now),
                    )
                )
                upcoming.sort(key=lambda entry: (entry["deadline"], entry["reference"]))
                del upcoming[20:]
        processed += 1
        if progress_callback and processed % 200 == 0:
            progress_callback(dict(images_done=processed))
    if progress_callback and processed % 200:
        progress_callback(dict(images_done=processed))
    return dict(
        settings=asdict(settings),
        classes=dict(classes),
        waiting=dict(waiting),
        unconfirmed_exports=[
            row for row in list_exports(conn) if row["received_at"] is None
        ],
        upcoming=upcoming,
        due_files=due_files,
        due_bytes=due_bytes,
        due_rows=due_rows,
        errors=errors,
        unknown_deadlines=unknown_deadlines,
        next_14_days=dict(
            photos=fortnight_photos, files=fortnight_files, bytes=fortnight_bytes
        ),
        forecast_inputs=forecast.inputs(),
    )
