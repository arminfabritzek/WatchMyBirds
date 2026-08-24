"""Scientific evidence contract for the single-station report.

Model proposals and automated decision states are candidates.  A biological
event enters the verified cohort only when an explicit event review marks its
evidence diagnostic and the canonical human-label facts independently confirm
bird presence and the same species for every event member.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from core.events import BirdEvent
from core.human_label_core import LabelProvenance

EVENT_REVIEW_OUTCOMES = frozenset(
    {"verified_species", "unresolved_bird", "no_bird"}
)
EVIDENCE_QUALITIES = frozenset({"diagnostic", "limited", "insufficient"})


class StationEventReviewError(ValueError):
    """Raised when an event assessment violates the evidence contract."""


@dataclass(frozen=True)
class StationEventReview:
    review_id: int
    event_key_snapshot: str
    detection_ids: tuple[int, ...]
    outcome: str
    candidate_species_key: str | None
    species_key: str | None
    evidence_quality: str
    event_start: str
    event_end: str
    created_at: str

    @property
    def is_diagnostic_species_record(self) -> bool:
        return (
            self.outcome == "verified_species"
            and self.evidence_quality == "diagnostic"
            and bool(self.species_key)
        )


@dataclass(frozen=True)
class ObservationEffort:
    available: bool
    sample_count: int
    coverage_start: str | None
    coverage_end: str | None
    window_hours: float
    app_hours: float
    camera_hours: float
    stream_hours: float
    detector_hours: float
    observation_hours: float
    day_observation_hours: float
    night_observation_hours: float
    unknown_light_hours: float
    ptz_active_hours: float
    outage_hours: float
    weather_exposure_hours: dict[int, float]


def _parse_iso_utc(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def calculate_observation_effort(conn: sqlite3.Connection) -> ObservationEffort:
    """Integrate bounded runtime samples into honest camera-hour effort."""
    try:
        rows = conn.execute(
            """
            SELECT sampled_at, sample_interval_seconds, app_online,
                   camera_online, stream_online, detector_ready,
                   detector_active, od_active, od_reason, ptz_state
            FROM station_runtime_samples
            ORDER BY sampled_at ASC, sample_id ASC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    empty = ObservationEffort(
        available=False,
        sample_count=0,
        coverage_start=None,
        coverage_end=None,
        window_hours=0.0,
        app_hours=0.0,
        camera_hours=0.0,
        stream_hours=0.0,
        detector_hours=0.0,
        observation_hours=0.0,
        day_observation_hours=0.0,
        night_observation_hours=0.0,
        unknown_light_hours=0.0,
        ptz_active_hours=0.0,
        outage_hours=0.0,
        weather_exposure_hours={},
    )
    if len(rows) < 2:
        return empty

    weather_rows = conn.execute(
        """
        SELECT timestamp, condition_code, is_day
        FROM weather_logs
        WHERE timestamp IS NOT NULL
        ORDER BY timestamp ASC
        """
    ).fetchall()
    weather_points = [
        (_parse_iso_utc(row["timestamp"]), row["condition_code"], row["is_day"])
        for row in weather_rows
    ]
    weather_points = [point for point in weather_points if point[0] is not None]

    parsed_samples = [(_parse_iso_utc(row["sampled_at"]), row) for row in rows]
    parsed_samples = [item for item in parsed_samples if item[0] is not None]
    if len(parsed_samples) < 2:
        return empty

    seconds = {
        "app": 0.0,
        "camera": 0.0,
        "stream": 0.0,
        "detector": 0.0,
        "observation": 0.0,
        "day": 0.0,
        "night": 0.0,
        "unknown_light": 0.0,
        "ptz_active": 0.0,
    }
    weather_exposure: dict[int, float] = {}

    for index, (sample_at, row) in enumerate(parsed_samples[:-1]):
        next_at = parsed_samples[index + 1][0]
        interval = max(1.0, float(row["sample_interval_seconds"] or 60.0))
        elapsed = max(0.0, (next_at - sample_at).total_seconds())
        duration = min(elapsed, interval * 1.5)
        if duration <= 0:
            continue
        if row["app_online"]:
            seconds["app"] += duration
        if row["camera_online"]:
            seconds["camera"] += duration
        if row["stream_online"]:
            seconds["stream"] += duration
        if row["detector_active"]:
            seconds["detector"] += duration
        if str(row["ptz_state"] or "") in {
            "settling",
            "acquiring",
            "tracking",
            "lost_grace",
            "returning",
        }:
            seconds["ptz_active"] += duration

        observation_active = bool(
            row["app_online"]
            and row["camera_online"]
            and row["stream_online"]
            and row["detector_ready"]
            and row["detector_active"]
            and row["od_active"]
        )
        if not observation_active:
            continue
        seconds["observation"] += duration

        nearest = None
        if weather_points:
            nearest = min(
                weather_points,
                key=lambda point: abs((point[0] - sample_at).total_seconds()),
            )
            if abs((nearest[0] - sample_at).total_seconds()) > 45 * 60:
                nearest = None
        if nearest is None:
            seconds["unknown_light"] += duration
            continue

        _, condition_code, is_day = nearest
        if is_day == 1:
            seconds["day"] += duration
        elif is_day == 0:
            seconds["night"] += duration
        else:
            seconds["unknown_light"] += duration
        if condition_code is not None:
            code = int(condition_code)
            weather_exposure[code] = weather_exposure.get(code, 0.0) + duration

    first_at = parsed_samples[0][0]
    last_at = parsed_samples[-1][0]
    window_seconds = max(0.0, (last_at - first_at).total_seconds())

    def hours(value: float) -> float:
        return round(value / 3600.0, 2)

    return ObservationEffort(
        available=seconds["observation"] > 0,
        sample_count=len(parsed_samples),
        coverage_start=first_at.isoformat(),
        coverage_end=last_at.isoformat(),
        window_hours=hours(window_seconds),
        app_hours=hours(seconds["app"]),
        camera_hours=hours(seconds["camera"]),
        stream_hours=hours(seconds["stream"]),
        detector_hours=hours(seconds["detector"]),
        observation_hours=hours(seconds["observation"]),
        day_observation_hours=hours(seconds["day"]),
        night_observation_hours=hours(seconds["night"]),
        unknown_light_hours=hours(seconds["unknown_light"]),
        ptz_active_hours=hours(seconds["ptz_active"]),
        outage_hours=hours(max(0.0, window_seconds - seconds["app"])),
        weather_exposure_hours={
            code: hours(value) for code, value in weather_exposure.items()
        },
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (p + (z * z / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z / (4 * total * total)))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _isotonic_blocks(pairs: list[tuple[float, int]]) -> list[dict[str, float | int]]:
    """Pool-adjacent-violators calibration without an external dependency."""
    blocks: list[dict[str, float | int]] = []
    for confidence, correct in sorted(pairs):
        blocks.append(
            {
                "min_score": confidence,
                "max_score": confidence,
                "correct": int(correct),
                "n": 1,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_rate = int(left["correct"]) / int(left["n"])
            right_rate = int(right["correct"]) / int(right["n"])
            if left_rate <= right_rate:
                break
            merged = {
                "min_score": float(left["min_score"]),
                "max_score": float(right["max_score"]),
                "correct": int(left["correct"]) + int(right["correct"]),
                "n": int(left["n"]) + int(right["n"]),
            }
            blocks[-2:] = [merged]
    for block in blocks:
        block["calibrated_precision"] = round(
            int(block["correct"]) / int(block["n"]), 3
        )
    return blocks


def build_model_validation(conn: sqlite3.Connection) -> dict[str, object]:
    """Evaluate model proposals only against canonical current human facts."""
    try:
        rows = conn.execute(
            """
            SELECT
                s.detection_id,
                s.proposal_species_key AS predicted_species,
                i.timestamp,
                d.bbox_w, d.bbox_h, d.quality_gallery_ok,
                d.sharpness_score, d.crop_brightness,
                (
                    SELECT c.cls_confidence
                    FROM classifications c
                    WHERE c.detection_id = s.detection_id
                    ORDER BY c.rank ASC, c.classification_id ASC
                    LIMIT 1
                ) AS model_confidence,
                bird.answer_value AS bird_presence,
                species.answer_value AS species_answer,
                species.species_key AS verified_species
            FROM label_subjects s
            JOIN detections d ON d.detection_id = s.detection_id
            JOIN images i ON i.filename = s.image_filename
            LEFT JOIN current_human_label_facts bird
              ON bird.subject_id = s.subject_id
             AND bird.fact_type = 'bird_presence'
            LEFT JOIN current_human_label_facts species
              ON species.subject_id = s.subject_id
             AND species.fact_type = 'species_identity'
            WHERE s.scope = 'object'
              AND (
                  bird.answer_value IN ('present', 'absent')
                  OR species.answer_value IN ('confirmed', 'corrected', 'wrong', 'unknown')
              )
            ORDER BY i.timestamp ASC, s.detection_id ASC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    records = []
    for row in rows:
        predicted = str(row["predicted_species"] or "").strip() or None
        actual = str(row["verified_species"] or "").strip() or None
        is_resolved_answer = (
            row["bird_presence"] == "present"
            and row["species_answer"] in {"confirmed", "corrected"}
            and actual is not None
        )
        is_negative = row["bird_presence"] == "absent"
        if not (is_resolved_answer or is_negative) or predicted is None:
            continue
        confidence = row["model_confidence"]
        if confidence is None:
            continue
        correct = int(is_resolved_answer and predicted == actual)
        bbox_area = float(row["bbox_w"] or 0.0) * float(row["bbox_h"] or 0.0)
        records.append(
            {
                "detection_id": int(row["detection_id"]),
                "timestamp": str(row["timestamp"] or ""),
                "predicted_species": predicted,
                "actual_species": actual,
                "confidence": float(confidence),
                "correct": correct,
                "tiny": bbox_area < 0.01,
                "quality_floored": row["quality_gallery_ok"] == 0,
                "night_like": (
                    row["crop_brightness"] is not None
                    and float(row["crop_brightness"]) < 45.0
                ),
            }
        )

    total = len(records)
    correct_total = sum(int(record["correct"]) for record in records)
    by_species: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_species.setdefault(str(record["predicted_species"]), []).append(record)
    species_rows = []
    for species, members in by_species.items():
        n = len(members)
        correct = sum(int(member["correct"]) for member in members)
        low, high = _wilson_interval(correct, n)
        species_rows.append(
            {
                "species": species,
                "reviewed": n,
                "correct": correct,
                "precision": round(correct / n, 3),
                "precision_low": round(low, 3),
                "precision_high": round(high, 3),
                "sufficient": n >= 20,
            }
        )
    species_rows.sort(key=lambda item: (-int(item["reviewed"]), str(item["species"])))

    confidence_bins = []
    for lower_int in range(0, 10):
        lower = lower_int / 10.0
        upper = (lower_int + 1) / 10.0
        members = [
            record
            for record in records
            if lower <= float(record["confidence"]) <= upper
            and (lower_int == 9 or float(record["confidence"]) < upper)
        ]
        if not members:
            continue
        confidence_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "n": len(members),
                "mean_confidence": round(
                    sum(float(member["confidence"]) for member in members)
                    / len(members),
                    3,
                ),
                "empirical_precision": round(
                    sum(int(member["correct"]) for member in members) / len(members),
                    3,
                ),
            }
        )

    dates = sorted({str(record["timestamp"])[:8] for record in records if record["timestamp"]})
    holdout = {"available": False, "reviewed": 0, "accuracy": None, "start": None}
    if total >= 30 and len(dates) >= 5:
        split_index = max(1, math.floor(len(dates) * 0.8))
        holdout_start = dates[min(split_index, len(dates) - 1)]
        holdout_rows = [
            record for record in records if str(record["timestamp"])[:8] >= holdout_start
        ]
        if holdout_rows:
            holdout = {
                "available": True,
                "reviewed": len(holdout_rows),
                "accuracy": round(
                    sum(int(record["correct"]) for record in holdout_rows)
                    / len(holdout_rows),
                    3,
                ),
                "start": holdout_start,
            }

    strata = []
    for key, label in (
        ("tiny", "Tiny boxes"),
        ("quality_floored", "Sharpness-floored crops"),
        ("night_like", "Dark crops"),
    ):
        members = [record for record in records if record[key]]
        strata.append(
            {
                "key": key,
                "label": label,
                "reviewed": len(members),
                "precision": (
                    round(
                        sum(int(member["correct"]) for member in members)
                        / len(members),
                        3,
                    )
                    if members
                    else None
                ),
            }
        )

    return {
        "available": total > 0,
        "reviewed_predictions": total,
        "correct_predictions": correct_total,
        "overall_precision": round(correct_total / total, 3) if total else None,
        "species": species_rows,
        "confidence_bins": confidence_bins,
        "calibration_available": total >= 30,
        "calibration_method": "isotonic" if total >= 30 else None,
        "calibration_blocks": (
            _isotonic_blocks(
                [(float(record["confidence"]), int(record["correct"])) for record in records]
            )
            if total >= 30
            else []
        ),
        "holdout": holdout,
        "quality_strata": strata,
    }


def build_candidate_review_priorities(
    conn: sqlite3.Connection,
    events: Iterable[BirdEvent],
) -> list[dict[str, object]]:
    """Rank candidate events by information value for human review."""
    event_list = list(events)
    detection_ids = {
        detection_id for event in event_list for detection_id in event.detection_ids
    }
    details: dict[int, sqlite3.Row] = {}
    if detection_ids:
        placeholders = ",".join("?" for _ in detection_ids)
        try:
            rows = conn.execute(
                f"""
                SELECT d.detection_id, d.bbox_w, d.bbox_h,
                       d.quality_gallery_ok, d.crop_brightness,
                       (
                           SELECT c.cls_confidence
                           FROM classifications c
                           WHERE c.detection_id = d.detection_id
                           ORDER BY c.rank ASC, c.classification_id ASC
                           LIMIT 1
                       ) AS cls_confidence
                FROM detections d
                WHERE d.detection_id IN ({placeholders})
                """,
                tuple(sorted(detection_ids)),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        details = {int(row["detection_id"]): row for row in rows}

    species_frequency = Counter(event.species or "unresolved" for event in event_list)
    priorities = []
    for event in event_list:
        members = [details.get(detection_id) for detection_id in event.detection_ids]
        members = [member for member in members if member is not None]
        confidences = [
            float(member["cls_confidence"])
            for member in members
            if member["cls_confidence"] is not None
        ]
        flags = []
        if not event.species:
            flags.append("unresolved")
        if species_frequency[event.species or "unresolved"] <= 3:
            flags.append("rare")
        if any(
            float(member["bbox_w"] or 0.0) * float(member["bbox_h"] or 0.0) < 0.01
            for member in members
        ):
            flags.append("tiny")
        if any(member["quality_gallery_ok"] == 0 for member in members):
            flags.append("blurred_or_low_quality")
        if any(
            member["crop_brightness"] is not None
            and float(member["crop_brightness"]) < 45.0
            for member in members
        ):
            flags.append("dark")
        if not confidences or min(confidences) < 0.75:
            flags.append("uncertain")
        weights = {
            "unresolved": 5,
            "rare": 4,
            "tiny": 3,
            "blurred_or_low_quality": 3,
            "dark": 2,
            "uncertain": 2,
        }
        priorities.append(
            {
                "event_key": event.event_key,
                "species": event.species,
                "detection_ids": list(event.detection_ids),
                "start_time": event.start_time,
                "photo_count": event.photo_count,
                "flags": flags,
                "priority_score": sum(weights[flag] for flag in flags),
                "model_confidence": round(max(confidences), 3) if confidences else None,
            }
        )
    priorities.sort(
        key=lambda item: (
            -int(item["priority_score"]),
            str(item["start_time"]),
            str(item["event_key"]),
        )
    )
    return priorities


def record_station_event_review(
    conn: sqlite3.Connection,
    *,
    event_key: str,
    detection_ids: Iterable[int],
    anchor_detection_id: int,
    outcome: str,
    evidence_quality: str,
    event_start: str,
    event_end: str,
    provenance: LabelProvenance,
    candidate_species_key: str | None = None,
    species_key: str | None = None,
) -> int:
    """Append one immutable event assessment without committing."""
    member_ids = tuple(dict.fromkeys(int(value) for value in detection_ids))
    if not member_ids or any(value <= 0 for value in member_ids):
        raise StationEventReviewError("event review requires detection ids")
    if int(anchor_detection_id) not in member_ids:
        raise StationEventReviewError("anchor detection must belong to the event")
    if outcome not in EVENT_REVIEW_OUTCOMES:
        raise StationEventReviewError("invalid event review outcome")
    if evidence_quality not in EVIDENCE_QUALITIES:
        raise StationEventReviewError("invalid evidence quality")

    clean_species = str(species_key or "").strip() or None
    if outcome == "verified_species" and clean_species is None:
        raise StationEventReviewError("verified species outcome requires species")
    if outcome != "verified_species" and clean_species is not None:
        raise StationEventReviewError("non-species outcome cannot carry species")

    placeholders = ",".join("?" for _ in member_ids)
    existing = conn.execute(
        f"SELECT detection_id FROM detections WHERE detection_id IN ({placeholders})",
        member_ids,
    ).fetchall()
    if {int(row[0]) for row in existing} != set(member_ids):
        raise StationEventReviewError("event contains an unknown detection")

    installation_id, app_version, _, source_kind, source_ref, created_at = (
        provenance.values()
    )
    cursor = conn.execute(
        """
        INSERT INTO station_event_reviews (
            event_key_snapshot, anchor_detection_id, outcome,
            candidate_species_key, species_key, evidence_quality,
            event_start, event_end, source_kind, source_ref,
            installation_id, app_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_key.strip(),
            int(anchor_detection_id),
            outcome,
            str(candidate_species_key or "").strip() or None,
            clean_species,
            evidence_quality,
            event_start,
            event_end,
            source_kind,
            source_ref,
            installation_id,
            app_version,
            created_at,
        ),
    )
    review_id = int(cursor.lastrowid)
    conn.executemany(
        """
        INSERT INTO station_event_review_members (review_id, detection_id)
        VALUES (?, ?)
        """,
        [(review_id, detection_id) for detection_id in member_ids],
    )
    return review_id


def fetch_current_station_event_reviews(
    conn: sqlite3.Connection,
) -> dict[frozenset[int], StationEventReview]:
    """Return the newest assessment for every exact event-member snapshot."""
    try:
        rows = conn.execute(
            """
            SELECT r.review_id, r.event_key_snapshot, r.outcome,
                   r.candidate_species_key, r.species_key, r.evidence_quality,
                   r.event_start, r.event_end, r.created_at, m.detection_id
            FROM station_event_reviews r
            JOIN station_event_review_members m ON m.review_id = r.review_id
            ORDER BY r.review_id ASC, m.detection_id ASC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        # Read-only compatibility for old/partial databases. Normal app startup
        # installs the additive tables before serving requests.
        return {}
    grouped: dict[int, dict[str, object]] = {}
    for row in rows:
        bucket = grouped.setdefault(
            int(row["review_id"]),
            {"row": row, "detection_ids": []},
        )
        detection_ids = bucket["detection_ids"]
        assert isinstance(detection_ids, list)
        detection_ids.append(int(row["detection_id"]))

    current: dict[frozenset[int], StationEventReview] = {}
    for review_id, payload in grouped.items():
        row = payload["row"]
        detection_ids = tuple(payload["detection_ids"])
        review = StationEventReview(
            review_id=review_id,
            event_key_snapshot=str(row["event_key_snapshot"]),
            detection_ids=detection_ids,
            outcome=str(row["outcome"]),
            candidate_species_key=row["candidate_species_key"],
            species_key=row["species_key"],
            evidence_quality=str(row["evidence_quality"]),
            event_start=str(row["event_start"]),
            event_end=str(row["event_end"]),
            created_at=str(row["created_at"]),
        )
        current[frozenset(detection_ids)] = review
    return current


def _fetch_verified_object_facts(
    conn: sqlite3.Connection,
    detection_ids: set[int],
) -> dict[int, dict[str, Mapping[str, object]]]:
    if not detection_ids:
        return {}
    placeholders = ",".join("?" for _ in detection_ids)
    try:
        rows = conn.execute(
            f"""
            SELECT detection_id, fact_type, answer_value, species_key
            FROM current_human_label_facts
            WHERE scope = 'object'
              AND detection_id IN ({placeholders})
              AND fact_type IN ('bird_presence', 'species_identity')
            """,
            tuple(sorted(detection_ids)),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    facts: dict[int, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        facts.setdefault(int(row["detection_id"]), {})[str(row["fact_type"])] = row
    return facts


def partition_events_by_evidence(
    conn: sqlite3.Connection,
    events: Iterable[BirdEvent],
) -> tuple[list[BirdEvent], list[BirdEvent], dict[str, int]]:
    """Split candidate events into verified biology and open candidates."""
    event_list = list(events)
    reviews = fetch_current_station_event_reviews(conn)
    all_detection_ids = {
        detection_id
        for event in event_list
        for detection_id in getattr(event, "detection_ids", ())
    }
    facts = _fetch_verified_object_facts(conn, all_detection_ids)

    verified: list[BirdEvent] = []
    candidates: list[BirdEvent] = []
    assessed = 0
    limited = 0
    human_labeled_candidate_events = 0
    human_labeled_candidate_objects = 0
    for event in event_list:
        detection_ids = list(getattr(event, "detection_ids", ()))
        review = reviews.get(frozenset(detection_ids))
        if review is not None:
            assessed += 1
            if review.evidence_quality != "diagnostic":
                limited += 1

        fact_complete = bool(review and review.is_diagnostic_species_record)
        if fact_complete:
            assert review is not None
            for detection_id in detection_ids:
                member_facts = facts.get(detection_id, {})
                bird = member_facts.get("bird_presence")
                species = member_facts.get("species_identity")
                if (
                    bird is None
                    or bird["answer_value"] != "present"
                    or species is None
                    or species["answer_value"] not in {"confirmed", "corrected"}
                    or species["species_key"] != review.species_key
                ):
                    fact_complete = False
                    break

        if fact_complete:
            verified.append(event)
        else:
            candidates.append(event)
            labeled_members = 0
            for detection_id in detection_ids:
                member_facts = facts.get(detection_id, {})
                bird = member_facts.get("bird_presence")
                species = member_facts.get("species_identity")
                if (
                    bird is not None
                    and bird["answer_value"] == "present"
                    and species is not None
                    and species["answer_value"] in {"confirmed", "corrected"}
                    and species["species_key"]
                ):
                    labeled_members += 1
            if labeled_members:
                human_labeled_candidate_events += 1
                human_labeled_candidate_objects += labeled_members

    try:
        review_counts = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(outcome = 'no_bird') AS rejected,
                   SUM(outcome = 'unresolved_bird') AS unresolved,
                   SUM(evidence_quality = 'diagnostic') AS diagnostic,
                   SUM(evidence_quality != 'diagnostic') AS limited
            FROM station_event_reviews
            """
        ).fetchone()
        total_reviews = review_counts["total"]
        rejected_reviews = review_counts["rejected"]
        unresolved_reviews = review_counts["unresolved"]
        diagnostic_reviews = review_counts["diagnostic"]
        limited_reviews = review_counts["limited"]
    except sqlite3.OperationalError:
        total_reviews = 0
        rejected_reviews = 0
        unresolved_reviews = 0
        diagnostic_reviews = 0
        limited_reviews = 0
    current_total = len(verified) + len(candidates)
    return verified, candidates, {
        "candidate_events": len(candidates),
        "verified_events": len(verified),
        "assessed_current_events": assessed,
        "limited_current_events": limited,
        "human_labeled_candidate_events": human_labeled_candidate_events,
        "human_labeled_candidate_objects": human_labeled_candidate_objects,
        "review_actions": int(total_reviews or 0),
        "rejected_review_actions": int(rejected_reviews or 0),
        "unresolved_review_actions": int(unresolved_reviews or 0),
        "diagnostic_review_actions": int(diagnostic_reviews or 0),
        "limited_review_actions": int(limited_reviews or 0),
        "rejection_rate_pct": round(
            100.0 * int(rejected_reviews or 0) / int(total_reviews or 1), 1
        ),
        "review_progress_pct": round(100.0 * assessed / current_total, 1)
        if current_total
        else 0.0,
    }


__all__ = [
    "EVIDENCE_QUALITIES",
    "EVENT_REVIEW_OUTCOMES",
    "ObservationEffort",
    "StationEventReview",
    "StationEventReviewError",
    "calculate_observation_effort",
    "build_candidate_review_priorities",
    "build_model_validation",
    "fetch_current_station_event_reviews",
    "partition_events_by_evidence",
    "record_station_event_review",
]
