"""Append-only utterance recorder for the Companion.

Writes one JSONL line per generated utterance to
``OUTPUT_DIR/companion/utterances_YYYY-MM-DD.jsonl``. Separate from
``images.db`` — Companion never writes to the detection database.

Each line carries enough provenance to support later retraining:
trigger id, timestamp, raw and cleaned text, model id, language,
tone, daypart, the input context echo, and a ``feedback`` slot that
starts as ``null`` and is updated by ``set_feedback`` when the
operator votes.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1


class CompanionRecorder:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir) / "companion"
        self._lock = threading.Lock()

    def record(
        self,
        *,
        text: str,
        raw_text: str,
        source: str,
        model_id: str,
        status: str,
        filter_reason: str,
        language: str,
        tone: str,
        daypart: str | None,
        trigger: str,
        context_echo: dict[str, Any] | None,
        user_message: str | None,
        elapsed_ms: int,
    ) -> dict[str, Any]:
        now = datetime.now()
        trigger_id = (
            f"utt_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
        )
        entry: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "trigger_id": trigger_id,
            "ts": now.isoformat(),
            "trigger": trigger,
            "text": text,
            "raw_text": raw_text,
            "source": source,
            "model_id": model_id,
            "status": status,
            "filter_reason": filter_reason or None,
            "language": language,
            "tone": tone,
            "daypart": daypart,
            "elapsed_ms": elapsed_ms,
            "feedback": None,
            "context": context_echo or {},
        }
        if user_message is not None:
            entry["user_message"] = user_message
        with self._lock:
            path = self._path_for(now)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def recent(self, *, limit: int = 50, hours: int = 168) -> list[dict[str, Any]]:
        """Return at most `limit` recent entries, newest first.

        ``hours`` bounds how far back we scan files; default is one
        week which is plenty for the API.
        """
        limit = max(1, min(int(limit), 200))
        cutoff = datetime.now() - timedelta(hours=max(1, int(hours)))
        rows: list[dict[str, Any]] = []
        with self._lock:
            if not self.base_dir.exists():
                return []
            for path in sorted(self.base_dir.glob("utterances_*.jsonl"), reverse=True):
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        ts = datetime.fromisoformat(str(row.get("ts") or ""))
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if ts < cutoff:
                        continue
                    rows.append(row)
                if len(rows) >= limit * 4:
                    # generous early-cut so we don't read months of files
                    break
        rows.sort(key=lambda r: str(r.get("ts") or ""), reverse=True)
        return rows[:limit]

    def set_feedback(self, trigger_id: str, vote: str | None) -> dict[str, Any] | None:
        if vote not in {"up", "down", None}:
            raise ValueError("vote must be 'up', 'down', or null")
        with self._lock:
            if not self.base_dir.exists():
                return None
            for path in sorted(self.base_dir.glob("utterances_*.jsonl")):
                updated = self._update_file(path, trigger_id=trigger_id, vote=vote)
                if updated is not None:
                    return updated
        return None

    def _path_for(self, now: datetime) -> Path:
        return self.base_dir / f"utterances_{now.date().isoformat()}.jsonl"

    def _update_file(
        self, path: Path, *, trigger_id: str, vote: str | None
    ) -> dict[str, Any] | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        rows: list[dict[str, Any]] = []
        updated: dict[str, Any] | None = None
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("trigger_id") == trigger_id:
                row["feedback"] = vote
                updated = row
            rows.append(row)
        if updated is None:
            return None
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        tmp.replace(path)
        return updated
