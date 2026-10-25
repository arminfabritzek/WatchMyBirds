"""One photo-quality order for SQL readers and in-memory cover selection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class _Criterion:
    column: str
    kind: Literal["flag", "sharp", "ptz", "number", "text"]
    default: float | str = 0.0

    def value(self, row: Mapping[str, Any]) -> int | float | str:
        raw = row.get(self.column)
        if self.column == "timestamp":
            raw = raw or row.get("image_timestamp")
        if self.kind == "sharp":
            return int(raw is None or float(raw) != 0)
        if self.kind == "ptz":
            return int(raw in ("preset", "manual_drive"))
        if self.kind == "flag":
            return int(float(raw or 0) != 0)
        if self.kind == "text":
            return str(raw or self.default)
        return float(self.default if raw is None else raw)

    def sql(self, columns: Mapping[str, str]) -> str:
        column = columns.get(self.column, self.column)
        if self.kind == "sharp":
            return f"(COALESCE({column}, 1) != 0)"
        if self.kind == "ptz":
            return f"(COALESCE({column}, '') IN ('preset', 'manual_drive'))"
        if self.kind == "flag":
            return f"(COALESCE({column}, 0) != 0)"
        default = "''" if self.kind == "text" else str(self.default)
        return f"COALESCE({column}, {default})"


_CRITERIA = (
    _Criterion("is_favorite", "flag"),
    _Criterion("quality_gallery_ok", "sharp"),
    _Criterion("ptz_origin", "ptz"),
    _Criterion("is_gallery_eligible", "flag"),
    _Criterion("aesthetic_score", "number", -1.0),
    _Criterion("score", "number"),
    _Criterion("bbox_quality", "number"),
    _Criterion("timestamp", "text", ""),
)


def cover_sort_key(row: Mapping[str, Any]) -> tuple[int | float | str, ...]:
    """Higher is better; callers retain their own handling of exact ties."""
    return tuple(criterion.value(row) for criterion in _CRITERIA)


def cover_order_sql(
    columns: Mapping[str, str] | None = None, *, alias: str = ""
) -> str:
    """Return descending ORDER BY terms for trusted column expressions."""
    expressions = (
        {criterion.column: f"{alias}.{criterion.column}" for criterion in _CRITERIA}
        if alias
        else {}
    )
    expressions.update(columns or {})
    return ", ".join(f"{criterion.sql(expressions)} DESC" for criterion in _CRITERIA)
