"""Compare isolated gallery read paths, optionally against an older source tree.

Example: python scripts/benchmark_gallery_reads.py --size 10000 --case species
Each invocation creates and removes a synthetic database in the system temp dir.
Cold means process-cache cold; the OS filesystem cache is not flushed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sqlite3
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class MeasuredCursor(sqlite3.Cursor):
    def fetchall(self) -> list:
        rows = super().fetchall()
        self.connection.materialized += len(rows)
        return rows

    def fetchone(self) -> Any:
        row = super().fetchone()
        self.connection.materialized += int(row is not None)
        return row

    def __iter__(self) -> MeasuredCursor:
        return self

    def __next__(self) -> Any:
        row = super().__next__()
        self.connection.materialized += 1
        return row


class MeasuredConnection(sqlite3.Connection):
    materialized = 0

    def execute(self, sql: str, parameters: Any = ()) -> MeasuredCursor:
        return self.cursor(factory=MeasuredCursor).execute(sql, parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument(
        "--case",
        choices=("species", "covers", "days", "events", "parallel", "ingestion"),
        required=True,
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--skip-allocation-replay",
        action="store_true",
        help="Measure process peak RSS without repeating the read under tracemalloc",
    )
    parser.add_argument(
        "--parallel-rows-only",
        action="store_true",
        help="Coalesce event-row queries without repeating event grouping",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    from config import get_config
    from core.gallery_core import pick_cover_for_group
    from utils.db import detections, events
    from utils.db.connection import _init_schema
    from web.services import cache_service, human_label_service

    with tempfile.TemporaryDirectory(prefix="wmb-read-benchmark-") as temporary:
        get_config()["OUTPUT_DIR"] = temporary
        db = Path(temporary) / "images.db"

        def connect() -> MeasuredConnection:
            conn = sqlite3.connect(db, factory=MeasuredConnection)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA mmap_size=2147483648")
            conn.execute("PRAGMA cache_size=100000")
            conn.materialized = 0
            return conn

        conn = connect()
        _init_schema(conn)
        for start in range(0, args.size, 10000):
            images, dets, classes = [], [], []
            for index in range(start, min(args.size, start + 10000)):
                # Spread across a year; enough rows per species to exercise pagination.
                stamp = (
                    datetime(2025, 1, 1)
                    + timedelta(days=index % 365, seconds=index % 86400)
                ).strftime("%Y%m%d_%H%M%S")
                name = f"{stamp}_{index}.jpg"
                species = f"Species_{index % 20}"
                images.append((name, stamp))
                dets.append((index + 1, name, stamp, int(index % 101 == 0)))
                classes.append((index + 1, species))
            conn.executemany(
                "INSERT INTO images(filename,timestamp,source_id,review_status) VALUES (?, ?, 1, 'confirmed_bird')",
                images,
            )
            conn.executemany(
                """INSERT INTO detections(detection_id,image_filename,created_at,is_favorite,
                bbox_x,bbox_y,bbox_w,bbox_h,od_class_name,od_confidence,score,status,decision_state,decision_level)
                VALUES (?, ?, ?, ?, .2,.2,.3,.3,'bird',.95,.95,'active','confirmed','species')""",
                dets,
            )
            conn.executemany(
                "INSERT INTO classifications(detection_id,cls_class_name,cls_confidence,rank,status) VALUES (?, ?, .95, 1,'active')",
                classes,
            )
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
        conn.materialized = 0
        queries: set[str] = set()
        conn.set_trace_callback(
            lambda sql: (
                queries.add(sql)
                if sql.lstrip().upper().startswith(("SELECT", "WITH"))
                else None
            )
        )
        modern = hasattr(detections, "fetch_species_page")

        def read() -> Any:
            if args.case == "events":
                return events.get_events_cached(conn)
            if modern:
                if args.case == "species":
                    return detections.fetch_species_page(
                        conn, "Species_1", min_score=0.5
                    )[0]
                if args.case == "covers":
                    return detections.fetch_species_covers(conn, min_score=0.5)
                return detections.fetch_gallery_days(conn, min_score=0.5)[0]
            rows = [
                dict(row)
                for row in detections.fetch_detections_for_gallery(
                    conn, order_by="time"
                )
            ]
            if args.case == "species":
                return [
                    r
                    for r in rows
                    if r["species_key"] == "Species_1" and (r["score"] or 0) >= 0.5
                ][:50]
            if args.case == "covers":
                groups: dict[str, list] = {}
                for row in rows:
                    if (row["score"] or 0) >= 0.5:
                        groups.setdefault(row["species_key"], []).append(row)
                return [pick_cover_for_group(group) for group in groups.values()]
            human_label_service.fetch_detection_review_states(
                conn, [r["detection_id"] for r in rows]
            )
            return detections.fetch_daily_covers(conn, min_score=0.5)

        samples = []
        if args.case == "ingestion":
            barrier = threading.Barrier(3)
            observations = []
            commits = []

            def ingest() -> None:
                with closing(connect()) as writer:
                    barrier.wait()
                    for index in range(100):
                        stamp = (
                            datetime(2026, 1, 1) + timedelta(seconds=index)
                        ).strftime("%Y%m%d_%H%M%S")
                        name = f"{stamp}_arrival.jpg"
                        cursor = writer.execute(
                            "INSERT INTO images(filename,timestamp,source_id,review_status) VALUES (?, ?, 1, 'confirmed_bird')",
                            (name, stamp),
                        )
                        cursor = writer.execute(
                            """INSERT INTO detections(image_filename,created_at,bbox_x,bbox_y,bbox_w,bbox_h,
                            od_class_name,od_confidence,score,status,decision_state,decision_level)
                            VALUES (?, ?, .2,.2,.3,.3,'bird',.95,.95,'active','confirmed','species')""",
                            (name, stamp),
                        )
                        writer.execute(
                            "INSERT INTO classifications(detection_id,cls_class_name,cls_confidence,rank,status) VALUES (?, 'Species_1', .95, 1,'active')",
                            (cursor.lastrowid,),
                        )
                        writer.commit()
                        commits.append(time.perf_counter())
                        time.sleep(0.005)

            def reader() -> None:
                with closing(connect()) as connection:
                    barrier.wait()
                    for _ in range(2):
                        connection.materialized = 0
                        began = time.perf_counter()
                        connection.execute("BEGIN")
                        if modern:
                            rows = detections.fetch_species_page(
                                connection, "Species_1", min_score=0.5
                            )[0]
                        else:
                            all_rows = [
                                dict(row)
                                for row in detections.fetch_detections_for_gallery(
                                    connection, order_by="time"
                                )
                            ]
                            rows = [
                                row
                                for row in all_rows
                                if row["species_key"] == "Species_1"
                                and (row["score"] or 0) >= 0.5
                            ][:50]
                            del all_rows
                        connection.commit()
                        ended = time.perf_counter()
                        observations.append(
                            dict(
                                seconds=round(ended - began, 4),
                                rows_materialized=connection.materialized,
                                result_rows=len(rows),
                                began=began,
                                ended=ended,
                            )
                        )
                        del rows

            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(ingest), pool.submit(reader), pool.submit(reader)]
                for job in jobs:
                    job.result()
            for observation in observations:
                observation["overlapping_commits"] = sum(
                    observation["began"] <= stamp <= observation["ended"]
                    for stamp in commits
                )
                del observation["began"], observation["ended"]
            samples.append(dict(arrivals=len(commits), readers=observations))
        elif args.case == "parallel":
            cache_service.clear()
            barrier = threading.Barrier(4)
            builds = []
            row_counts = []

            def build() -> Any:
                builds.append(1)
                with closing(connect()) as reader:
                    result = (
                        detections.fetch_detections_for_gallery(reader, order_by="time")
                        if args.parallel_rows_only
                        else events.fetch_events_for_analysis(reader)
                    )
                    row_counts.append(reader.materialized)
                    return result

            def worker() -> int:
                barrier.wait()
                return len(cache_service.cached("benchmark.events", 60, build))

            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: worker(), range(4)))
            samples.append(
                dict(
                    seconds=round(time.perf_counter() - began, 4),
                    builders=len(builds),
                    builder_kind="event_rows"
                    if args.parallel_rows_only
                    else "event_grouping",
                    rows_materialized=sum(row_counts),
                    results=results,
                )
            )
        else:
            events.clear_events_cache()
            for label in ("cold", "warm"):
                gc.collect()
                conn.materialized = 0
                began = time.perf_counter()
                result = read()
                samples.append(
                    dict(
                        state=label,
                        seconds=round(time.perf_counter() - began, 4),
                        rows_materialized=conn.materialized,
                        result_rows=len(result),
                    )
                )
                del result
            if not args.skip_allocation_replay:
                events.clear_events_cache()
                gc.collect()
                tracemalloc.start()
                result = read()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                del result
                samples.append(dict(python_peak_mib=round(peak / 1024**2, 2)))
        plans = set()
        conn.set_trace_callback(None)
        for query in queries:
            for row in conn.execute("EXPLAIN QUERY PLAN " + query):
                plans.add(row["detail"])
        report = dict(
            size=args.size,
            case=args.case,
            implementation="current" if modern else "baseline",
            machine=platform.machine(),
            python=platform.python_version(),
            sqlite=sqlite3.sqlite_version,
            samples=samples,
            process_peak_rss_mib=round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2 if sys.platform == "darwin" else 1024),
                2,
            ),
            query_plans=sorted(plans),
        )
        print(json.dumps(report))
        events.clear_events_cache()
        conn.close()


if __name__ == "__main__":
    main()
