"""Image review projection shared by gallery corrections and Review."""

import sqlite3

from utils.db.detections import object_review_pending_sql, table_columns
from utils.db.review_queue import update_review_status
from utils.review_metadata import (
    REVIEW_STATUS_CONFIRMED_BIRD,
    REVIEW_STATUS_NO_BIRD,
    REVIEW_STATUS_UNTAGGED,
)


def refresh_review_image_visibility(
    conn: sqlite3.Connection, filename: str, gallery_threshold: float
) -> str:
    image_fact = conn.execute(
        """
        SELECT answer_value FROM current_human_label_facts
        WHERE scope = 'image' AND image_filename = ? AND fact_type = 'bird_presence'
        """,
        (filename,),
    ).fetchone()
    # A box or species projection cannot retract a whole-image human answer.
    if image_fact and image_fact[0] == "absent":
        conn.execute(
            "UPDATE images SET review_status = ? WHERE filename = ?",
            (REVIEW_STATUS_NO_BIRD, filename),
        )
        return REVIEW_STATUS_NO_BIRD

    active_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM detections d
        WHERE d.image_filename = ?
          AND COALESCE(d.status, 'active') = 'active'
        """,
        (filename,),
    ).fetchone()[0]

    if active_count == 0:
        if image_fact and image_fact[0] == "present":
            review_status = REVIEW_STATUS_CONFIRMED_BIRD
        else:
            review_status = REVIEW_STATUS_UNTAGGED
        if review_status == REVIEW_STATUS_UNTAGGED:
            conn.execute(
                """
                UPDATE images
                SET review_status = 'untagged'
                WHERE filename = ?
                """,
                (filename,),
            )
        else:
            update_review_status(conn, [filename], review_status, commit=False)
        return review_status

    pending = object_review_pending_sql(table_columns(conn, "detections"))
    unresolved = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM detections d
        WHERE d.image_filename = ?
          AND COALESCE(d.status, 'active') = 'active'
          AND {pending}
        """,
        (filename, gallery_threshold),
    ).fetchone()[0]

    if unresolved == 0:
        update_review_status(
            conn, [filename], REVIEW_STATUS_CONFIRMED_BIRD, commit=False
        )
        return REVIEW_STATUS_CONFIRMED_BIRD

    conn.execute(
        """
        UPDATE images
        SET review_status = 'untagged'
        WHERE filename = ?
        """,
        (filename,),
    )
    return REVIEW_STATUS_UNTAGGED
