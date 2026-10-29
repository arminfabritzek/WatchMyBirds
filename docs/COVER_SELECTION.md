# Cover Selection

How WatchMyBirds chooses the photo that represents a species, a day, an event
or a report entry. This document describes the implemented behaviour; when it
and the code disagree, the code in `utils/cover_selection.py` is authoritative
and this document must be corrected.

## The shared order

Every surface ranks candidate photos with one tuple, defined once in
`utils/cover_selection.py` (`_CRITERIA`). Python callers use
`cover_sort_key(row)`; SQL callers use `cover_order_sql()`. Higher wins, and
the criteria are compared strictly in this order. A later criterion only
matters when all earlier ones are equal.

| # | Column | Value used for ranking | Missing value counts as | Set by |
|---|--------|------------------------|-------------------------|--------|
| 1 | `is_favorite` | 1 if non-zero, else 0 | 0 | The star in the UI (manual only) |
| 2 | `quality_gallery_ok` | 0 only if the value is exactly 0, else 1 | 1 (unscored counts as sharp) | Nightly sharpness job |
| 3 | `ptz_origin` | 1 if `preset` or `manual_drive`, else 0 | 0 | Capture context of a PTZ camera |
| 4 | `is_gallery_eligible` | 1 if non-zero, else 0 | 0 | Nightly aesthetic tagger ("AI pick") |
| 5 | `aesthetic_score` | the number | -1.0 | Nightly aesthetic tagger |
| 6 | `score` | the number | 0.0 | Detection/classification at capture |
| 7 | `bbox_quality` | the number | 0.0 | Box-quality signal at capture |
| 8 | `timestamp` | text, so the newest capture wins | empty text | Capture time of the photo |

Consequences of this order:

- A favorite always beats a non-favorite.
- Among several favorites, the star does not distinguish them. The winner is
  decided by criteria 2–8. An AI pick (criterion 4) therefore beats a newer
  favorite that is not an AI pick, regardless of aesthetic score. The species
  page and gallery day covers opt out of this; see [Rotation](#rotation).
- The capture time is only the last tie-breaker. The time a star was set is
  not stored anywhere, so "most recently starred" cannot influence the result.
- Whether the original photo still exists is not part of the order. A
  favorite whose original has expired can stay the cover; its crop is then
  whatever size was stored before the original expired.
- `quality_gallery_ok` only ranks; it never hides a photo from selection.

### How the nightly signals are produced

- **`quality_gallery_ok`** (`web/services/nightly_jobs/sharpness_job.py`):
  crops in the bottom `GALLERY_QUALITY_BOTTOM_PCT` percent (default 15) of
  this station's sharpness scores get 0, all other scored crops get 1. With
  fewer than `GALLERY_QUALITY_MIN_SCORED` (default 200) scored crops, or with
  the percentage set to 0, every scored crop gets 1. Unscored crops stay NULL.
- **`is_gallery_eligible`** (`scripts/aesthetic_tag_nightly.py`): per species
  and day, the top `TOP_N_PER_SPECIES_PER_DAY` (3) confirmed detections by
  `aesthetic_score` are set to 1, provided their score is at least
  `MIN_SCORE_FOR_TAG` (0.10). The job never touches `is_favorite` and never
  resets the flag on later runs, except for rows re-scored with `--rescore`.

## Surfaces

Each surface first restricts the candidate set, then applies the shared order.
Only the restriction and the tie-breaker after the shared order differ.

| Surface | Candidates | After the shared order | Code |
|---------|-----------|------------------------|------|
| Species page (`/species`) | Current observations of each species, including Unknown bird. If a date range is set and the species has observations in it, only those; otherwise all of them | Rotates among favorites (`rotate_favorites=True`), then `object_key` ascending | `fetch_species_covers` in `utils/db/detections.py` |
| Station report highlights | Covers from the species query without rotation for the report period, sorted again by the shared order; the first six are shown | — | `core/automatic_station_report.py` |
| Gallery day covers (`/gallery`) | Current observations of each day, excluding Unknown species | Rotates among favorites (`rotate_favorites=True`), then `detection_id` descending, then `object_key` | `fetch_daily_covers` via `fetch_gallery_days` in `utils/db/detections.py` |
| Daily report / Telegram summary | Admitted observations of that day, per species | `object_key` ascending | `utils/daily_report.py` |
| Live page species summary | Today's detections, per species | Random choice only among photos whose entire tuple is equal | `pick_cover_for_group` in `core/gallery_core.py` |
| Today's visitors / Best of Species boards | Today's detections, or the SQL candidate pool (`fetch_species_story_board_candidates`, ranked by the shared order, then `detection_id` descending) | Main photo as in the row above. Up to three extra frames are drawn at random from the next six ranked photos on every page load | `build_species_story_board`, `_choose_story_board_frames` in `core/gallery_core.py` |
| Random favorites | Admitted observations that are a favorite or an AI pick, best one per species | `RANDOM()` among exact ties; species order is random | `fetch_random_favorites` in `utils/db/detections.py` |
| Event cover (Review) | Members of one event; context-only members are skipped when actionable members exist | Capture timestamp, then `detection_id` (both newest first) | `core/events.py` |
| Best object of a new frame | Confirmed objects of the frame just captured | — | `detectors/detection_manager.py` |
| Telegram alert batch | Pending detections per species | — | `detectors/services/notification_service.py` |

Notes on the last two rows: when a frame is captured, only `score` and
`bbox_quality` are known, so the best object of a frame is ranked by those two
criteria. The Telegram batch reads the stored detection row (plus the image's
`ptz_origin`) and ranks with the full order, using whatever nightly signals
already exist; for fresh detections those are still empty.

## Rotation

The species page and gallery day covers rotate among favorites. With
`cover_order_sql(rotate_favorites=True)`, a `RANDOM()` term follows the
favorite criterion, so every page load draws one of the species' or day's
favorites. Photos without a star keep criteria 2–8, so a species or day
without favorites shows the same photo on every load.

All other covers do not rotate. Reports, event covers, the Live page species
summary and notifications always return the same photo for the same data.
Randomness otherwise exists only in the extra frames of the Today's visitors
and Best of Species boards, in the species order of the random-favorites list,
and between photos whose complete ranking tuple is identical.

## Known gaps

- `fetch_species_covers` accepts `min_score` and `favorites_only` but does not
  use them.
- No timestamp is stored when a favorite is set.
- AI picks are never withdrawn automatically, so old AI picks keep their
  advantage over newer photos indefinitely.
