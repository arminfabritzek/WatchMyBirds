# Aesthetic Auto-Tagger — Deployment

A nightly batch job that scores every new bird crop with a CLIP
"facing-camera" probability and auto-tags the top-3-per-species-per-day as
`is_favorite=1` for selected species (small songbirds where the score has
been validated).

## Architecture (2026-05-02 onward)

The scheduler runs **inside the main app process** as a daemon thread
(`web/services/aesthetic_tag_scheduler.py`) so Pi and Docker behave
identically. An earlier design ran the tagger from a systemd timer
against a separate venv; that path was removed once the in-process
scheduler became the single production path on both targets.

Zero-touch deploy: `pip install -r requirements.txt -r requirements-aesthetic.txt`
and start the app — the scheduler initialises, daemon thread polls
once per 30 s, fires `aesthetic_tag_nightly.main_with_args([])` once
per day at the configured `AESTHETIC_TAG_TIME` (default 02:10).

**The tagger is a default-ON shipped feature.** Every standard build
installs `requirements-aesthetic.txt`: the `Dockerfile`, both CI image
workflows (`build-release.yml` / `build-dev.yml`), and the RPi OTA
updater (`rpi/update.sh`). To turn it off, set
`AESTHETIC_TAG_ENABLED=False` — that is the supported opt-out, not a
separate image.

### Why two requirement files

The split exists for **CPU-index isolation**, not because the tagger
is optional. `torch` / `torchvision` must resolve from the PyTorch CPU
index (`--index-url https://download.pytorch.org/whl/cpu`); a bare
`pip install torchvision` on PyPI drags ~3.8 GB of CUDA wheels onto the
CPU-only Pi. Keeping that index directive in a separate `-r` file with
its own header means the main `requirements.txt` resolver run is never
forced through the PyTorch index. Merging the files would gain nothing
(aesthetic ships everywhere regardless) and lose that isolation.

If the packages are genuinely absent (a hand-stripped venv), the
scheduler logs a single warning and stays idle; the app boots normally
without the tagger. This is a safety net, not a shipped "slim" build.

> **Deploy caveat.** A code-only deploy (copying source without
> re-running `pip`) will not pick up a newly added aesthetic dependency
> (e.g. `open_clip_torch`). The scheduler then silently falls back to
> idle despite the flag being on. Only a fresh image build and the OTA
> updater (`rpi/update.sh`) install aesthetic deps automatically. After
> adding an aesthetic dependency, install it explicitly into the target
> venv with the CPU index
> (`pip install --index-url https://download.pytorch.org/whl/cpu ...`)
> and verify the imported version — don't trust the pip success line
> alone on a root-owned venv.

## What it ships

| File | Purpose |
|---|---|
| `scripts/aesthetic_tag_nightly.py` | The worker. CLI flags: `--since`, `--limit`, `--dry-run`, `--skip-tagging`. Also imported in-process. |
| `scripts/test_aesthetic_tag_nightly.py` | Worker smoke tests. |
| `web/services/aesthetic_tag_scheduler.py` | Daemon-thread scheduler (replaces systemd). |
| `tests/test_aesthetic_tag_scheduler.py` | Scheduler unit tests (8). |
| `tests/test_aesthetic_integration.py` | UI ranking tests (10). |
| `requirements-aesthetic.txt` | torch + open_clip deps, split out for CPU-index isolation (installed by every standard build). |
| `Dockerfile` | Pip-installs both requirement files in two RUN steps. |
| `utils/db/connection.py` | Adds `aesthetic_score REAL` + `aesthetic_score_at TEXT` to `detections`. |

## Schema additions

Two new nullable columns on `detections`. Idempotent migration via
`_ensure_column_on_table`:

```python
_ensure_column_on_table(conn, "detections", "aesthetic_score", "REAL")
_ensure_column_on_table(conn, "detections", "aesthetic_score_at", "TEXT")
```

`aesthetic_score`: float in `[0, 1]`. CLIP softmax probability for "bird with
face/head/chest visible toward viewer" vs. "bird seen from behind".

`aesthetic_score_at`: ISO-8601 UTC timestamp of last computation. Acts as
idempotency key — runs that find both columns populated skip the detection.

The job sets `is_favorite = 1` and `rating_source = 'auto'` for selected
detections; manual favorites (`rating_source = 'manual'`) are never touched.

## Validation summary (2026-04-30)

Two complementary validation passes:

### Per-species isolated AUC (single-score, no bucketing)

| Species | N labels | AUC |
|---|---|---|
| Parus_major (Kohlmeise) | 47 + 19 | 0.69–0.83 |
| unknown (CLS-rejected) | 68 | 0.67 |
| Cyanistes_caeruleus (Blaumeise) | 25 | 0.43–0.64 |
| Columba_palumbus (Ringeltaube) | 56 + 113 | 0.35–0.58 |

By isolated AUC alone, only Parus would qualify.

### Real-world Pi-output simulation (5 days × 12 species, top-3 per bucket)

Manual judgement on 111 picks: **~67 % were good** ("2 of 3 are good").
This is ~3–6× lift over random selection given a typical favorite-rate
of 10–20 % in the underlying pool.

Source data came from a local validation run over five days of station output.

### Decision

The per-(species, day) bucketing self-corrects: even when a species'
absolute AUC is low, it surfaces *its* best detections of the day, and
the user accepts ~33 % noise as the cost of system-wide species coverage.

**Production default**: `TAGGABLE_SPECIES = set()` (empty, all species
eligible). To restrict, populate with species names. The `MIN_SCORE_FOR_TAG`
threshold (default 0.10) prevents "best-of-a-bad-day" tags on weak buckets.

## Resource estimates (Pi 5, CPU only)

| Metric | Value |
|---|---|
| Model RAM (CLIP ViT-B/32) | ~700 MB |
| Inference time per crop | ~0.7 s (1.4 img/s, measured 2026-05-01) |
| 1000 crops/night | ~12 min |
| Disk: torch + open_clip stack | ~900 MB (in the app venv, `--index-url cpu`) |
| Disk: HF model cache | ~340 MB |
| App memory headroom | ~700 MB transient while CLIP is loaded |

The job is single-threaded by default (`OMP_NUM_THREADS=2`). Don't push
parallelism higher — the detector pipeline has priority during the day, the
tagger runs at 02:10 when traffic is expected to be low.

## First-time deploy

**Pi (image-based)**: the image build runs `pip install -r requirements.txt
-r requirements-aesthetic.txt` against the app venv. After flash, the
scheduler is automatically active when the app boots — no manual setup.

**Docker**: the Dockerfile installs both requirement files in two RUN
steps. After `docker pull && docker run`, the scheduler is active.

**Dev machine**: `pip install --index-url https://download.pytorch.org/whl/cpu
--extra-index-url https://pypi.org/simple -r requirements-aesthetic.txt`
adds the optional stack to your existing venv.

The CLIP model (~340 MB) downloads on the first scheduler fire and is
then cached. To pre-warm so the first run doesn't stall, run the
following once before the first scheduled fire:

```bash
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"
```

### Configuration

| Config key | Default | Purpose |
|---|---|---|
| `AESTHETIC_TAG_ENABLED` | `True` | Master switch. Set `False` to disable the scheduler entirely. |
| `AESTHETIC_TAG_TIME` | `"02:10"` | When to run, `HH:MM` 24h. |

## Smoke test before going live

Check what the scheduler actually tagged after the first daily fire
(or after manually invoking the worker via `python -m scripts.aesthetic_tag_nightly`):

```sql
SELECT
  COALESCE(c.cls_class_name, 'unknown') AS species,
  substr(d.created_at, 1, 10) AS day,
  COUNT(*) AS auto_tagged
FROM detections d
LEFT JOIN classifications c ON c.detection_id = d.detection_id AND c.rank = 1
WHERE d.is_favorite = 1
  AND d.rating_source = 'auto'
  AND d.created_at > '2026-04-29'
GROUP BY species, day
ORDER BY day DESC, species;
```

Expected: 3 per (species, day) for `Parus_major`, `Cyanistes_caeruleus`,
and `Columba_palumbus` when `TAGGABLE_SPECIES` is set to that conservative
subset. With the current production default (`TAGGABLE_SPECIES = set()`),
all CLS-labelled species are eligible; `unknown` is not tagged unless
`TAG_UNKNOWN_SPECIES` is enabled.

## Operational notes

**Manual run** (Pi or Docker, app's own venv):
```bash
# On the Pi
sudo -u watchmybirds /opt/app/.venv/bin/python \
    /opt/app/scripts/aesthetic_tag_nightly.py --since 2026-04-29 --verbose
# Inside the Docker container
docker exec <container> python /app/scripts/aesthetic_tag_nightly.py --since 2026-04-29 --verbose
```

**Backfill historical detections** (will take hours for large windows):
```bash
python scripts/aesthetic_tag_nightly.py --since 2026-03-01 --skip-tagging
```
The `--skip-tagging` flag computes scores without setting `is_favorite`, so
the historical record stays unchanged but you can later analyze score
distributions per species.

**Untag everything if the formula changes:**
```sql
UPDATE detections SET is_favorite = 0, rating_source = 'auto'
WHERE rating_source = 'auto' AND is_favorite = 1;
UPDATE detections SET aesthetic_score = NULL, aesthetic_score_at = NULL;
```
Then re-run the job.

## Failure modes and recovery

| Symptom | Cause | Fix |
|---|---|---|
| App memory spikes while CLIP is loaded | Model stack needs ~700 MB transient RAM | Disable `AESTHETIC_TAG_ENABLED` on low-RAM hosts, or schedule during idle hours |
| Job takes too long | Backlog, e.g. multi-day catch-up | Run manually with `--since` covering a smaller window |
| All scores ~0.5 | CLIP loaded wrong weights | Delete `~/.cache/huggingface`, re-warm |
| `crop missing for det N` warnings | Crop file deleted while DB row remained | Expected; the job logs and moves on |
| Detector pipeline becomes slow | Concurrent CPU contention | Lower `AESTHETIC_TAGGER_TORCH_THREADS` or raise `AESTHETIC_TAGGER_NICE` |

## Future work

1. **Migrate to ONNX runtime** to drop the ~900 MB torch + open_clip
   stack (~3 h engineering). Pi already has `onnxruntime` for the main
   detector. CLIP-ViT-B/32 has stable ONNX exports.
2. **Per-species formulas** (e.g. `colorfulness + clip_with_food` for
   pigeons) once we have ≥ 100 labels per species.
3. **Logistic-regression score** trained on accumulated user labels. The
   script structure already supports per-species score functions —
   `score_image()` can become a dispatcher.
