-- WatchMyBirds telemetry heartbeats schema.
--
-- One row per (installation_id, UTC date). The PRIMARY KEY enforces
-- "at most one heartbeat per install per day" at the DB layer — even
-- if a misbehaving client pings 100 times, the table stays clean.
--
-- 90-day retention is enforced by a daily cron in the Worker that
-- runs DELETE WHERE ts < unixepoch() - 90*86400.
--
-- Privacy posture: no IP, no country, no locale, no exact RAM bytes,
-- no kernel version, no Pi model string. See the Privacy page for
-- the full list of what we explicitly do NOT collect.

CREATE TABLE IF NOT EXISTS heartbeats (
    installation_id   TEXT    NOT NULL,
    date              TEXT    NOT NULL,         -- 'YYYY-MM-DD' (UTC)
    app_version       TEXT    NOT NULL,
    os                TEXT    NOT NULL,         -- 'linux', 'darwin', 'windows'
    arch              TEXT    NOT NULL,         -- 'aarch64', 'x86_64', 'armv7l'
    cpu_count         INTEGER NOT NULL,
    total_ram_gb      INTEGER NOT NULL,         -- rounded whole GB
    python_version    TEXT    NOT NULL,
    detector_variant  TEXT    NOT NULL,
    ts                INTEGER NOT NULL,         -- unix epoch seconds
    PRIMARY KEY (installation_id, date)
);

-- Index for retention sweeps (cron DELETE on ts).
CREATE INDEX IF NOT EXISTS idx_heartbeats_ts ON heartbeats(ts);

-- Index for date-range aggregations (DAU/WAU/MAU queries).
CREATE INDEX IF NOT EXISTS idx_heartbeats_date ON heartbeats(date);

-- Daily aggregates: the aggregate-only retention model.
--
-- A nightly cron at 04:30 UTC aggregates the previous UTC day's raw
-- heartbeats into one row per (date, app_version, hardware cohort,
-- detector_variant), then deletes those raw rows. This makes the
-- privacy claim stronger:
--
--   Raw retention:           "we keep individual heartbeats up to
--   90 days, then delete"
--
--   Aggregation (this table): "we keep individual heartbeats less
--   than 24 hours; we keep only counts after that"
--
-- The aggregate has NO installation_id — there is no way to track
-- an individual install across days from this table alone.
--
-- Aggregates are kept indefinitely (they are tiny, ~1 row per
-- distinct cohort per day). If retention later becomes a concern,
-- the table can be pruned by date.
CREATE TABLE IF NOT EXISTS daily_aggregates (
    date              TEXT    NOT NULL,    -- 'YYYY-MM-DD' (UTC)
    app_version       TEXT    NOT NULL,
    os                TEXT    NOT NULL,
    arch              TEXT    NOT NULL,
    cpu_count         INTEGER NOT NULL,
    total_ram_gb      INTEGER NOT NULL,
    detector_variant  TEXT    NOT NULL,
    install_count     INTEGER NOT NULL,    -- COUNT(DISTINCT installation_id) on this cohort+date
    PRIMARY KEY (date, app_version, os, arch, cpu_count, total_ram_gb, detector_variant)
);

-- Index for date-range aggregate reads (DAU trend queries).
CREATE INDEX IF NOT EXISTS idx_daily_aggregates_date ON daily_aggregates(date);
