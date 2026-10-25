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

CREATE INDEX IF NOT EXISTS idx_heartbeats_date ON heartbeats(date);

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

CREATE TABLE IF NOT EXISTS diagnostic_reports (
    installation_id TEXT NOT NULL,
    date TEXT NOT NULL,
    consent_version INTEGER NOT NULL CHECK (consent_version = 2),
    deployment TEXT NOT NULL,
    metrics TEXT NOT NULL,
    ts INTEGER NOT NULL,
    PRIMARY KEY (installation_id, date)
);
CREATE INDEX IF NOT EXISTS idx_diagnostic_reports_date ON diagnostic_reports(date);
CREATE TABLE IF NOT EXISTS telemetry_jobs (
    name TEXT PRIMARY KEY,
    last_success INTEGER NOT NULL
);
