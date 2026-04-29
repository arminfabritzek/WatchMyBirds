#!/usr/bin/env bash
# ----------------------------------------------------------------------
# WatchMyBirds USB Backup (Phase 2 of usb-data-backup plan)
#
# Snapshots the SQLite database, captured imagery (originals + derivatives),
# and the installed app code to a USB stick mounted at /mnt/wmb-backup.
#
# Runs as the unprivileged 'watchmybirds' user. NO root, NO polkit, NO sudo.
#
# Three load-bearing mechanics (per the plan):
#   1. SQLite is captured via the online .backup pragma -- WAL-aware,
#      respects writers, no need to stop the app.
#   2. Imagery is rsync'd with --link-dest pointing at the previous
#      successful snapshot, so unchanged frames are HARDLINKED instead
#      of recopied. ext4 is mandatory for this; the mount unit enforces it.
#   3. A COMPLETED marker is written ONLY at the very end. Any directory
#      under snapshots/ without that marker is treated as orphan from a
#      crashed run and is purged on the next invocation.
#
# Exit codes:
#    0 -- snapshot completed and verified
#    2 -- usage error (bad CLI args)
#   10 -- mount missing (stick not present)
#   11 -- wrong filesystem on stick (auto-detected, currently means non-ext4)
#   12 -- not enough free space on stick
#   13 -- DB backup failed
#   14 -- frames rsync failed
#   15 -- app rsync failed
#   16 -- DB integrity check failed (snapshot kept, marked CORRUPT)
#   17 -- SHA verification failed (snapshot kept, marked CORRUPT)
#   18 -- another backup is already running
# ----------------------------------------------------------------------

set -u
# We deliberately do NOT 'set -e' globally -- we want to handle failures
# explicitly so we can mark snapshots CORRUPT instead of leaving orphans.
set -o pipefail

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
readonly MOUNT_POINT="/mnt/wmb-backup"
readonly BACKUP_DEVICE="/dev/disk/by-label/WMB-BACKUP"
readonly SNAPSHOTS_DIR="${MOUNT_POINT}/snapshots"
readonly LATEST_LINK="${MOUNT_POINT}/latest"
readonly LOG_FILE="${MOUNT_POINT}/BACKUP_LOG.txt"

readonly APP_DIR="/opt/app"
readonly DATA_DIR="${APP_DIR}/data"
readonly OUTPUT_DIR="${DATA_DIR}/output"
readonly DB_PATH="${OUTPUT_DIR}/images.db"

# Minimum free space on the stick before starting (bytes). 2 GiB.
readonly MIN_FREE_BYTES=$((2 * 1024 * 1024 * 1024))

# Orphan threshold: dirs without COMPLETED older than this are purged.
readonly ORPHAN_AGE_MIN=60

# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
usage() {
    cat <<EOF >&2
Usage: $(basename "$0") --kind <scheduled|manual>

  --kind scheduled   Daily automatic snapshot via systemd timer.
                     Retention: 7 daily, 4 weekly, 6 monthly.
  --kind manual      Operator-triggered snapshot (UI button or CLI).
                     Retention: 3 most recent manual snapshots.

A '--kind pre-ota' will be added in the follow-up plan that enables
OTA. v1 is intentionally write-only (no restore command here either).
EOF
    exit 2
}

KIND=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind)
            KIND="${2:-}"
            shift 2 || usage
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

case "$KIND" in
    scheduled|manual) ;;
    *) usage ;;
esac

# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------
TS_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_NAME="${STAMP}_${KIND}"
SNAPSHOT_DIR="${SNAPSHOTS_DIR}/${SNAPSHOT_NAME}"

# Log to both stderr (journal capture) and BACKUP_LOG.txt on the stick.
log() {
    local msg
    msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "${msg}" >&2
    # Best-effort -- if the stick disappears mid-run, journal still has it.
    if [[ -d "${MOUNT_POINT}" ]]; then
        echo "${msg}" >> "${LOG_FILE}" 2>/dev/null || true
    fi
}

die() {
    local code="$1"; shift
    log "FATAL($code): $*"
    exit "$code"
}

# ----------------------------------------------------------------------
# Step 1: Mount + filesystem checks
# ----------------------------------------------------------------------
if ! mountpoint -q "${MOUNT_POINT}"; then
    # Touching the path will trigger the systemd .automount unit. If that
    # also fails we know the stick is genuinely absent.
    ls "${MOUNT_POINT}" >/dev/null 2>&1 || true
    if ! mountpoint -q "${MOUNT_POINT}"; then
        if [[ -e "${BACKUP_DEVICE}" ]]; then
            LABEL_FSTYPE="$(blkid -o value -s TYPE "${BACKUP_DEVICE}" 2>/dev/null || true)"
            if [[ -n "${LABEL_FSTYPE}" && "${LABEL_FSTYPE}" != "ext4" ]]; then
                die 11 "USB backup volume has filesystem '${LABEL_FSTYPE}', expected ext4. Reformat per docs/USB_BACKUP.md."
            fi
        fi
        die 10 "USB backup volume not mounted at ${MOUNT_POINT} -- stick missing or wrong label."
    fi
fi

FSTYPE="$(findmnt -n -o FSTYPE "${MOUNT_POINT}" 2>/dev/null || true)"
if [[ "${FSTYPE}" != "ext4" ]]; then
    die 11 "USB backup volume has filesystem '${FSTYPE:-unknown}', expected ext4. Reformat per docs/USB_BACKUP.md."
fi

# The mount unit should hand the mounted root to watchmybirds via
# X-mount.owner/group/mode. Keep this as a defensive runtime check for
# manual mounts or older util-linux versions that ignore those options.
if [[ "$(stat -c '%U' "${MOUNT_POINT}")" != "watchmybirds" ]]; then
    log "Fixing ownership of ${MOUNT_POINT} (was $(stat -c '%U:%G' "${MOUNT_POINT}"))"
    # We can't chown as watchmybirds -- needs to happen once via harden.sh
    # or root manually. Surface the issue but continue if writable.
    if [[ ! -w "${MOUNT_POINT}" ]]; then
        die 11 "Mount point not writable for $(id -un); run as root once: chown -R watchmybirds:watchmybirds ${MOUNT_POINT}"
    fi
fi

mkdir -p "${SNAPSHOTS_DIR}" || die 11 "Cannot create ${SNAPSHOTS_DIR}"

# Serialize scheduled and manual runs. The lock lives on the mounted stick,
# so it is shared across the systemd service and the web-triggered process
# even when either service uses PrivateTmp.
exec 9>"${MOUNT_POINT}/.wmb-backup.lock" || die 11 "Cannot create backup lock file"
if ! flock -n 9; then
    die 18 "Another USB backup is already running."
fi

# ----------------------------------------------------------------------
# Step 2: Orphan cleanup
# ----------------------------------------------------------------------
# Any snapshot directory without COMPLETED that's older than ORPHAN_AGE_MIN
# minutes is from a crashed previous run -- purge it before we start, so
# space estimates in the next step are accurate.
shopt -s nullglob
for dir in "${SNAPSHOTS_DIR}"/*/; do
    [[ -d "${dir}" ]] || continue
    if [[ ! -f "${dir}COMPLETED" ]] && [[ ! -f "${dir}CORRUPT" ]]; then
        # find -mmin checks modification time of the directory.
        if [[ -n "$(find "${dir}" -maxdepth 0 -mmin +${ORPHAN_AGE_MIN} 2>/dev/null)" ]]; then
            log "Purging orphan snapshot: ${dir##*/snapshots/}"
            rm -rf "${dir}"
        fi
    fi
done
shopt -u nullglob

# ----------------------------------------------------------------------
# Step 3: Disk space guard
# ----------------------------------------------------------------------
FREE_BYTES="$(df -PB1 "${MOUNT_POINT}" | awk 'NR==2 {print $4}')"
if [[ -z "${FREE_BYTES}" || "${FREE_BYTES}" -lt "${MIN_FREE_BYTES}" ]]; then
    die 12 "Free space on stick (${FREE_BYTES:-0} B) below threshold (${MIN_FREE_BYTES} B). Prune snapshots or use a larger stick."
fi

# ----------------------------------------------------------------------
# Step 4: Resolve previous successful snapshot for --link-dest
# ----------------------------------------------------------------------
PREV_SNAPSHOT=""
if [[ -L "${LATEST_LINK}" ]]; then
    cand="$(readlink -f "${LATEST_LINK}" 2>/dev/null || true)"
    if [[ -n "${cand}" && -f "${cand}/COMPLETED" ]]; then
        PREV_SNAPSHOT="${cand}"
    fi
fi
# Fallback: scan for newest COMPLETED snapshot if symlink is stale.
if [[ -z "${PREV_SNAPSHOT}" ]]; then
    while IFS= read -r dir; do
        if [[ -f "${dir}/COMPLETED" ]]; then
            PREV_SNAPSHOT="${dir}"
            break
        fi
    done < <(find "${SNAPSHOTS_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)
fi

if [[ -n "${PREV_SNAPSHOT}" ]]; then
    log "Using --link-dest base: ${PREV_SNAPSHOT##*/}"
else
    log "No previous snapshot found -- this run will be a full copy."
fi

# ----------------------------------------------------------------------
# Step 5: Create snapshot skeleton
# ----------------------------------------------------------------------
log "Starting ${KIND} snapshot: ${SNAPSHOT_NAME}"
mkdir -p "${SNAPSHOT_DIR}/data" "${SNAPSHOT_DIR}/app" || die 11 "Cannot create snapshot tree at ${SNAPSHOT_DIR}"
echo "${KIND}" > "${SNAPSHOT_DIR}/kind"

mark_corrupt() {
    local reason="$1"
    log "Marking snapshot CORRUPT: ${reason}"
    {
        echo "reason: ${reason}"
        echo "ts: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${SNAPSHOT_DIR}/CORRUPT" 2>/dev/null || true
}

# ----------------------------------------------------------------------
# Step 6: SQLite database (online .backup, WAL-safe)
# ----------------------------------------------------------------------
DB_DST="${SNAPSHOT_DIR}/data/images.db"
DB_BYTES=0
if [[ -f "${DB_PATH}" ]]; then
    log "Snapshotting SQLite database via .backup..."
    if sqlite3 "${DB_PATH}" ".backup '${DB_DST}'"; then
        DB_BYTES="$(stat -c '%s' "${DB_DST}" 2>/dev/null || echo 0)"
    else
        mark_corrupt "sqlite3 .backup failed"
        die 13 "sqlite3 .backup failed for ${DB_PATH}"
    fi

    # Integrity check on the COPY (don't lock the live DB further).
    log "Running integrity_check on snapshot DB..."
    INTEGRITY="$(sqlite3 "${DB_DST}" 'pragma integrity_check;' 2>&1)"
    if [[ "${INTEGRITY}" != "ok" ]]; then
        mark_corrupt "integrity_check: ${INTEGRITY}"
        die 16 "DB integrity check failed: ${INTEGRITY}"
    fi

    # SHA-256 -- written next to the DB so manual recovery can verify.
    (cd "${SNAPSHOT_DIR}/data" && sha256sum images.db > images.db.sha256) \
        || die 13 "sha256sum failed on snapshot DB"
else
    log "WARN: ${DB_PATH} not present -- skipping DB backup."
fi

# ----------------------------------------------------------------------
# Step 7: Imagery + per-output state (rsync with --link-dest dedup)
# ----------------------------------------------------------------------
# We snapshot the entire OUTPUT_DIR (images, settings.yaml, model
# downloads metadata, ingest state) EXCEPT the live DB itself (already
# captured above) and any large transient caches.
log "Syncing imagery + output state via rsync..."
RSYNC_LINKDEST=()
if [[ -n "${PREV_SNAPSHOT}" && -d "${PREV_SNAPSHOT}/data/output" ]]; then
    RSYNC_LINKDEST=(--link-dest="${PREV_SNAPSHOT}/data/output")
fi

mkdir -p "${SNAPSHOT_DIR}/data/output"
# Excludes:
#   - images.db / -wal / -shm: captured via sqlite .backup (consistent)
#   - backup/: live app's transient backup staging dir
#   - restore_tmp/: live app's restore staging
#   - .restart_required: marker file, not data
if ! rsync -a --delete \
        --exclude='images.db' --exclude='images.db-wal' --exclude='images.db-shm' \
        --exclude='backup/' \
        --exclude='restore_tmp/' \
        --exclude='backup_before_restore/' \
        --exclude='.restart_required' \
        "${RSYNC_LINKDEST[@]}" \
        "${OUTPUT_DIR}/" \
        "${SNAPSHOT_DIR}/data/output/"; then
    die 14 "rsync of imagery (${OUTPUT_DIR}) failed"
fi

# ----------------------------------------------------------------------
# Step 8: App code (rsync, link-dest dedup)
# ----------------------------------------------------------------------
log "Syncing app code..."
APP_LINKDEST=()
if [[ -n "${PREV_SNAPSHOT}" && -d "${PREV_SNAPSHOT}/app" ]]; then
    APP_LINKDEST=(--link-dest="${PREV_SNAPSHOT}/app")
fi

# Exclude data/ (already captured via output snapshot) and .venv (rebuilt
# on every release; copying it bloats the stick by ~500 MB per snapshot).
if ! rsync -a --delete \
        --exclude='data/' \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='node_modules/' \
        "${APP_LINKDEST[@]}" \
        "${APP_DIR}/" \
        "${SNAPSHOT_DIR}/app/"; then
    die 15 "rsync of app code (${APP_DIR}) failed"
fi

# ----------------------------------------------------------------------
# Step 9: Manifest
# ----------------------------------------------------------------------
APP_VERSION_FILE="${APP_DIR}/APP_VERSION"
APP_VERSION="unknown"
if [[ -f "${APP_VERSION_FILE}" ]]; then
    APP_VERSION="$(tr -d '[:space:]' < "${APP_VERSION_FILE}")"
fi

json_escape() {
    local s="${1:-}"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "${s}"
}

HOSTNAME_JSON="$(json_escape "$(hostname)")"
USER_JSON="$(json_escape "$(id -un)")"
APP_VERSION_JSON="$(json_escape "${APP_VERSION}")"
PREV_SNAPSHOT_JSON="$(json_escape "${PREV_SNAPSHOT##*/snapshots/}")"

OUTPUT_BYTES="$(du -sb "${SNAPSHOT_DIR}/data/output" 2>/dev/null | awk '{print $1}')"
APP_BYTES="$(du -sb "${SNAPSHOT_DIR}/app" 2>/dev/null | awk '{print $1}')"
TOTAL_BYTES="$(du -sb "${SNAPSHOT_DIR}" 2>/dev/null | awk '{print $1}')"
TS_END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "${SNAPSHOT_DIR}/manifest.json" <<EOF
{
  "schema_version": 1,
  "snapshot_name": "${SNAPSHOT_NAME}",
  "kind": "${KIND}",
  "started_at": "${TS_START}",
  "completed_at": "${TS_END}",
  "app_version": "${APP_VERSION_JSON}",
  "host": "${HOSTNAME_JSON}",
  "user": "${USER_JSON}",
  "previous_snapshot": "${PREV_SNAPSHOT_JSON}",
  "sizes": {
    "db_bytes": ${DB_BYTES},
    "output_bytes": ${OUTPUT_BYTES:-0},
    "app_bytes": ${APP_BYTES:-0},
    "total_bytes": ${TOTAL_BYTES:-0}
  }
}
EOF

# ----------------------------------------------------------------------
# Step 10: COMPLETED marker (atomic-ish: this is the LAST file written)
# ----------------------------------------------------------------------
echo "${TS_END}" > "${SNAPSHOT_DIR}/COMPLETED"

# Update 'latest' symlink to point at this snapshot.
ln -sfn "${SNAPSHOT_DIR}" "${LATEST_LINK}.tmp"
mv -Tf "${LATEST_LINK}.tmp" "${LATEST_LINK}"

log "Snapshot ${SNAPSHOT_NAME} completed (${TOTAL_BYTES:-0} B total)."

# ----------------------------------------------------------------------
# Step 11: Retention
# ----------------------------------------------------------------------
# We compute retention IN-PROCESS rather than relying on systemd or cron
# helpers, because we want kind-aware policies:
#   scheduled -> 7 daily, 4 weekly (Mondays), 6 monthly (1st of month)
#   manual    -> last 3
# CORRUPT snapshots are NEVER auto-deleted -- the operator decides.

prune_manual() {
    local keep=3
    local count=0
    while IFS= read -r dir; do
        if [[ "$(cat "${dir}/kind" 2>/dev/null)" == "manual" ]] \
            && [[ -f "${dir}/COMPLETED" ]]; then
            count=$((count + 1))
            if (( count > keep )); then
                log "Retention(manual): pruning ${dir##*/snapshots/}"
                rm -rf "${dir}"
            fi
        fi
    done < <(find "${SNAPSHOTS_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)
}

prune_scheduled() {
    # Strategy: build a "keep set" of stamps, then delete everything
    # scheduled+COMPLETED that's not in it.
    declare -A KEEP
    local count_daily=0
    local count_weekly=0
    local count_monthly=0
    local last_week=""
    local last_month=""

    while IFS= read -r dir; do
        local name="${dir##*/}"
        [[ "${name}" =~ ^([0-9]{8})_([0-9]{6})_scheduled$ ]] || continue
        local datepart="${BASH_REMATCH[1]}"
        [[ -f "${dir}/COMPLETED" ]] || continue

        # 7 most recent daily
        if (( count_daily < 7 )); then
            KEEP["${name}"]=1
            count_daily=$((count_daily + 1))
        fi

        # 4 most recent weekly (one per ISO week, newest in that week wins)
        local iso_week
        iso_week="$(date -d "${datepart:0:4}-${datepart:4:2}-${datepart:6:2}" +%G-W%V 2>/dev/null || echo "")"
        if [[ -n "${iso_week}" && "${iso_week}" != "${last_week}" && "${count_weekly}" -lt 4 ]]; then
            KEEP["${name}"]=1
            last_week="${iso_week}"
            count_weekly=$((count_weekly + 1))
        fi

        # 6 most recent monthly
        local ym="${datepart:0:6}"
        if [[ "${ym}" != "${last_month}" && "${count_monthly}" -lt 6 ]]; then
            KEEP["${name}"]=1
            last_month="${ym}"
            count_monthly=$((count_monthly + 1))
        fi
    done < <(find "${SNAPSHOTS_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)

    # Delete everything not in KEEP.
    while IFS= read -r dir; do
        local name="${dir##*/}"
        [[ "${name}" =~ ^[0-9]{8}_[0-9]{6}_scheduled$ ]] || continue
        [[ -f "${dir}/COMPLETED" ]] || continue
        if [[ -z "${KEEP[${name}]:-}" ]]; then
            log "Retention(scheduled): pruning ${name}"
            rm -rf "${dir}"
        fi
    done < <(find "${SNAPSHOTS_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)
}

prune_manual
prune_scheduled

# Truncate BACKUP_LOG.txt to last ~30 days. Awk keeps lines whose embedded
# ISO timestamp is within the window.
if [[ -f "${LOG_FILE}" ]]; then
    cutoff="$(date -u -d '30 days ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
    if [[ -n "${cutoff}" ]]; then
        tmp="$(mktemp "${LOG_FILE}.XXXXXX" 2>/dev/null || true)"
        if [[ -n "${tmp}" ]]; then
            awk -v c="${cutoff}" '
                match($0, /\[[0-9-]+T[0-9:]+Z\]/) {
                    ts = substr($0, RSTART+1, RLENGTH-2);
                    if (ts >= c) print;
                    next;
                }
                { print }
            ' "${LOG_FILE}" > "${tmp}" && mv -f "${tmp}" "${LOG_FILE}"
        fi
    fi
fi

# Best-effort: flush dirty buffers so a yanked stick keeps what we wrote.
sync

log "Done."
exit 0
