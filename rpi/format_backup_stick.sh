#!/usr/bin/env bash
# ----------------------------------------------------------------------
# WatchMyBirds USB Format (one-click stick prep from the Pi UI)
#
# DESTRUCTIVE. Wipes the requested device and creates a single ext4
# partition labelled WMB-BACKUP, mounted at /mnt/wmb-backup.
#
# Invoked by wmb-format-backup.service as root. The web UI passes the
# target device (e.g. /dev/sda) via env var WMB_TARGET_DEV, and the
# expected confirmation string via WMB_CONFIRM. Both are validated.
#
# Hard guards (refuse to run unless ALL hold):
#   1. WMB_TARGET_DEV is set and matches /dev/sd[a-z] only (no sda1!)
#   2. WMB_TARGET_DEV is NOT /dev/mmcblk* or /dev/nvme*
#   3. WMB_CONFIRM equals "FORMAT" (typed by operator in UI)
#   4. Target's ID_BUS is exactly "usb" (not ata, not scsi)
#   5. Target size is in [4 GB, 2 TB] -- catches both micro-sticks
#      and "I attached the wrong drive" mistakes
#   6. Target is NOT currently mounted (no partition of it)
#   7. /opt/app/data is NOT on the same device
#   8. Target reports as a removable device via udev
#
# Exit codes:
#    0 -- success, stick formatted and mounted
#    2 -- usage / bad arguments
#   10 -- guard rejected the device (wrong type / size / mounted / etc.)
#   11 -- format step failed (mkfs / parted)
#   12 -- mount step failed
# ----------------------------------------------------------------------

set -euo pipefail

readonly LABEL="WMB-BACKUP"
readonly MOUNT_POINT="/mnt/wmb-backup"
readonly EXPECTED_CONFIRM="FORMAT"
# Min/max device size, bytes. 4 GB lower (no useful sticks below);
# 2 TB upper (catches accidentally-attached drives).
readonly MIN_BYTES=$((4 * 1024 * 1024 * 1024))
readonly MAX_BYTES=$((2 * 1024 * 1024 * 1024 * 1024))

# Status file the API endpoint reads while polling.
readonly STATUS_FILE="/opt/app/data/usb_format_status.json"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

write_status() {
    # Args: state, message
    local state="$1"
    local msg="$2"
    mkdir -p "$(dirname "${STATUS_FILE}")"
    # JSON-escape the message minimally (only quotes and backslashes).
    local escaped="${msg//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    cat > "${STATUS_FILE}" <<EOF
{
  "state": "${state}",
  "message": "${escaped}",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "target": "${WMB_TARGET_DEV:-}"
}
EOF
    # Make readable by the app user.
    chown watchmybirds:watchmybirds "${STATUS_FILE}" 2>/dev/null || true
    chmod 644 "${STATUS_FILE}" 2>/dev/null || true
}

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

die() {
    local code="$1"; shift
    local msg="$*"
    log "FATAL($code): ${msg}"
    write_status "error" "${msg}"
    exit "$code"
}

# ----------------------------------------------------------------------
# Step 1: Validate inputs
# ----------------------------------------------------------------------

write_status "starting" "Validating request..."

if [[ -z "${WMB_TARGET_DEV:-}" ]]; then
    die 2 "WMB_TARGET_DEV not set"
fi

if [[ "${WMB_CONFIRM:-}" != "${EXPECTED_CONFIRM}" ]]; then
    die 2 "WMB_CONFIRM does not match expected value"
fi

# Allowed device pattern: /dev/sd[a-z] -- no partitions, no other prefixes.
if ! [[ "${WMB_TARGET_DEV}" =~ ^/dev/sd[a-z]$ ]]; then
    die 10 "Target must match /dev/sd[a-z] (got: ${WMB_TARGET_DEV})"
fi

# Negative-list check (defense in depth, even though regex above is strict).
case "${WMB_TARGET_DEV}" in
    /dev/mmcblk*|/dev/nvme*|/dev/loop*|/dev/ram*|/dev/zram*)
        die 10 "Refusing to operate on system device: ${WMB_TARGET_DEV}"
        ;;
esac

if [[ ! -b "${WMB_TARGET_DEV}" ]]; then
    die 10 "${WMB_TARGET_DEV} is not a block device"
fi

# ----------------------------------------------------------------------
# Step 2: Verify it's a real USB stick
# ----------------------------------------------------------------------

write_status "validating" "Checking device type..."

bus="$(udevadm info --query=property --name="${WMB_TARGET_DEV}" 2>/dev/null \
       | awk -F= '$1=="ID_BUS" {print $2}' || true)"
if [[ "${bus}" != "usb" ]]; then
    die 10 "Device bus is '${bus:-unknown}', expected 'usb'. Refusing to format internal storage."
fi

# Removable-flag check (extra safety).
removable="$(cat /sys/block/$(basename "${WMB_TARGET_DEV}")/removable 2>/dev/null || echo 0)"
if [[ "${removable}" != "1" ]]; then
    die 10 "Device is not flagged as removable. Refusing."
fi

# ----------------------------------------------------------------------
# Step 3: Size sanity
# ----------------------------------------------------------------------

size_bytes="$(lsblk -bdno SIZE "${WMB_TARGET_DEV}" 2>/dev/null || echo 0)"
if (( size_bytes < MIN_BYTES )); then
    die 10 "Device too small: ${size_bytes} bytes (min ${MIN_BYTES})"
fi
if (( size_bytes > MAX_BYTES )); then
    die 10 "Device too large: ${size_bytes} bytes (max ${MAX_BYTES}). Looks like an internal drive."
fi
size_gb=$((size_bytes / 1024 / 1024 / 1024))
log "Device ${WMB_TARGET_DEV}: ${size_gb} GB, USB, removable. OK to format."

# ----------------------------------------------------------------------
# Step 4: Refuse if any partition of target is mounted
# ----------------------------------------------------------------------

if mount | grep -qE "^${WMB_TARGET_DEV}[0-9]* "; then
    die 10 "A partition of ${WMB_TARGET_DEV} is currently mounted. Unmount first."
fi

# Refuse if /opt/app/data ends up on the same device (catastrophic).
data_src="$(df -P /opt/app/data 2>/dev/null | awk 'NR==2 {print $1}' || true)"
if [[ -n "${data_src}" && "${data_src}" == "${WMB_TARGET_DEV}"* ]]; then
    die 10 "App data is on ${data_src}, same device as target. Refusing."
fi

# ----------------------------------------------------------------------
# Step 5: Wipe + new GPT + ext4 + label
# ----------------------------------------------------------------------

write_status "wiping" "Wiping existing partitions..."
wipefs -af "${WMB_TARGET_DEV}" || die 11 "wipefs failed"

write_status "partitioning" "Creating new GPT partition table..."
parted --script "${WMB_TARGET_DEV}" \
    mklabel gpt \
    mkpart primary ext4 1MiB 100% \
    || die 11 "parted failed"

# Wait for /dev/sdX1 to materialize via udev.
partprobe "${WMB_TARGET_DEV}" || true
udevadm settle --timeout=10 || true
sleep 2

if [[ ! -b "${WMB_TARGET_DEV}1" ]]; then
    die 11 "${WMB_TARGET_DEV}1 did not appear after partprobe"
fi

write_status "formatting" "Formatting as ext4 with label ${LABEL}..."
mkfs.ext4 -F -L "${LABEL}" -m 0 \
    -E lazy_itable_init=0,lazy_journal_init=0 \
    "${WMB_TARGET_DEV}1" \
    || die 11 "mkfs.ext4 failed"

# Wait for udev to register the new label.
udevadm settle --timeout=10 || true
sleep 1

# ----------------------------------------------------------------------
# Step 6: Mount + chown to watchmybirds
# ----------------------------------------------------------------------

write_status "mounting" "Mounting at ${MOUNT_POINT}..."
mkdir -p "${MOUNT_POINT}"

# If something is already mounted there (autofs stub or stale mount), unmount.
if mountpoint -q "${MOUNT_POINT}"; then
    umount "${MOUNT_POINT}" 2>/dev/null || true
fi

# Trigger the systemd automount unit by accessing the path -- this picks
# up the freshly-labelled device and mounts it via the existing
# mnt-wmb\x2dbackup.mount configuration.
ls "${MOUNT_POINT}" >/dev/null 2>&1 || true
sleep 2

if ! mountpoint -q "${MOUNT_POINT}"; then
    # Automount didn't trigger; try direct mount.
    mount "/dev/disk/by-label/${LABEL}" "${MOUNT_POINT}" \
        || die 12 "Mount failed even with direct invocation"
fi

# Hand on-disk ownership to the app user (ext4 ignores uid= mount option).
chown -R watchmybirds:watchmybirds "${MOUNT_POINT}"
chmod 755 "${MOUNT_POINT}"
sudo -u watchmybirds mkdir -p "${MOUNT_POINT}/snapshots"

# ----------------------------------------------------------------------
# Step 7: Done
# ----------------------------------------------------------------------

write_status "success" "Stick formatted and mounted at ${MOUNT_POINT}."
log "Format complete: ${WMB_TARGET_DEV} -> ${LABEL} on ${MOUNT_POINT}"
exit 0
