# USB Backup

WatchMyBirds writes daily snapshots of the SQLite database, captured frames,
and the installed app code to a USB stick mounted at `/mnt/wmb-backup`. This
is your protection against SD-card death, which is the single most common
hardware failure on a long-running Raspberry Pi.

Raspberry Pi appliances provide guided recovery in **Settings → Data &
Backups**. The CLI (`scripts/recover_from_snapshot.py`) uses the same recovery
engine and remains available as a technical fallback or for supported data
migration on another CPU architecture.

## What you need

- Any USB stick **at least 2× the size of your SD card** (rule of thumb:
  daily snapshots use hardlinks for unchanged frames, so total stick usage
  is roughly *live data + 1 full copy + a few % drift per day*).
- A Linux/macOS machine to format the stick. (Windows can't write ext4
  natively; use a Live USB or WSL.)

## One-time setup

> **Warning:** the format step **destroys all data on the stick**. Make
> sure you've picked the right device.

1. Plug the stick into your Mac/Linux box (not the Pi yet) and find its
   device node:

   ```bash
   lsblk           # Linux
   diskutil list   # macOS
   ```

   Look for the new device, e.g. `/dev/sdb`. **Use the whole-disk node**
   (`/dev/sdb`), not a partition (`/dev/sdb1`), unless you already have a
   partition layout you want to keep.

2. Format with ext4 and the label `WMB-BACKUP`:

   ```bash
   # Linux
   sudo mkfs.ext4 -L WMB-BACKUP /dev/sdb
   ```

   The label is **mandatory** — the Pi mounts the stick by label, not by
   device path, so it survives re-plugging into a different USB port.

3. Eject cleanly, plug into the Pi.

That's it. The systemd automount unit picks it up on first access, the
daily timer runs at 03:00, and the Settings UI surfaces the stick state.

The Hugging Face model-download cache is excluded from new snapshots and ignored
when restoring older snapshots. Models are obtained by the installed application.

## Backup progress and errors

Settings shows the current backup stage and refreshes automatically: preparation,
image copy, database copy, recent-image copy, application copy, verification,
and finishing. These are stage updates, not a percentage or time estimate.
The storage bar shows USB capacity, not backup progress.

The final result replaces the running message. If the recorded process has gone
away without writing a result, the page reports an interrupted backup. Copy
errors are retained in `BACKUP_LOG.txt` on the stick; manual backup output also
appears in the application journal. Do not remove the source SD card until a
completed snapshot has passed verification.

## Why ext4?

The backup script uses `rsync --link-dest` so daily snapshots share
hardlinks for unchanged frames. **Without hardlinks, every snapshot would
be a full copy** — the stick fills up in days instead of months. FAT32 and
exFAT don't support hardlinks. NTFS does, but its Linux driver is too slow
and fragile for unattended overnight runs.

The mount is also locked down: `nosuid,nodev,noexec` means even if someone
plants an executable on the stick, the Pi refuses to run it.

## Recovery and migration

Snapshot directories are named `YYYYMMDD_HHMMSS_<kind>/` under
`/mnt/wmb-backup/snapshots/` (`<kind>` is `scheduled` or `manual`); the
newest valid one is also reachable via the `/mnt/wmb-backup/latest`
symlink. Each contains:

- `data/images.db` — SQLite database (verified with `pragma
  integrity_check` at capture time; re-verify with `sha256sum -c
  images.db.sha256` before trusting an old snapshot)
- `data/output/` — captured frames and per-output app state
- `app/` — the installed app code at capture time (forensic only; app
  code is never restored by the recovery command — see below)
- `manifest.json` — what was captured, sizes, integrity hashes
- `COMPLETED` — marker file. **Trust no snapshot directory that lacks
  this file** — it crashed mid-write.

### Guided Raspberry Pi recovery

On a replacement SD card, finish the local password setup, connect the
existing `WMB-BACKUP` stick, then open **Settings → Data & Backups**. The page
lists completed snapshots and shows **We found a backup. Restore your birds?**
Choose **Review & restore** to see the date, source device, image/detection/
classification counts, app compatibility, required and available space, and
the settings policy. Nothing is restored merely because a stick is connected.

On a populated installation the same action clearly says that Restore replaces
current data. It is deliberately separate from archive **Merge**, which combines
collections. Two confirmations are required before the request is submitted.

The dedicated `wmb-recovery.service` then:

1. revalidates the fixed snapshot identifier and destination;
2. takes the shared maintenance lock and stops `app.service`;
3. validates, stages, and verifies the recovered database and retained media;
4. retains the complete previous output directory as a checkpoint;
5. atomically publishes the recovered directory; and
6. restarts WatchMyBirds and waits for `/healthz`.

The browser moves to a small independent, token-protected status page on port
8051 before the app stops. If app startup fails, that page remains usable and
offers **Retry app startup** and, when a checkpoint exists, **Restore previous
checkpoint**. This avoids both a permanent spinner and an SSH-only failure path.

Current Wi-Fi, operating-system network configuration, SSH keys, and installed
application binaries are never restored. Backup runtime behavior is restored,
but the destination's admin password, camera/relay connection values, Telegram
credentials, and telemetry installation identity are preserved when they
already exist. Source values in those protected fields are ignored when the
destination has no value. Destination `cameras.yaml` and `go2rtc.yaml` are kept;
on a fresh destination the source device files are omitted. This keeps browser
access and device identity usable after both fresh migration and replacement
recovery without importing another device's credentials.

Guided orchestration is only claimed for Raspberry Pi images that install the
runner, polkit rule, state directories, and firewall rule. Docker guided
recovery is not implemented because a stopped container cannot safely own its
host/container restart. Data migration through the shared CLI remains supported
across CPU architectures; no binaries or virtualenv files are restored.

### Technical CLI fallback

`scripts/recover_from_snapshot.py` ships in the repository and runs directly
against a snapshot directory without an archive upload. It calls the same core
validation, staging, checkpoint, publication, and interruption-repair code as
the guided runner.

**It never starts or stops any service.** Stop the app first so nothing
holds the database open while it is replaced:

```bash
# Raspberry Pi
sudo systemctl stop app.service

# Docker
docker compose stop app
```

**Fresh install / SD-card died (migration mode, default)** — refuses if
the destination database already has rows:

```bash
.venv/bin/python scripts/recover_from_snapshot.py \
    --snapshot /mnt/wmb-backup/latest \
    --destination /opt/app/data/output \
    --mode migration --app-stopped
```

**Replacing a populated installation (recovery mode)** — a deliberate,
explicit action. Requires `--force` and `--app-stopped`. The command stages
and verifies all output data before replacing the destination. The complete
previous output directory is retained beside the destination as
`output-before-restore-<unique-id>`; its path is printed before publication.
A publication error restores that directory automatically.

```bash
.venv/bin/python scripts/recover_from_snapshot.py \
    --snapshot /mnt/wmb-backup/snapshots/20260901_030000_scheduled \
    --destination /opt/app/data/output \
    --mode recovery --force --app-stopped
```

Then restart the app:

```bash
sudo systemctl start app.service       # Raspberry Pi
docker compose start app               # Docker
```

The command validates the manifest version, COMPLETED marker, database checksum,
SQLite integrity, and expected originals before writing the destination. Available
original checksums are verified; older rows without hashes receive existence
checks only. Retention-deleted originals are allowed to be absent. Per-output
state is restored subject to the destination identity/credential policy above.
App code is not restored.

Run as the app user, or as root against an existing destination owned by that user.
Both modes require all app processes to be stopped and `--app-stopped` to acknowledge
that precondition. The command does not independently detect all open connections.
For Docker, operate on the stopped container's **host output directory**, not an
active bind mount inside a container. Staging requires space for the incoming
output alongside the retained old directory. A copy failure leaves the destination
untouched. An atomic journal beside the output directory records staging,
checkpoint, and publication phases. A later run removes abandoned staging,
restores the complete checkpoint if interruption happened between directory
renames, or recognizes that publication already finished. Do not remove a
retained checkpoint until recovery and application startup are verified.
Journal contents, staged files and directories, and the parent directory after
each rename are flushed before the next phase is recorded. Both the recovery
unit and the main app run reconciliation synchronously before app startup.

USB backup performs a second file-copy pass after its database snapshot and checks
that snapshot's expected media before marking it complete. Concurrent changes may
cause a run to fail verification; copying order alone does not provide an atomic
snapshot. TERM/INT failures are recorded when the USB volume remains writable;
power loss, SIGKILL, or a disconnected disk cannot guarantee a final status record.

Older snapshots captured before this command existed use the same
directory layout and work with it unchanged — there is nothing to
regenerate.

## What is NOT backed up

- Audio recordings (audio is currently archived as a feature; will be
  added separately if/when audio returns to mainline)
- `/opt/app/.venv/` — pip-rebuilt on every release, no point copying
- System config (`/etc/`, network settings, SSH keys, wifi credentials)
  — these are baked into the image, not the backup
- Encrypted secrets at rest — the stick is plain ext4. If your threat
  model includes "someone steals the stick", encrypt at the volume level
  yourself (LUKS) before formatting; the mount unit accepts any ext4
  volume regardless of underlying encryption.

## Status & troubleshooting

The Settings page surfaces:

- **Stick connected / missing** — automount only mounts `WMB-BACKUP`-labelled
  ext4 volumes; anything else is reported as `wrong-fs` with format
  instructions.
- **Free space + warning at >80% full** — old snapshots beyond the
  retention window are pruned automatically (7 daily, 4 weekly, 6 monthly
  for scheduled; latest 3 for manual triggers).
- **Last 5 snapshots** — with completion state and verification hash.

If the Pi reports `wrong-fs` repeatedly, the most common cause is a stick
that came pre-formatted as exFAT or FAT32 from the factory. Reformat as
above.

If the stick keeps showing as missing after replugging, run on the Pi:

```bash
journalctl -u 'mnt-wmb*backup*' --since '10 min ago'
```

This is also the right thing to attach when reporting a backup-related
issue.
