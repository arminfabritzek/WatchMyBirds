# USB Backup

WatchMyBirds writes daily snapshots of the SQLite database, captured frames,
and the installed app code to a USB stick mounted at `/mnt/wmb-backup`. This
is your protection against SD-card death, which is the single most common
hardware failure on a long-running Raspberry Pi.

Recovery and migration are a supported, scripted CLI command
(`scripts/recover_from_snapshot.py`) that runs directly against a snapshot
directory — no archive step, no in-app restore UI. See *Recovery and
migration* below.

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

`scripts/recover_from_snapshot.py` is the supported recovery/migration
command. It ships in the repository (not an agent-only tool) and runs
directly against a snapshot directory — no archive upload, no
intermediate `.tar.gz`, no in-app restore UI.

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
    --mode migration
```

**Replacing a populated installation (recovery mode)** — a deliberate,
explicit action. Requires `--force`, and always takes a safety checkpoint
of the current database into `<destination>/backup_before_restore/`
before touching anything (the checkpoint path is printed on completion):

```bash
.venv/bin/python scripts/recover_from_snapshot.py \
    --snapshot /mnt/wmb-backup/snapshots/20260901_030000_scheduled \
    --destination /opt/app/data/output \
    --mode recovery --force
```

Then restart the app:

```bash
sudo systemctl start app.service       # Raspberry Pi
docker compose start app               # Docker
```

The command validates the snapshot (COMPLETED marker, checksum,
`integrity_check`, media directory present) before writing anything, and
refuses a corrupt or incomplete snapshot outright. Pass `--skip-media` to
restore the database only (useful for a quick metadata-only recovery
check). It never restores the `app/` tree — app code always comes from
the running release image; rolling back app code is OTA's job, not
backup's, when OTA rollback ships.

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
