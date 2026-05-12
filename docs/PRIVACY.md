# Privacy

WatchMyBirds runs entirely on your Raspberry Pi. Nothing about your bird
activity, images, or detections ever leaves the device.

There is exactly **one** piece of optional outbound traffic: an anonymous
daily heartbeat that lets us count active installations. **It is off by
default**, and the only way to turn it on is the toggle in
**Settings → Privacy** in your running install.

This document mirrors the `/privacy` page in the running app. If they ever
disagree, the code in `web/services/telemetry_service.py` and
`infra/telemetry-worker/src/worker.js` is authoritative — please file a bug.

---

## What we send (only when you opt in)

One small JSON payload, once per UTC day, to
`https://heartbeat-wmb.starmin.de/v1/heartbeat`:

**Schema:**

```json
{
  "installation_id":   "(32 random hex chars, generated once when you opt in)",
  "app_version":       "v0.X.Y",
  "os":                "linux | darwin | windows",
  "arch":              "aarch64 | x86_64 | armv7l",
  "cpu_count":         <integer>,
  "total_ram_gb":      <integer, rounded to whole GB>,
  "python_version":    "3.x.y",
  "detector_variant":  "yolox-tiny-int8 | fasterrcnn | unknown"
}
```

**Example** — what a Raspberry Pi 5 (8 GB) actually sends:

```json
{
  "installation_id":   "a3f2c81d9b4e47f6a0c1d8e2b5f93a7c",
  "app_version":       "v0.2.10",
  "os":                "linux",
  "arch":              "aarch64",
  "cpu_count":         4,
  "total_ram_gb":      8,
  "python_version":    "3.12.3",
  "detector_variant":  "yolox-tiny-int8"
}
```

…and what an Apple Silicon dev machine (M1, 16 GB) sends:

```json
{
  "installation_id":   "a3f2c81d9b4e47f6a0c1d8e2b5f93a7c",
  "app_version":       "v0.2.10",
  "os":                "darwin",
  "arch":              "aarch64",
  "cpu_count":         8,
  "total_ram_gb":      16,
  "python_version":    "3.12.3",
  "detector_variant":  "fasterrcnn"
}
```

That's it. Eight fields, all anonymous. The `installation_id` is a random
number we generate on your device — it is **not** derived from your
hardware, MAC, hostname, or anything else identifiable.

## What we explicitly never send

- Your IP address, country, region, or any geo-location
- Your locale, language, or timezone
- Your hostname, MAC address, or any hardware serial
- The Raspberry Pi model string or kernel version
- Exact RAM in bytes (we round to whole GB on purpose)
- Any image, video, or audio data
- Any species names, observation counts, or detection events
- Error messages, stack traces, or debug logs
- Camera URLs, settings, passwords, or anything from your config
- Your email or any identifier from your operating system

## Where the data lives

The heartbeat is received by a tiny Cloudflare Worker and stored in a
Cloudflare D1 database with `jurisdiction=eu`, meaning Cloudflare guarantees
the data is stored and processed only in the European Union. The endpoint is
`https://heartbeat-wmb.starmin.de/v1/heartbeat`.

The Worker explicitly drops the IP address, country code, and all
Cloudflare-injected location metadata before writing to the database. The
Worker source code is open and reviewable in
[`infra/telemetry-worker/`](../infra/telemetry-worker/).

## How long we keep it

Each individual heartbeat is deleted **within 24 hours** of being received.
A nightly cron at 04:30 UTC aggregates yesterday's heartbeats into per-day,
per-cohort counts (e.g. "1 install on `aarch64` with 8 GB RAM running
v0.2.10 on 2026-05-06"), then deletes the raw rows.

The aggregate table has **no `installation_id`**. There is no way to track
an individual install across days from what we keep. We never archive,
export, or back up the raw heartbeats — once they are aggregated, the
per-install timeline is gone forever.

## Your controls

- **Off (default)** — nothing is sent. Ever. The first time you toggle
  telemetry on, we generate a random `installation_id` and store it in
  `settings.yaml` on your device. Until then, no ID exists.

- **Toggle off later** — pings stop immediately. Your `installation_id`
  stays the same, so if you turn it back on, you're counted as the same
  install (not a new one).

- **Rotate ID** — if you want the next opt-in to be counted as a fresh
  install, click the **Rotate ID** button in Settings → Privacy. This wipes
  your current ID and generates a new one. Old raw rows in the cloud are
  aggregated and deleted within 24 hours of being received, so a rotation
  has effect very quickly.

- **Block at the firewall** — the heartbeat hostname is deliberately
  separate from any other WatchMyBirds endpoint, so you can firewall-block
  `heartbeat-wmb.starmin.de` without breaking anything else in the app.

- **Override the endpoint** — set `telemetry_endpoint` in `settings.yaml` to
  point at any URL you control (or `http://localhost/discard`). The
  toggle's "on" state then sends to wherever you said.

## Why we ask at all

WatchMyBirds is a small open-source project maintained by one person.
GitHub stars and clones don't tell us if anyone actually runs the app —
only if they bookmarked it. A daily heartbeat lets us see whether the
project is being used, which informs decisions about what to fix, what to
build next, and whether continuing is worth the time. That's the entire
purpose of this feature.

## Questions or concerns

If you spot something wrong, want clarification, or want to request
deletion of any data: open an issue on
[GitHub](https://github.com/arminfabritzek/WatchMyBirds/issues).

Because the data is anonymous, we can't selectively delete a specific
install's history. Use the **Rotate ID** button or wait 90 days for the
existing rows to self-delete.

---

This page describes the behavior of the heartbeat as of the current app
version. The Worker source code in
[`infra/telemetry-worker/src/worker.js`](../infra/telemetry-worker/src/worker.js)
and the client code in
[`web/services/telemetry_service.py`](../web/services/telemetry_service.py)
are the authoritative reference. If they ever disagree with this page, the
code wins — please file a bug.
