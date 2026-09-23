# Privacy

WatchMyBirds runs on your Raspberry Pi. Bird activity, images, and detections
stay there unless you explicitly enable an outbound integration such as
Telegram or MQTT. MQTT sends detection metadata and an image URL to the broker
you configure; a consumer that can reach WatchMyBirds can fetch that image.

The separate optional traffic to the WatchMyBirds operator is a voluntary
daily heartbeat that helps us understand installation activity and software adoption. **It is off by
default**, and you can enable it during initial password setup or in
**Settings → Privacy** in your running install.

This document mirrors the `/privacy` page in the running app. If they ever
disagree, please report the discrepancy. The collection code is in `web/services/telemetry_service.py` and
`infra/telemetry-worker/src/worker.js`.

---

## What we send (only when you opt in)

One small JSON payload, once per UTC day, to
`https://watchmybirds-telemetry.wmb-infra.workers.dev/v1/heartbeat`:

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

These eight fields describe an installation, not a named user. The `installation_id` is a random
number we generate on your device — it is **not** derived from your
hardware, MAC, or hostname. The stable ID lets us recognize repeat
reports from the same installation; the data is pseudonymous, not fully anonymous.

## What the heartbeat does not include

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

## Training-data exports

The Export page offers a bundle of your reviewed birds — a ZIP built on this
device, containing selected images and the labels you confirmed. It stays
here. WatchMyBirds never sends it anywhere; only you can do that, by hand.

Sharing it is entirely optional and nothing in the app depends on it. The
page invites you to pass your corrections on, because that is how the models
everyone runs got better in the first place. Ignoring the invitation changes
nothing.

If you do want to share, please don't attach the archive to a public issue —
it contains your own images. Get in touch without it and we'll arrange a
private transfer.

## Where the data lives

The heartbeat is received by a tiny Cloudflare Worker and stored in a
Cloudflare D1 database with `jurisdiction=eu`, configured for EU data jurisdiction. The endpoint is
`https://watchmybirds-telemetry.wmb-infra.workers.dev/v1/heartbeat`.

The Worker explicitly drops the IP address, country code, and all
Cloudflare-injected location metadata before writing to the database. The
Worker source code is open and reviewable in
[`infra/telemetry-worker/`](../infra/telemetry-worker/).

The connection exposes its source IP to Cloudflare as the network provider;
we do not include it in the heartbeat payload or store it in our telemetry database.

## Storage and retention

Cloudflare's scheduled daily job at 04:30 UTC aggregates the previous day's
heartbeats and removes raw rows from earlier dates. This is a daily cleanup,
not a guaranteed 24-hour deletion deadline.

The maintainer also keeps a local history of installation IDs, report dates,
versions, and the technical fields listed above. This history is used to
count returning installations, follow version adoption, and prioritize
compatibility work. It currently has no automatic expiry. Aggregated counts
contain no installation IDs and are also retained for trend analysis.

## Your controls

- **Off (default)** — nothing is sent. Ever. The first time you toggle
  telemetry on, we generate a random `installation_id` and store it in
  `settings.yaml` on your device. Until then, no ID exists.

- **Toggle off later** — pings stop immediately. Your `installation_id`
  stays the same, so if you turn it back on, you're counted as the same
  install (not a new one).

- **Rotate ID** — if you want the next opt-in to be counted as a fresh
  install, click the **Rotate ID** button in Settings → Privacy. This wipes
  your current ID and generates a new one. Future reports use the new ID. Rotation does not delete
  previously received reports or the local history.

- **Block at the firewall** — the heartbeat hostname is deliberately
  separate from any other WatchMyBirds endpoint, so you can firewall-block
  `watchmybirds-telemetry.wmb-infra.workers.dev` without breaking anything else in the app.

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

For a request concerning an installation, we may need its installation ID
from your device to locate the relevant records. Do not post it publicly;
ask for a private way to share it. Turning telemetry off stops future
reports but does not automatically delete previously received data.

---

This page describes the behavior of the heartbeat as of the current app
version. The Worker source code in
[`infra/telemetry-worker/src/worker.js`](../infra/telemetry-worker/src/worker.js)
and the client code in
[`web/services/telemetry_service.py`](../web/services/telemetry_service.py)
describe the collection mechanism. Please report discrepancies with this
notice so they can be corrected.
