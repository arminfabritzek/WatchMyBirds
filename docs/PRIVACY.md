# Privacy

WatchMyBirds runs on your Raspberry Pi. Bird activity, images, and detections
stay there unless you explicitly enable an outbound integration such as
Telegram or MQTT. MQTT sends detection metadata and an image URL to the broker
you configure; a consumer that can reach WatchMyBirds can fetch that image.

WatchMyBirds disables ONNX Runtime's upstream telemetry before loading the
inference library. This is independent of the optional installation reports below.

The web interface serves its scripts and styles from your installation and uses
system fonts. Opening a page does not request assets from Google Fonts or a CDN.
External links, such as Wikipedia and GitHub, connect to those sites when opened.

The separate optional traffic to the WatchMyBirds operator is a voluntary
installation report that helps us understand adoption, with a separate choice for usage and diagnostic summaries. **It is off by
default**, and you can enable it during initial password setup or in
**Settings → Privacy** in your running install.

This document mirrors the `/privacy` page in the running app. If they ever
disagree, please report the discrepancy. The collection code is in `web/services/telemetry_service.py` and
`infra/telemetry-worker/src/worker.js`.

---

## Basic reports (only when you opt in)

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

## Optional usage and diagnostics (consent version 2)

The second, initially unchecked choice enables additional daily summaries. Existing
consent never enables these fields automatically. Declining does not restrict features.
The client stores the consent version and decision time locally in settings.yaml.

In addition to the eight basic fields, these reports contain `schema_version: 2`,
`consent_version: 2`, the UTC `report_date`, `deployment` (`container`,
`raspberry_pi`, or `source`), and a bounded `metrics` object:

| Metric | Meaning |
|---|---|
| `setup_completed` | Initial password setup completed after choosing diagnostics |
| `camera_frame`, `camera_error` | A decoded frame arrived / camera recovery was requested |
| `model_ready`, `model_error` | Detector initialization succeeded / failed when inference was requested |
| `inference_ok`, `inference_error` | An inference call succeeded / raised an error |
| `review_used`, `correction_used`, `export_used` | An authenticated operator successfully used the corresponding page/action |
| `web_error` | An authenticated request returned a server error |
| `uptime_bucket` | Longest observed process uptime that day: <1h, 1–6h, 6–24h, 1–7d, 7–30d, or ≥30d (codes 0–5) |
| `inference_latency_bucket` | Slowest observed successful inference: <50, 50–100, 100–250, 250–1000, 1000–5000, or ≥5000 ms (codes 0–5) |

Flags are 0/1, not event counts. Missing fields mean unobserved, not failure or
non-use. Setup completion does not prove camera setup succeeded. Export use does
not prove an archive was received or shared. Version and deployment describe the
runtime when the report is sent; they may have changed since a queued report day.
Source checkouts fall back to Git's version/tag/commit description, with a dirty
marker for local changes, when no packaged version is available.

Daily summaries are refreshed at most hourly and deduplicated by installation/date.
A local queue retains today and up to seven prior UTC dates for retries after network
outages or restarts. Retries use backoff; failed writes are never marked delivered.
Turning sharing off or reverting to basic reports clears queued diagnostics.
Previously delivered reports remain until expiry or a deletion request.

## What the heartbeat does not include

- Your IP address, country, region, or any geo-location
- Your locale, language, or timezone
- Your hostname, MAC address, or any hardware serial
- The Raspberry Pi model string or kernel version
- Exact RAM in bytes (we round to whole GB on purpose)
- Any image, video, or audio data
- Any species names, observation counts, or detection events
- Raw error messages, stack traces, or debug logs (diagnostics sends only the fixed flags above)
- Camera URLs, settings, passwords, or anything from your config
- Your email or any identifier from your operating system

## Training-data exports

The Export page prepares ZIP parts of answered photos in the background on this
device. A temporary database copy freezes the answer context; completed ZIPs are
then downloaded to your browser. Local ZIP staging is limited to 2 GiB, with
additional space needed for the database copy. Artifacts older than seven days
are cleaned up on startup or the next preparation; active downloads are preserved.
You can remove each local ZIP after confirming receipt. Each selected photo
includes every object and image
statement on it, manual objects, answer history and model proposal context.
Sample IDs contain the stable local `labeling_installation_id`, separate from
the optional heartbeat ID. It identifies the labeling installation across exports.

The device records the export's date, answer revisions and your explicit
"Export received" confirmation date. Unconfirmed exports can be downloaded again
with their original context. Changed answers wait again. WatchMyBirds does not
upload this bundle or send species counts to a server; sharing requires your own
manual transfer. A later optional upload would need its own explanation and
per-station opt-in before implementation.

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

Cloudflare D1 is the reporting authority. Individual installation reports and
consented diagnostics cover today and the preceding 89 UTC dates. The scheduled
04:30 UTC job updates daily installation/cohort counts and removes expired reports
in one transaction. The maintainer's local reporting cache follows the same window;
it is not needed to preserve cloud history. Cached rows expire when the cache runs.
If a scheduled job fails, physical cloud deletion waits for the next successful
run; reporting queries exclude expired dates. Cleanup success is recorded for monitoring.

Daily aggregate installation/cohort counts contain no installation IDs and remain
for trend analysis. They cannot reconstruct unique returning installations across
days. Diagnostic details are not retained in these long-term aggregates. Small
cohorts can still be distinctive; we do not describe the whole dataset as anonymous.

Cloudflare's database recovery backups have a separate provider-controlled retention
window. Restored databases must run the retention job before reporting access resumes.
Local diagnostic queues are included if an operator backs up the entire data directory;
such operator-managed copies must be covered by that operator's backup policy.

## Your controls

- **Off (default)** — nothing is sent. Ever. The first time you toggle
  telemetry on, we generate a random `installation_id` and store it in
  `settings.yaml` on your device. Until then, no ID exists.

- **Toggle off later** — pings stop immediately. Your `installation_id`
  stays the same, so if you turn it back on, you're counted as the same
  install (not a new one).

- **Rotate ID** — if you want the next opt-in to be counted as a fresh
  install, click the **Rotate ID** button in Settings → Privacy. This wipes
  your current ID and generates a new one. Future reports use the new ID. Rotation clears pending local diagnostics, but does not delete previously received reports; those expire under the 90-day policy.

- **Block at the firewall** — the heartbeat hostname is deliberately
  separate from any other WatchMyBirds endpoint, so you can firewall-block
  `watchmybirds-telemetry.wmb-infra.workers.dev` without breaking anything else in the app.

- **Override the endpoint** — set `telemetry_endpoint` in `settings.yaml` to
  point at an HTTPS URL you control (or an HTTP loopback URL such as `http://localhost/discard`). Redirects are refused. The
  toggle's "on" state then sends to wherever you said.

## Why we ask at all

WatchMyBirds is a small open-source project maintained by one person.
GitHub stars and clones don't tell us if anyone actually runs the app —
only if they bookmarked it. A daily heartbeat lets us see whether the
project is being used, which informs decisions about what to fix, what to
build next, and whether continuing is worth the time. Optional diagnostics additionally helps identify setup problems, release regressions, performance issues and feature use.

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
