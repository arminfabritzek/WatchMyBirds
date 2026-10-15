# Local UI development with appliance data

Develop the existing Flask/Jinja UI on a Mac or Linux computer using a private
copy of a Raspberry Pi's database and bird images. This is a developer tool,
not a second appliance, native mobile client, or deployment workflow.

## Where it belongs

The reusable scripts, tests and this guide belong in the existing WatchMyBirds
repository. They use the same UI and data services as the appliance; a separate
repository would introduce a second version of those interfaces. Private data,
source-device addresses, local passwords and generated manifests stay outside
Git. No command below commits, pushes, publishes or deploys anything.

## Prerequisites

Use Python 3.12+ and the project's virtual environment/dependencies. SSH access
to your own appliance must already work without interactive password prompts.
The remote account needs existing permission to read the output directory and
run `sudo -n python3`, `sudo -n rsync`, `sudo -n cat`, and cleanup of generated
`/tmp/wmb-local-pull-*` files. The tool does not change those permissions.
Both machines need rsync. Provision dependencies through the normal project
setup; do not run the hardware entrypoint `main.py` for this workflow.

## First use

From the repository root:

```bash
.venv/bin/python scripts/local_dev.py refresh --ssh-target admin@pi.local
.venv/bin/python scripts/local_dev.py start
```

Replace the SSH target with your device. `--remote-output` defaults to
`/opt/app/data/output`. Open **http://127.0.0.1:8080**. The local admin password
is independently generated in `local-credentials.json` in the private data root;
the Pi password is not fetched. Use the normal login form. Stop the local
server with Ctrl-C. The development notice shows the source snapshot time.

On macOS the default private data root is
`~/Library/Application Support/WatchMyBirds-dev`. On other systems pass a
suitable private directory using `--data-root` before the subcommand. A root
inside a Git checkout is refused. The first pull needs enough disk space for
the bird media, cache, snapshot and editable work copy. APFS copy-on-write avoids
full duplicate copies on supported macOS volumes; other filesystems use ordinary
independent copies and can need roughly three times the source media size.

## Daily workflow

- `start`: run the current editable work generation; local edits persist.
- `refresh --ssh-target user@host`: get a new verified Pi snapshot. Existing
  work is preserved; only the first successful pull creates a working copy.
- `reset`: create a fresh working generation from the latest verified snapshot.
  Older work remains on disk; no local edits are silently overwritten.
- `status`: show the current snapshot and working generation.

The server and data operations share a lock. Stop the server before refresh or
reset. `start --port 8081` selects another loopback port. Snapshots/caches/older
work are not automatically pruned; manage their disk usage explicitly.

## Data guarantees and limits

The Pi application stays running. The helper creates an online SQLite backup
with read-only source access. It inventories originals and optimized/thumb
bird images, with SHA-256 hashes. rsync pre-copies media, then reconciles it after
the backup; unchanged cached images are reused. Every included downloaded file,
the DB checksum, DB integrity, foreign keys and row counts are verified before
publishing the `latest.json` pointer. A failed pull keeps the last verified
snapshot and current work. Incomplete generations may remain for diagnosis.

Only bird media and the SQLite snapshot are exported. The Pi's `.env`, camera
credentials, settings, Flask keys, device identity, models and logs are not
copied. The DB itself can contain station/source metadata and image references,
so the complete data root remains private and must never be committed.

Media already absent on the source is recorded as `source_missing_media` in the
manifest. It is not silently invented or presented as a transfer success for
that image. Files present at hashing time but missing/changed after transfer
cause verification to fail. This is a verified DB snapshot with verified media,
not an atomic filesystem snapshot of the running appliance. Retention flags
allow intentionally removed originals to be distinguished from expected ones.

## Local runtime boundary

The dedicated launcher configures its working directory before importing the
real web factory. It reasserts disabled integrations after configuration loading.
It never constructs the DetectionManager or starts main.py's background jobs.
Originals and derivatives in work are independent files, never hardlinks to the
snapshot. Existing login, authorization and CSRF checks remain active.

Server-side outbound sockets, DNS to external hosts and child processes are
blocked by a process-local Python audit hook. HTTP routes permit browsing,
media, analytics, local labels, canonical training export, manual storage diagnostics
and trash/favorites; camera, configuration,
system management, ingestion, rescans and automatic background operations return 503.
The ordinary station report uses one automatic derived-view worker against the
local database copy; it performs no inference, capture or retention. The
explicitly started storage scan runs in one background thread against the local
working copy; it measures files without changing retention settings or media.
Review day navigation and local review decisions are available; Review Deep Scan
remains blocked because it schedules background inference. The read-only Settings
page is available for browser-local Light/Dark selection; configuration writes
remain blocked.
The browser is restricted to same-origin API connections and no embedded relay
frames. The service listens only on 127.0.0.1 and rejects non-loopback Host values.
This is a development guard, not an operating-system sandbox for arbitrary code.

The Live page can show real saved visits but **does not play the live camera**.
Models, detection performance, camera reconnection and appliance behavior still
need separate hardware tests. Browser fonts/styles keep the existing CDN usage.

## What this profile can test

Use this profile for galleries, stored-image viewers, labeling, favorites, trash,
analytics and responsive layouts with real copied data. It renders the same
maintained templates as the appliance; it does not add a separate UI mode.

It cannot validate PTZ movements, live transport, camera reconnects or recording.
Those require the normal appliance runtime with an explicitly selected test
camera/device. Removing the local guards would not provide a representative
hardware environment. Docker runs of the normal runtime have the same need for
real camera/device integration tests.

## Check a finished training export

The offline checker reads the actual canonical v2 ZIP. It does not start Flask,
connect to a camera or database, or change the ZIP or original images:

```bash
.venv/bin/python scripts/check_training_export.py /path/to/export.zip
```

Open `index.html` in the newly created, timestamped directory beside the ZIP.
The same directory contains `report.json` for automated inspection and resized
preview copies. It needs no server, login, JavaScript or external resources.
Existing output directories are never overwritten. To choose a new destination:

```bash
.venv/bin/python scripts/check_training_export.py /path/to/export.zip --report /tmp/export-check
```

Checks cover required files and versions, unsafe/duplicate archive paths, image
integrity, manifest counts and view membership, image/object references, COCO
sizes/categories, pixel and normalized boxes, and species labels against exported
object records. The report puts findings first and links image-specific findings
to previews. Orange boxes come from COCO; blue boxes and species labels come
from `classifier_ready.jsonl`. Toggle the two overlays independently.

All media are checked; the default preview limit is 60. Use `--preview-limit 0`
for all previews. Large or unreadable inputs fail explicitly: at most 100,000 ZIP
entries, 64 MiB per metadata file / 128 MiB total metadata, 256 MiB per image and
40 megapixels per decoded image. No archive entry is extracted to its supplied
path. Reports contain image copies; keep them private like the source export.

Exit codes: `0` means the supported consistency checks passed, `1` means findings
include errors, and `2` means invalid usage or report-output failure. Warnings do
not fail the process. The report includes the ZIP's SHA-256 for identification.
Passing does not establish agreement with the live database, detect records
omitted consistently from every export file, or prove biological labels/box
placement correct. Those still need source comparison and visual review.

## Pi deployment

Local start and refresh never imply permission to copy code to the Pi. Use the
existing explicit deployment workflow after local validation and authorization.
For this workflow, say “start locally” or “pull Pi data”; do not use a deployment
shortcut when the intended action is a Mac test.
