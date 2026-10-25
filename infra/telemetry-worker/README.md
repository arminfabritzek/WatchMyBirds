# WatchMyBirds telemetry Worker

Cloudflare D1 holds 90 UTC dates of opt-in installation history. The maintainer's
computer is a read-only reporting client, not a collection or retention dependency.
See [Privacy](../../docs/PRIVACY.md) for the exact consent, fields and retention policy.

## Protocol and storage

`POST /v1/heartbeat` accepts the original eight-field payload and the explicitly
consented version-2 payload described in Privacy. Existing clients remain supported.
`heartbeats` stores one row per installation/date; `diagnostic_reports` stores the
optional metrics for that same date. Both writes are atomic. A 204 acknowledges
persistence; a 503 must be retried. Version-2 clients refresh daily summaries at most
hourly, with a seven-day backfill window. This is self-reported installation activity,
not authenticated people or guaranteed device health.

At 04:30 UTC, one transaction rebuilds completed dates backed by raw reports,
preserves older aggregate-only history, expires individual rows older than the
90-date window, and records success in `telemetry_jobs`. Failed transactions roll
back and are retried on the next scheduled run. Reads must also filter the window.
No schema migration can recover reports already removed by the old daily pruning.
Long-term aggregates hold installation/cohort counts only, not diagnostic details.

## Security

- JSON only, 4096-byte streamed body cap, strict field/type/range/date allowlists.
- Parameterized SQL; database diagnostics never returned to callers.
- Cloudflare rate-limit bindings: 120 requests/minute for ingress and 12/minute per
  installation. Missing/broken bindings fail closed with 503; denials return 429.
- Limits are approximate and per Cloudflare location, not a global budget or identity
  check. Distributed abuse and forged installation IDs remain possible. Monitor
  Worker traffic and D1 usage; no shared secret is embedded in public clients.
- IP/location metadata is not stored in D1 or used as a rate-limit key.
- `/health` is liveness only. Use the authenticated D1 API for cleanup freshness;
  alert if `telemetry_jobs.daily_retention` is over 48 hours old.
- No public reporting endpoint. Keep D1 Read and deploy credentials separate.

Rate-limit semantics: https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/

## Local development

Wrangler builds/deploys the Worker; Miniflare tests it locally with disposable D1
instances. Their patched dependencies, including undici, run on developer machines
and CI and are not imported by the deployed Worker or Python appliance.
Use the npm version in package.json, `npm ci --ignore-scripts`, `npm test`, and
`npm audit --package-lock-only --include=dev`. No cloud credentials are needed for tests.

## Upgrade order (requires separate deployment authorization)

1. Apply `schema.sql` to the existing EU-jurisdiction D1 database. It is additive and
   idempotent: existing tables/rows remain; diagnostics and job-status tables are new.
2. Check that rate-limit namespace IDs 10421 and 10422 are unused by unrelated Workers
   in this account, then deploy the Worker and its configured bindings/schedule.
3. Verify storage acknowledgements and scheduled cleanup with an explicitly authorized
   production check. Deploy updated clients and activate the private reporting client.
4. Ask operators to opt into the expanded scope. Never upgrade old consent silently.

Commands for an authorized rollout: `npm run db:apply`, then `npm run deploy`.
Applying schema alone does not change the old Worker's daily deletion behavior.
Avoid rollback to the old pruning Worker: it would remove retained raw history.
A restored backup must run retention before reporting resumes.

## Reporting

The private Python reader uses fixed SELECT queries via a D1 Read token and joins
retained reports with diagnostic flags. Its cache trims expired individual rows on
opening and merging, preserves aggregate trends, and exits nonzero on fetch failure.
Never add that private reader, token, device mapping or installation IDs to this repo.

`npm run db:dau`, `db:wau`, and `db:mau` count distinct reporting IDs for today,
7 dates and 30 dates. Coverage is incomplete until sufficient history accumulates;
these are neither total users nor retention percentages. For returning activity,
count IDs reporting on more than one distinct date within the chosen window.
