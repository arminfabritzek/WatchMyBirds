# WatchMyBirds Telemetry Worker

Cloudflare Worker that receives anonymous opt-in usage heartbeats
from running WatchMyBirds installations.

**This Worker does not run unless an operator explicitly enables
telemetry in Settings → Privacy.** The default in the app is OFF.
See `docs/PRIVACY.md` (or Settings → Privacy in a running install)
for the full data policy.

## Architecture

- **Endpoint:** `POST https://watchmybirds-telemetry.wmb-infra.workers.dev/v1/heartbeat`
- **Storage:** Cloudflare D1 (`watchmybirds-heartbeats`),
  `jurisdiction=eu` for hard EU data residency.
- **Retention:** 90 days, enforced by daily cron (04:00 UTC) inside
  this same Worker.
- **Read access:** local `wrangler d1 execute` only. No public
  dashboard, no admin endpoint.

## What gets stored

One row per (installation_id, UTC date) — see `schema.sql`. PK
prevents duplicate writes per day even from misbehaving clients.

What does NOT get stored: IP address, country, region, locale,
hostname, MAC, exact RAM bytes, kernel version, Pi model string,
observation count, species names, image data. The Worker explicitly
does not read CF-injected request metadata before writing to D1.

## Local development

```bash
npm install
npx wrangler login                  # browser opens for OAuth
npx wrangler dev                    # local dev with mock D1
```

## First-time deploy

1. Create the D1 database with EU jurisdiction:
   ```bash
   npx wrangler d1 create watchmybirds-heartbeats \
     --location=weur \
     --jurisdiction=eu
   ```
   Copy the `database_id` from the output into `wrangler.toml`.

2. Apply the schema:
   ```bash
   npm run db:apply
   ```

3. Deploy the Worker:
   ```bash
   npm run deploy
   ```

4. Test:
   ```bash
   curl -X POST https://watchmybirds-telemetry.wmb-infra.workers.dev/v1/heartbeat \
     -H "Content-Type: application/json" \
     -d '{
       "installation_id": "0123456789abcdef0123456789abcdef",
       "app_version": "v0.0.0-test",
       "os": "linux",
       "arch": "aarch64",
       "cpu_count": 4,
       "total_ram_gb": 8,
       "python_version": "3.12.3",
       "detector_variant": "yolox-tiny-int8"
     }'
   # expect: 204 No Content
   ```

5. Verify the row landed:
   ```bash
   npm run db:dau
   ```

## Operations

- **DAU/WAU/MAU snapshot:**
  ```bash
  npm run db:dau
  npm run db:wau
  npm run db:mau
  ```
- **Live request stream:** `npm run tail`
- **Free-tier headroom:** 100k row-writes/day, 5M row-reads/day,
  5GB storage. We can support ~70k installations on the free plan
  before hitting storage; well-watch on the dashboard.

## When to upgrade plan

If we ever cross ~50k active opted-in installations consistently,
upgrade to Workers Paid ($5/mo) for 10M req/day and 25M D1 row-writes.
Until then, Free tier is fine and aligned with the project's
hobbyist-scale economics.
