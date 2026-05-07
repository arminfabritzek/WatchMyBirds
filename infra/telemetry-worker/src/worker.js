/**
 * WatchMyBirds telemetry Worker.
 *
 * Receives anonymous opt-in usage heartbeats. Privacy guardrails:
 *  - Strict allowlist of payload fields (rejects unknown keys at 400).
 *  - Drops all CF-injected request metadata (country, region, IP).
 *  - Returns 204 for any successful or non-validation failure (no
 *    side-channel diagnostic info leak).
 *  - 90-day retention enforced by daily cron, not client TTL.
 *
 * Storage: Cloudflare D1, jurisdiction=eu (set at DB creation time
 * via `wrangler d1 create ... --jurisdiction=eu`).
 */

// User-Agent allowlist. Defense-in-depth against dumb scanners; trivially
// spoofable by a motivated attacker, but eliminates 99% of bot noise from
// CT-log crawlers, Shodan-style scans, and generic vulnerability probes.
// Match anything starting with "WatchMyBirds-Heartbeat/" — version-suffix
// can change freely without breaking validation.
const USER_AGENT_RE = /^WatchMyBirds-Heartbeat\/[\w.+-]+/;

const ALLOWED_FIELDS = [
  'installation_id',
  'app_version',
  'os',
  'arch',
  'cpu_count',
  'total_ram_gb',
  'python_version',
  'detector_variant',
];

const FIELD_TYPES = {
  installation_id: 'string',
  app_version: 'string',
  os: 'string',
  arch: 'string',
  cpu_count: 'number',
  total_ram_gb: 'number',
  python_version: 'string',
  detector_variant: 'string',
};

// Hard caps per field — defense against accidental or malicious bloat.
const FIELD_MAX_LEN = 64;
const INSTALLATION_ID_HEX_RE = /^[0-9a-f]{32}$/;

function validatePayload(body) {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return { ok: false, reason: 'body must be a JSON object' };
  }

  // Reject unknown keys.
  for (const key of Object.keys(body)) {
    if (!ALLOWED_FIELDS.includes(key)) {
      return { ok: false, reason: `unknown field: ${key}` };
    }
  }

  // Require all known keys.
  for (const key of ALLOWED_FIELDS) {
    if (!(key in body)) {
      return { ok: false, reason: `missing field: ${key}` };
    }
    const expected = FIELD_TYPES[key];
    if (typeof body[key] !== expected) {
      return { ok: false, reason: `wrong type for ${key}: expected ${expected}` };
    }
    if (expected === 'string' && body[key].length > FIELD_MAX_LEN) {
      return { ok: false, reason: `field too long: ${key}` };
    }
  }

  if (!INSTALLATION_ID_HEX_RE.test(body.installation_id)) {
    return { ok: false, reason: 'installation_id must be 32 lowercase hex chars' };
  }

  if (!Number.isInteger(body.cpu_count) || body.cpu_count < 1 || body.cpu_count > 256) {
    return { ok: false, reason: 'cpu_count out of range' };
  }
  if (!Number.isInteger(body.total_ram_gb) || body.total_ram_gb < 0 || body.total_ram_gb > 4096) {
    return { ok: false, reason: 'total_ram_gb out of range' };
  }

  return { ok: true };
}

async function handleHeartbeat(request, env) {
  if (request.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  // User-Agent allowlist — block obvious scanners early, before we parse
  // anything. Returns 404 (not 400) to make this look like the path
  // doesn't exist for non-clients, reducing scanner reconnaissance value.
  const ua = request.headers.get('user-agent') || '';
  if (!USER_AGENT_RE.test(ua)) {
    return new Response('Not Found', { status: 404 });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return new Response('invalid JSON', { status: 400 });
  }

  const v = validatePayload(body);
  if (!v.ok) {
    return new Response(v.reason, { status: 400 });
  }

  const utcDate = new Date().toISOString().slice(0, 10); // 'YYYY-MM-DD'
  const ts = Math.floor(Date.now() / 1000);

  try {
    await env.DB.prepare(
      `INSERT OR REPLACE INTO heartbeats
       (installation_id, date, app_version, os, arch, cpu_count, total_ram_gb, python_version, detector_variant, ts)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
      .bind(
        body.installation_id,
        utcDate,
        body.app_version,
        body.os,
        body.arch,
        body.cpu_count,
        body.total_ram_gb,
        body.python_version,
        body.detector_variant,
        ts,
      )
      .run();
  } catch (err) {
    // Swallow — never expose DB errors to caller. Log to Worker tail.
    console.error('d1_insert_failed', err.message);
    return new Response(null, { status: 204 });
  }

  return new Response(null, { status: 204 });
}

export default {
  async fetch(request, env, _ctx) {
    const url = new URL(request.url);

    if (url.pathname === '/v1/heartbeat') {
      return handleHeartbeat(request, env);
    }

    if (url.pathname === '/health') {
      return new Response('ok', { status: 200 });
    }

    return new Response('Not Found', { status: 404 });
  },

  // Cron handler. Two schedules registered in wrangler.toml:
  //   "0 4 * * *"  — 04:00 UTC, 90-day retention sweep (legacy safety net)
  //   "30 4 * * *" — 04:30 UTC, daily aggregation + raw-row prune
  //
  // We dispatch on event.cron because Cloudflare passes the schedule
  // string that fired the event. Keeps the two jobs visibly separate
  // even though they share one handler.
  async scheduled(event, env, _ctx) {
    if (event.cron === '30 4 * * *') {
      // Aggregation + 24h prune. Aggregate yesterday's raw heartbeats
      // into daily_aggregates by (cohort, app_version, detector_variant),
      // then delete those raw rows. After this fires every day, the
      // heartbeats table only ever contains today's rows.
      try {
        const aggregateResult = await env.DB.prepare(
          `INSERT OR REPLACE INTO daily_aggregates
             (date, app_version, os, arch, cpu_count, total_ram_gb,
              detector_variant, install_count)
           SELECT
             date,
             app_version,
             os,
             arch,
             cpu_count,
             total_ram_gb,
             detector_variant,
             COUNT(DISTINCT installation_id) AS install_count
           FROM heartbeats
           WHERE date = date('now', '-1 day')
           GROUP BY date, app_version, os, arch, cpu_count, total_ram_gb,
                    detector_variant`
        ).run();

        const pruneResult = await env.DB.prepare(
          `DELETE FROM heartbeats WHERE date < date('now')`
        ).run();

        console.log('daily_aggregation', {
          aggregates_written: aggregateResult.meta?.changes ?? 0,
          raw_rows_deleted: pruneResult.meta?.changes ?? 0,
        });
      } catch (err) {
        console.error('daily_aggregation_failed', err.message);
      }
      return;
    }

    // Default branch: 90-day safety-net retention sweep at 04:00 UTC.
    // Should be a no-op once the 04:30 aggregation cron has been
    // running for a day, since raw rows older than today are deleted
    // there. Kept as belt-and-braces against an aggregation failure.
    const cutoff = Math.floor(Date.now() / 1000) - 90 * 86400;
    try {
      const result = await env.DB.prepare(
        'DELETE FROM heartbeats WHERE ts < ?'
      ).bind(cutoff).run();
      console.log('retention_sweep', { deleted: result.meta?.changes ?? 0, cutoff });
    } catch (err) {
      console.error('retention_sweep_failed', err.message);
    }
  },
};
