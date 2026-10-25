/**
 * WatchMyBirds telemetry Worker.
 *
 * Receives pseudonymous opt-in usage heartbeats. Privacy guardrails:
 *  - Strict allowlist of payload fields (rejects unknown keys at 400).
 *  - Drops all CF-injected request metadata (country, region, IP).
 *  - Acknowledges only persisted heartbeats; database errors stay private.
 *  - Retains 90 UTC dates of reports and aggregates before expiry.
 *
 * Storage: Cloudflare D1, jurisdiction=eu (set at DB creation time
 * via `wrangler d1 create ... --jurisdiction=eu`).
 */

const USER_AGENT_RE = /^WatchMyBirds-Heartbeat\/[\w.+-]+$/;
const MAX_BODY_BYTES = 4096;
const METRIC_LIMITS = {
  setup_completed: 1, camera_frame: 1, camera_error: 1, model_ready: 1,
  model_error: 1, inference_ok: 1, inference_error: 1, review_used: 1,
  correction_used: 1, export_used: 1, web_error: 1,
  uptime_bucket: 5, inference_latency_bucket: 5,
};
const METRIC_MERGE = Object.keys(METRIC_LIMITS).map(key =>
  `'${key}', CASE WHEN json_type(diagnostic_reports.metrics, '$.${key}') IS NULL
   AND json_type(excluded.metrics, '$.${key}') IS NULL THEN NULL ELSE
   max(COALESCE(json_extract(diagnostic_reports.metrics, '$.${key}'), 0),
       COALESCE(json_extract(excluded.metrics, '$.${key}'), 0)) END`
).join(', ');
const EXTRA_FIELDS = ['schema_version', 'consent_version', 'report_date', 'deployment', 'metrics'];

function response(text, status, extra = {}) {
  return new Response(text, { status, headers: {
    'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...extra,
  }});
}

async function readBody(request) {
  if (Number(request.headers.get('content-length')) > MAX_BODY_BYTES) throw 413;
  const reader = request.body?.getReader();
  if (!reader) throw 400;
  let size = 0;
  const chunks = [];
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES) { await reader.cancel(); throw 413; }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { throw 400; }
}

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

function validatePayload(body, today) {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return { ok: false, reason: 'body must be a JSON object' };
  }

  // Reject unknown keys.
  for (const key of Object.keys(body)) {
    if (!ALLOWED_FIELDS.includes(key) && !(body.schema_version === 2 && EXTRA_FIELDS.includes(key))) {
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

  if (body.schema_version === 2) {
    if (body.consent_version !== 2 || !EXTRA_FIELDS.every(key => Object.hasOwn(body, key))) {
      return { ok: false, reason: 'explicit diagnostics consent required' };
    }
    const oldest = new Date(Date.parse(today) - 7 * 86400000).toISOString().slice(0, 10);
    if (typeof body.report_date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(body.report_date)
        || body.report_date < oldest || body.report_date > today
        || !Number.isFinite(Date.parse(body.report_date))
        || new Date(body.report_date).toISOString().slice(0, 10) !== body.report_date) {
      return { ok: false, reason: 'report date outside accepted window' };
    }
    if (!['container', 'raspberry_pi', 'source'].includes(body.deployment)
        || typeof body.metrics !== 'object' || body.metrics === null || Array.isArray(body.metrics)) {
      return { ok: false, reason: 'invalid diagnostics' };
    }
    for (const [key, value] of Object.entries(body.metrics)) {
      if (!Object.hasOwn(METRIC_LIMITS, key) || !Number.isInteger(value) || value < 0 || value > METRIC_LIMITS[key]) {
        return { ok: false, reason: 'invalid metric' };
      }
    }
  }
  return { ok: true };
}

async function handleHeartbeat(request, env) {
  if (request.method !== 'POST') return response('Method Not Allowed', 405);
  if (!USER_AGENT_RE.test(request.headers.get('user-agent') || '')) return response('Not Found', 404);
  if (request.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') {
    return response('JSON required', 415);
  }
  try {
    // This bounds anonymous ingress without retaining source IP addresses.
    if (!(await env.INGRESS_LIMITER.limit({ key: 'heartbeats' })).success) {
      return response('Retry later', 429, { 'Retry-After': '60' });
    }
  } catch { return response('Temporarily unavailable', 503); }
  let body;
  try { body = await readBody(request); }
  catch (status) { return response('Invalid request body', status === 413 ? 413 : 400); }
  const today = new Date().toISOString().slice(0, 10);
  const validation = validatePayload(body, today);
  if (!validation.ok) return response(validation.reason, 400);
  try {
    if (!(await env.INSTALLATION_LIMITER.limit({ key: body.installation_id })).success) {
      return response('Retry later', 429, { 'Retry-After': '60' });
    }
    const day = body.schema_version === 2 ? body.report_date : today;
    const ts = Math.floor(Date.now() / 1000);
    const statements = [env.DB.prepare(
      `INSERT OR REPLACE INTO heartbeats
       (installation_id, date, app_version, os, arch, cpu_count, total_ram_gb, python_version, detector_variant, ts)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(body.installation_id, day, body.app_version, body.os, body.arch,
           body.cpu_count, body.total_ram_gb, body.python_version, body.detector_variant, ts)];
    if (body.schema_version === 2) {
      statements.push(env.DB.prepare(
        `INSERT INTO diagnostic_reports
         (installation_id, date, consent_version, deployment, metrics, ts) VALUES (?, ?, 2, ?, ?, ?)
         ON CONFLICT (installation_id, date) DO UPDATE SET
           metrics=json_patch('{}', json_object(${METRIC_MERGE})),
           deployment=excluded.deployment, ts=max(diagnostic_reports.ts, excluded.ts)`
      ).bind(body.installation_id, day, body.deployment, JSON.stringify(body.metrics), ts));
    }
    await env.DB.batch(statements);
    return response(null, 204);
  } catch {
    console.error('heartbeat_persistence_failed');
    return response('Temporarily unavailable', 503);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/v1/heartbeat') return handleHeartbeat(request, env);
    if (url.pathname === '/health') return response('ok', 200);
    return response('Not Found', 404);
  },

  async scheduled(event, env) {
    if (event.cron !== '30 4 * * *') return;
    const today = new Date(event.scheduledTime).toISOString().slice(0, 10);
    const oldest = new Date(event.scheduledTime - 89 * 86400000).toISOString().slice(0, 10);
    try {
      await env.DB.batch([
        // Replace only dates backed by raw rows; preserve pre-migration history.
        env.DB.prepare(`DELETE FROM daily_aggregates WHERE date IN
          (SELECT DISTINCT date FROM heartbeats WHERE date < ?)`).bind(today),
        env.DB.prepare(`INSERT INTO daily_aggregates
          (date, app_version, os, arch, cpu_count, total_ram_gb, detector_variant, install_count)
          SELECT date, app_version, os, arch, cpu_count, total_ram_gb, detector_variant,
                 COUNT(DISTINCT installation_id) FROM heartbeats WHERE date < ?
          GROUP BY date, app_version, os, arch, cpu_count, total_ram_gb, detector_variant`).bind(today),
        env.DB.prepare('DELETE FROM diagnostic_reports WHERE date < ?').bind(oldest),
        env.DB.prepare('DELETE FROM heartbeats WHERE date < ?').bind(oldest),
        env.DB.prepare(`INSERT OR REPLACE INTO telemetry_jobs (name, last_success)
                        VALUES ('daily_retention', ?)`).bind(Math.floor(event.scheduledTime / 1000)),
      ]);
      console.log('daily_retention_complete');
    } catch (error) {
      console.error('daily_retention_failed');
      throw error;
    }
  },
};
