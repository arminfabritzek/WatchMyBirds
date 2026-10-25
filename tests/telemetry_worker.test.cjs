const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { createRequire } = require('node:module');
const path = require('node:path');
const { test } = require('node:test');

const workerDir = path.resolve(__dirname, '../infra/telemetry-worker');
const workerRequire = createRequire(path.join(workerDir, 'package.json'));
const { Miniflare, convertV4MiniflareOptions } = workerRequire('miniflare');
const source = readFileSync(path.join(workerDir, 'src/worker.js'), 'utf8');
const workerPromise = import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const payload = {
  installation_id: '0123456789abcdef0123456789abcdef',
  app_version: 'test', os: 'linux', arch: 'aarch64', cpu_count: 4,
  total_ram_gb: 8, python_version: '3.12.12', detector_variant: 'test',
};
const event = { cron: '30 4 * * *', scheduledTime: Date.parse('2026-10-24T04:30:00Z') };

async function database(t) {
  const options = convertV4MiniflareOptions({
    modules: true, script: source, compatibilityDate: '2026-01-01',
    host: '127.0.0.1', d1Databases: ['DB'], d1Persist: false,
    ratelimits: { INGRESS_LIMITER: { simple: { limit: 120, period: 60 }, namespace_id: '10421' }, INSTALLATION_LIMITER: { simple: { limit: 12, period: 60 }, namespace_id: '10422' } },
  });
  options.telemetry = { enabled: false };
  const mf = new Miniflare(options);
  t.after(() => mf.dispose());
  const db = await mf.getD1Database('DB');
  const schema = readFileSync(path.join(workerDir, 'schema.sql'), 'utf8')
    .replace(/--[^\n]*/g, '').split(';').map(s => s.trim()).filter(Boolean);
  await db.batch(schema.map(sql => db.prepare(sql)));
  return { mf, db, worker: (await workerPromise).default };
}

function request(body = payload) {
  return ['https://example.test/v1/heartbeat', {
    method: 'POST', headers: { 'User-Agent': 'WatchMyBirds-Heartbeat/test', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }];
}

async function seed(db, date, id = payload.installation_id, version = 'test') {
  await db.prepare(`INSERT INTO heartbeats VALUES (?, ?, ?, 'linux', 'aarch64', 4, 8, '3.12', 'test', 1)`)
    .bind(id, date, version).run();
}

async function rows(db, table) {
  return (await db.prepare(`SELECT * FROM ${table} ORDER BY date${table === "diagnostic_reports" ? "" : ", app_version"}`).all()).results;
}

test('heartbeat acknowledgement follows persistence and retries deduplicate', async t => {
  const { mf, db } = await database(t);
  for (let i = 0; i < 2; i++) {
    const response = await mf.dispatchFetch(...request());
    assert.equal(response.status, 204);
  }
  const stored = await rows(db, 'heartbeats');
  assert.equal(stored.length, 1);
  assert.equal(stored[0].installation_id, payload.installation_id);
  assert.equal(stored[0].app_version, payload.app_version);
});

test('database write failure is retryable without leaking diagnostics', async t => {
  const { mf, db } = await database(t);
  await db.prepare("CREATE TRIGGER fail_write BEFORE INSERT ON heartbeats BEGIN SELECT RAISE(ABORT, 'private database detail'); END").run();
  const failed = await mf.dispatchFetch(...request());
  assert.equal(failed.status, 503);
  assert.equal(await failed.text(), 'Temporarily unavailable');
  assert.deepEqual(await rows(db, 'heartbeats'), []);
  await db.prepare('DROP TRIGGER fail_write').run();
  assert.equal((await mf.dispatchFetch(...request())).status, 204);
  assert.equal((await rows(db, 'heartbeats')).length, 1);
});

test('unknown payload fields are rejected before persistence', async t => {
  const { mf, db } = await database(t);
  assert.equal((await mf.dispatchFetch(...request({ ...payload, ip: 'test' }))).status, 400);
  assert.deepEqual(await rows(db, 'heartbeats'), []);
});

test('aggregation catches up missed dates and preserves current/future dates and old counts', async t => {
  const { db, worker } = await database(t);
  await db.prepare("INSERT INTO daily_aggregates VALUES ('2026-01-01','old','linux','aarch64',4,8,'test',7)").run();
  await seed(db, '2026-01-02');
  await seed(db, '2026-10-22');
  await seed(db, '2026-10-22', 'a'.repeat(32));
  await seed(db, '2026-10-22', 'b'.repeat(32), 'other');
  await seed(db, '2026-10-23');
  await seed(db, '2026-10-24');
  await seed(db, '2026-10-25');
  await worker.scheduled(event, { DB: db });
  assert.equal((await rows(db, 'heartbeats')).length, 6);
  const aggregates = await rows(db, 'daily_aggregates');
  assert.deepEqual(aggregates.map(r => [r.date, r.app_version, r.install_count]), [
    ['2026-01-01','old',7], ['2026-01-02','test',1],
    ['2026-10-22','other',1], ['2026-10-22','test',2], ['2026-10-23','test',1],
  ]);
  assert.equal('installation_id' in aggregates[0], false);
  await worker.scheduled(event, { DB: db });
  assert.deepEqual(await rows(db, 'daily_aggregates'), aggregates);
});

for (const [operation, table] of [['INSERT', 'daily_aggregates'], ['DELETE', 'heartbeats']]) {
  test(`failed ${operation} rolls back aggregation and pruning; next run recovers`, async t => {
    const { db, worker } = await database(t);
    await seed(db, '2026-01-01');
    await db.prepare(`CREATE TRIGGER fail_step BEFORE ${operation} ON ${table} BEGIN SELECT RAISE(ABORT, 'injected failure'); END`).run();
    await assert.rejects(worker.scheduled(event, { DB: db }), /injected failure/);
    assert.equal((await rows(db, 'heartbeats')).length, 1);
    assert.deepEqual(await rows(db, 'daily_aggregates'), []);
    await db.prepare('DROP TRIGGER fail_step').run();
    await worker.scheduled(event, { DB: db });
    assert.deepEqual(await rows(db, 'heartbeats'), []);
    assert.equal((await rows(db, 'daily_aggregates'))[0].install_count, 1);
  });
}

test('legacy retention trigger never deletes unaggregated data', async t => {
  const { db, worker } = await database(t);
  await seed(db, '2020-01-01');
  await worker.scheduled({ ...event, cron: '0 4 * * *' }, { DB: db });
  assert.equal((await rows(db, 'heartbeats')).length, 1);
  assert.deepEqual(await rows(db, 'daily_aggregates'), []);
});

const diagnostics = () => ({ ...payload, schema_version: 2, consent_version: 2,
  report_date: new Date().toISOString().slice(0,10), deployment: 'source',
  metrics: { camera_frame: 1, review_used: 1, inference_latency_bucket: 2 } });

test('diagnostics require explicit versioned consent and bounded metrics', async t => {
  const { mf, db } = await database(t);
  for (const invalid of [
    { ...diagnostics(), consent_version: 1 },
    { ...diagnostics(), metrics: { camera_url: 'rtsp://private' } },
    { ...diagnostics(), metrics: { review_used: 5 } },
    { ...diagnostics(), report_date: '2020-01-01' },
    { ...diagnostics(), deployment: 'secret-hostname' },
  ]) assert.equal((await mf.dispatchFetch(...request(invalid))).status, 400);
  assert.equal((await mf.dispatchFetch(...request(diagnostics()))).status, 204);
  assert.equal((await rows(db, 'diagnostic_reports')).length, 1);
  assert.equal((await rows(db, 'heartbeats')).length, 1);
  const updated = diagnostics(); updated.metrics.export_used = 1;
  assert.equal((await mf.dispatchFetch(...request(updated))).status, 204);
  assert.equal((await rows(db, 'diagnostic_reports')).length, 1);
  assert.equal(JSON.parse((await rows(db, 'diagnostic_reports'))[0].metrics).export_used, 1);
  assert.equal((await mf.dispatchFetch(...request(diagnostics()))).status, 204);
  const merged = JSON.parse((await rows(db, 'diagnostic_reports'))[0].metrics);
  assert.equal(merged.export_used, 1);
  assert.equal('uptime_bucket' in merged, false);
});

test('body limit applies to streamed bytes and content type is required', async t => {
  const { mf, db } = await database(t);
  const [url, options] = request();
  assert.equal((await mf.dispatchFetch(url, { ...options, body: ' '.repeat(4097) })).status, 413);
  assert.equal((await mf.dispatchFetch(url, { ...options, headers: { 'User-Agent': 'WatchMyBirds-Heartbeat/test' } })).status, 415);
  assert.equal((await rows(db, 'heartbeats')).length, 0);
});

test('rate limits fail closed and refuse before database writes', async t => {
  const { worker, db } = await database(t);
  let reads = 0;
  const deny = { limit: async () => ({ success: false }) };
  const allow = { limit: async () => ({ success: true }) };
  const badDB = { prepare: () => { reads++; throw new Error('unexpected'); } };
  const call = env => worker.fetch(new Request(...request()), env);
  assert.equal((await call({ DB: badDB, INGRESS_LIMITER: deny })).status, 429);
  assert.equal((await call({ DB: badDB })).status, 503);
  assert.equal((await call({ DB: badDB, INGRESS_LIMITER: allow, INSTALLATION_LIMITER: deny })).status, 429);
  assert.equal(reads, 0);
});

test('retention keeps exactly 90 UTC dates and preserves totals before deletion', async t => {
  const { db, worker } = await database(t);
  await seed(db, '2026-07-26');
  await seed(db, '2026-07-27');
  await seed(db, '2026-10-24');
  await db.prepare("INSERT INTO diagnostic_reports VALUES (?, '2026-07-26', 2, 'source', '{}', 1)").bind(payload.installation_id).run();
  await worker.scheduled(event, { DB: db });
  assert.deepEqual((await rows(db, 'heartbeats')).map(r => r.date), ['2026-07-27', '2026-10-24']);
  assert.equal((await rows(db, 'diagnostic_reports')).length, 0);
  assert.equal((await rows(db, 'daily_aggregates')).length, 2);
  assert.ok((await db.prepare('SELECT last_success FROM telemetry_jobs').first()).last_success);
});
