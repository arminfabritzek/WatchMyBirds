const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

const template = fs.readFileSync('templates/settings.html', 'utf8');
const script = template.slice(
    template.indexOf('    let retentionPreviewTimer = null;'),
    template.indexOf('    function retentionRun()')
);

function harness(replies) {
    const out = {textContent: ''};
    const timers = new Map();
    const events = {};
    const requests = [];
    let nextTimer = 0;
    const context = vm.createContext({
        document: {getElementById: () => out},
        window: {
            addEventListener: (name, callback) => { events[name] = callback; },
            clearTimeout: id => timers.delete(id),
            setTimeout: callback => { timers.set(++nextTimer, callback); return nextTimer; }
        },
        fetch: async url => {
            requests.push(url);
            const data = await replies.shift();
            return {ok: true, json: async () => data};
        },
        _fmtBytes: value => `${value} B`,
        _updateRetentionLastRun: () => {}
    });
    vm.runInContext(script, context);
    return {out, timers, events, requests, context};
}

function complete(photos = 1) {
    return {status: 'complete', enabled: false, migration_ready: true,
        next_14_days: {photos, files: photos, bytes: photos},
        calculated_at: '2026-10-30T12:00:00Z'};
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test('long previews show progress and poll until completion without restarting', async () => {
    const state = harness([
        {status: 'running', progress: {images_done: 3, images_total: 10}},
        complete()
    ]);
    state.context.retentionPreview();
    await settle();
    assert.match(state.out.textContent, /3 \/ 10 photos checked/);
    assert.deepEqual(state.requests, ['/api/v1/retention/preview?refresh=1']);
    const [id, poll] = state.timers.entries().next().value;
    state.timers.delete(id);
    await poll();
    assert.equal(state.requests[1], '/api/v1/retention/preview');
    assert.match(state.out.textContent, /Due within the next 14 days/);
    assert.match(state.out.textContent, /Calculated/);
    assert.equal(state.timers.size, 0);
});

test('an older response cannot overwrite a more recent explicit refresh', async () => {
    let resolveOld;
    const old = new Promise(resolve => { resolveOld = resolve; });
    const state = harness([old, complete(2)]);
    state.context.retentionPreview();
    state.context.retentionPreview();
    await settle();
    assert.match(state.out.textContent, /2 photos/);
    resolveOld({status: 'running', progress: {images_done: 0}});
    await settle();
    assert.match(state.out.textContent, /2 photos/);
    assert.equal(state.timers.size, 0);
});

test('leaving the page prevents pending responses from restarting polling', async () => {
    let resolveReply;
    const pending = new Promise(resolve => { resolveReply = resolve; });
    const state = harness([pending]);
    state.context.retentionPreview(false);
    state.events.pagehide();
    resolveReply({status: 'running', progress: {images_done: 1}});
    await settle();
    assert.equal(state.timers.size, 0);
    assert.deepEqual(state.requests, ['/api/v1/retention/preview']);
});

test('failed calculations stop polling and offer an explicit retry', async () => {
    const state = harness([{status: 'failed'}]);
    state.context.retentionPreview(false);
    await settle();
    assert.match(state.out.textContent, /Refresh deletion preview to retry/);
    assert.equal(state.timers.size, 0);
});
