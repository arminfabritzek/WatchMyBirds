const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const source = fs.readFileSync('assets/js/training_export.js', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
function harness(replies, autoJob = 'job') {
    const progress = {textContent: ''};
    const node = {dataset: {exportJob: 'job', jobState: 'running'}, querySelector: () => progress};
    const timers = new Map(), events = {}, replaced = [], assigned = [];
    const context = vm.createContext({URL,
        document: {currentScript: {dataset: {autoJob}}, querySelectorAll: () => [node]},
        window: {location: {href: 'http://localhost/admin/canonical-dataset?job=job', replace: url => replaced.push(String(url)), assign: url => assigned.push(url)}, history: {replaceState: () => {}}, addEventListener: (name, fn) => {events[name] = fn;}},
        fetch: async () => { const body = await replies.shift(); return {ok: true, json: async () => body}; },
        setTimeout: fn => { timers.set(1,fn); return 1; }, clearTimeout: id => timers.delete(id)
    });
    vm.runInContext(source,context);
    return {progress,events,timers,replaced,assigned};
}

test('progress shows bounded current-step bytes and repeats status polling', async () => {
    const state=harness([{job_id:'job',status:'running',phase:'archives',processed:1000,total:2000,part_count:1,bytes_done:1048576,bytes_total:2097152}]);
    await settle();
    assert.match(state.progress.textContent,/1000 \/ 2000 photos/);
    assert.match(state.progress.textContent,/1.0 \/ about 2.0 MiB/);
    assert.equal(state.timers.size,1);
});

test('paused preparation downloads its first available unconfirmed part once', async () => {
    const state=harness([{job_id:'job',status:'paused',parts:[{export_id:'old',available:0,received_at:'received'},{export_id:'ready',available:1,received_at:null}]}]);
    await settle();
    assert.equal(state.replaced.length,1);
    assert.equal(new URL(state.replaced[0]).searchParams.get('download'),'ready');
    assert.equal(state.timers.size,0);
});

test('leaving the page prevents late status replies and downloads', async () => {
    let resolve;
    const state=harness([new Promise(done => {resolve=done;})]);
    state.events.pagehide();
    resolve({job_id:'job',status:'complete',parts:[{export_id:'ready',available:1}]});
    await settle();
    assert.deepEqual(state.replaced,[]);
    assert.equal(state.timers.size,0);
});

test('failed status reads retry without starting an export', async () => {
    const state=harness([Promise.reject(new Error('unavailable'))]);
    await settle();
    assert.match(state.progress.textContent,/Retrying/);
    assert.equal(state.timers.size,1);
    assert.deepEqual(state.replaced,[]);
});
