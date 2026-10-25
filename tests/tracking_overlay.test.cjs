const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../assets/js/tracking_overlay.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function status(auto = true, overlay = true) {
    return {overlay_enabled: overlay, auto_ptz: {
        configured_enabled: auto, enabled: auto, state: 'tracking',
        last_bbox: [0.2, 0.2, 0.1, 0.1], seconds_until_return: null,
    }};
}

function setup(initial, reply = async () => ({ok: true, json: async () => status()})) {
    const calls = [];
    const timers = new Map();
    const documentEvents = {};
    const windowEvents = {};
    const canvas = {hidden: false, getContext: () => ({clearRect() {}})};
    let sequence = 0;
    const document = {hidden: false, getElementById: () => canvas,
        addEventListener(name, callback) { documentEvents[name] = callback; }};
    const window = {wmbPtzStatus: initial, requestAnimationFrame() {},
        addEventListener(name, callback) { windowEvents[name] = callback; }};
    vm.runInNewContext(source, {document, window, performance: {now: () => 0},
        setTimeout(callback, delay) { timers.set(++sequence, {callback, delay}); return sequence; },
        clearTimeout(id) { timers.delete(id); },
        fetch(url, options) { calls.push({url, options}); return reply(); },
    });
    return {calls, timers, document, canvas,
        publish(data) { windowEvents['wmb:ptz-status']({detail: data}); },
        visibility(hidden) { document.hidden = hidden; documentEvents.visibilitychange(); },
        async nextPoll() {
            const [id, timer] = timers.entries().next().value;
            timers.delete(id);
            timer.callback();
            await tick();
        },
    };
}

test('the overlay waits for the Live snapshot without issuing an extra request', () => {
    const state = setup();
    assert.equal(state.calls.length, 0);
    assert.equal(state.timers.size, 0);
    assert.equal(state.canvas.hidden, true);
    state.publish(status(false));
    assert.equal(state.timers.size, 0);
});

for (const [name, data] of [
    ['Auto Cam off', status(false)],
    ['overlay off', status(true, false)],
    ['no camera', {overlay_enabled: true, auto_ptz: {}}],
    ['persisted off overrides the legacy alias', {
        ...status(), auto_ptz: {...status().auto_ptz, configured_enabled: false},
    }],
]) {
    test(`${name} stops the fast loop`, async () => {
        const state = setup(status());
        assert.equal(state.timers.size, 1);
        state.publish(data);
        await tick();
        assert.equal(state.calls.length, 0);
        assert.equal(state.timers.size, 0);
        assert.equal(state.canvas.hidden, true);
    });
}

test('visible tracking polls every 200 ms after the previous request finishes', async () => {
    let resolve;
    const pending = new Promise(done => { resolve = done; });
    const state = setup(status(), () => pending);
    assert.equal([...state.timers.values()][0].delay, 200);
    await state.nextPoll();
    assert.equal(state.calls.length, 1);
    assert.equal(state.calls[0].url, '/api/v1/ptz/auto/status');
    assert.equal(state.calls[0].options.credentials, 'same-origin');
    assert.equal(state.timers.size, 0);
    state.publish(status());
    assert.equal(state.timers.size, 0, 'a slow request cannot overlap another overlay poll');
    resolve({ok: true, json: async () => status()});
    await tick();
    assert.equal(state.timers.size, 1);
    assert.equal([...state.timers.values()][0].delay, 200);
});

test('hidden tabs stop and wait for a fresh snapshot before resuming', () => {
    const state = setup(status());
    state.visibility(true);
    assert.equal(state.timers.size, 0);
    state.publish(status());
    assert.equal(state.timers.size, 0);
    state.visibility(false);
    assert.equal(state.timers.size, 0);
    state.publish(status());
    assert.equal(state.timers.size, 1);
    assert.equal(state.canvas.hidden, false);
});

test('a delayed response cannot restart tracking after Auto Cam was switched off', async () => {
    let resolve;
    const state = setup(status(), () => new Promise(done => { resolve = done; }));
    await state.nextPoll();
    state.publish(status(false));
    resolve({ok: true, json: async () => status()});
    await tick();
    assert.equal(state.timers.size, 0);
    assert.equal(state.canvas.hidden, true);
});

test('an old failed request cannot disable a newer active snapshot', async () => {
    let reject;
    const state = setup(status(), () => new Promise((_, failed) => { reject = failed; }));
    await state.nextPoll();
    state.publish(status());
    reject(new TypeError('old request failed'));
    await tick();
    assert.equal(state.timers.size, 1);
    assert.equal(state.canvas.hidden, false);
});

for (const [name, reply] of [
    ['HTTP failure', async () => ({ok: false})],
    ['network failure', async () => { throw new TypeError('Failed to fetch'); }],
]) {
    test(`${name} waits for the next slow snapshot`, async () => {
        const state = setup(status(), reply);
        await state.nextPoll();
        assert.equal(state.timers.size, 0);
        assert.equal(state.canvas.hidden, true);
        state.publish(status());
        assert.equal(state.timers.size, 1);
    });
}
