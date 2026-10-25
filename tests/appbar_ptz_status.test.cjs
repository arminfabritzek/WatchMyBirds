const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const template = fs.readFileSync(path.join(__dirname, '../templates/partials/appbar.html'), 'utf8');
const first = template.indexOf('(function () {', template.indexOf('// ─── Auto-Cam status:'));
const last = template.indexOf('    function toggleNav()', first);
assert.ok(first >= 0 && last > first);
const source = template.slice(first, last);
const tick = () => new Promise(resolve => setImmediate(resolve));

function setup({hidden = false, authenticated = true, streamOwnsStatus = false} = {}) {
    const calls = [];
    const intervals = [];
    const events = {};
    const button = {dataset: {}, setAttribute() {}};
    const document = {hidden, getElementById: () => button,
        addEventListener(name, callback) { events[name] = callback; }};
    const window = {__streamPageOwnsAutoCam: streamOwnsStatus};
    vm.runInNewContext(source.replace(
        "{% if session.get('authenticated') %}true{% else %}false{% endif %}",
        String(authenticated),
    ), {document, window, performance: {now: () => 0},
        setInterval(callback, delay) { intervals.push({callback, delay}); },
        fetch(url) {
            calls.push(url);
            return Promise.resolve({ok: true, json: async () => ({
                status: 'success', auto_ptz: {camera_id: 0, configured_enabled: false},
            })});
        },
    });
    return {calls, intervals, document, events};
}

test('the read-only status mirror skips hidden tabs and refreshes on return', async () => {
    const state = setup();
    await tick();
    assert.equal(state.calls.length, 1);
    assert.equal(state.intervals[0].delay, 5000);
    state.document.hidden = true;
    state.intervals[0].callback();
    state.events.visibilitychange();
    assert.equal(state.calls.length, 1);
    state.document.hidden = false;
    state.events.visibilitychange();
    await tick();
    assert.equal(state.calls.length, 2);
});

test('an initially hidden page waits until it becomes visible', () => {
    const state = setup({hidden: true});
    assert.equal(state.calls.length, 0);
    state.document.hidden = false;
    state.events.visibilitychange();
    assert.equal(state.calls.length, 1);
});

for (const options of [{authenticated: false}, {streamOwnsStatus: true}]) {
    test(`the appbar does not poll when ${JSON.stringify(options)}`, () => {
        const state = setup(options);
        assert.equal(state.calls.length, 0);
        assert.equal(state.intervals.length, 0);
    });
}
