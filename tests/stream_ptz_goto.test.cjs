const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const template = fs.readFileSync(path.join(__dirname, '../templates/stream.html'), 'utf8');
const gallery = fs.readFileSync(path.join(__dirname, '../assets/js/gallery_utils.js'), 'utf8');

function between(source, start, end) {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first);
    assert.ok(first >= 0 && last > first);
    return source.slice(first, last);
}

const gotoSource = between(template, 'async function gotoPreset(token)', '// Auto-Cam toggle:');
const authSource = between(gallery, 'function isAuthRedirect(resp)', '// Return a same-origin');
const loginSource = between(gallery, 'function redirectToLogin()', '/* =========================================');

function response(status = 200, data = {status: 'success'}, overrides = {}) {
    return {status, ok: status >= 200 && status < 300,
        headers: {get: () => 'application/json'},
        json: async () => data, ...overrides};
}

function setup(reply) {
    const calls = [];
    const toasts = [];
    const context = {
        cameraId: 0, overviewPreset: 'Preset022', aimArmed: false,
        settling: true, returnDeadline: 12000, stopped: 0, painted: 0,
        csrfHeaders: headers => ({...headers, 'X-CSRFToken': 'test-token'}),
        stopCountdownTicker() { context.stopped++; },
        paintStatusPill() { context.painted++; },
        fetch: async (url, options) => {
            calls.push({url, options});
            return typeof reply === 'function' ? reply() : reply;
        },
        window: {
            wmToast: (...args) => toasts.push(args),
            location: {pathname: '/stream', search: '?ptz_debug=1', hash: '', href: ''},
        },
    };
    vm.runInNewContext(authSource + loginSource + gotoSource, context);
    return {context, calls, toasts};
}

function assertCountdownUnchanged(context) {
    assert.equal(context.settling, true);
    assert.equal(context.returnDeadline, 12000);
    assert.equal(context.stopped, 0);
    assert.equal(context.painted, 0);
}

test('overview only clears the countdown after an accepted response', async () => {
    let resolve;
    const pending = new Promise(done => { resolve = done; });
    const {context, calls, toasts} = setup(() => pending);
    const request = context.gotoPreset('Preset022');
    assertCountdownUnchanged(context);
    assert.equal(calls[0].url, '/api/v1/cameras/0/ptz/goto');
    assert.equal(calls[0].options.method, 'POST');
    assert.equal(calls[0].options.headers['X-CSRFToken'], 'test-token');
    assert.deepEqual(JSON.parse(calls[0].options.body), {preset_token: 'Preset022'});
    resolve(response());
    await request;
    assert.equal(context.settling, false);
    assert.equal(context.returnDeadline, 0);
    assert.equal(context.stopped, 1);
    assert.equal(context.painted, 1);
    assert.equal(toasts.length, 0);
});

for (const status of [500, 403]) {
    test(`HTTP ${status} is visible and preserves the countdown`, async () => {
        const {context, toasts} = setup(response(status, null, {
            headers: {get: () => 'text/html'},
            json: async () => { throw new SyntaxError('HTML error page'); },
        }));
        await context.gotoPreset('Preset022');
        assertCountdownUnchanged(context);
        assert.equal(toasts.length, 1);
        assert.match(toasts[0][0], new RegExp(`HTTP ${status}`));
        assert.equal(toasts[0][1], 'error');
        assert.equal(context.window.location.href, '');
    });
}

for (const [name, reply] of [
    ['login redirect', response(200, null, {
        redirected: true, url: 'http://station.local/login?next=/api/v1/cameras/0/ptz/goto',
        headers: {get: () => 'text/html'},
    })],
    ['unauthorized response', response(401)],
]) {
    test(`${name} asks for login without treating the preset as accepted`, async () => {
        const {context, toasts} = setup(reply);
        await context.gotoPreset('Preset022');
        assertCountdownUnchanged(context);
        assert.equal(context.window.location.href, '/login?next=%2Fstream%3Fptz_debug%3D1');
        assert.deepEqual(toasts[0], ['Session expired. Please log in.', 'error']);
    });
}

for (const [name, reply] of [
    ['network failure', () => { throw new TypeError('Failed to fetch'); }],
    ['invalid JSON', response(200, null, {json: async () => { throw new SyntaxError(); }})],
    ['application error', response(200, {status: 'error'})],
    ['missing acknowledgement', response(200, {})],
]) {
    test(`${name} shows uncertainty without clearing the countdown`, async () => {
        const {context, toasts} = setup(reply);
        await context.gotoPreset('Preset022');
        assertCountdownUnchanged(context);
        assert.equal(toasts.length, 1);
        assert.match(toasts[0][0], /Could not confirm.*Check the live view/);
        assert.equal(toasts[0][1], 'error');
    });
}
