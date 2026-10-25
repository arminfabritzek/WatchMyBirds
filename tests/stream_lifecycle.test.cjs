const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const template = fs.readFileSync(path.join(__dirname, '../templates/stream.html'), 'utf8');
// Plain index scanning, not a tag regexp: this only has to split a known
// local template, and every regexp shape trips CodeQL's bad-tag-filter rule.
function extractScripts(html) {
    const blocks = [];
    for (let cursor = 0; ; ) {
        const open = html.indexOf('<script', cursor);
        if (open < 0) break;
        const bodyStart = html.indexOf('>', open);
        if (bodyStart < 0) break;
        const close = html.indexOf('</script', bodyStart);
        if (close < 0) break;
        blocks.push(html.slice(bodyStart + 1, close));
        cursor = close + '</script'.length;
    }
    return blocks;
}

const scripts = extractScripts(template);

function setup(relay = true) {
    const elements = {};
    function element() {
        const listeners = {};
        const classes = new Set();
        let source = '';
        const sourceWrites = [];
        return {style: {setProperty(key, value) { this[key] = value; }}, dataset: {},
            naturalWidth: 0, naturalHeight: 0, complete: false, listeners, sourceWrites,
            get src() { return source; },
            set src(value) { source = value; sourceWrites.push(value); },
            lastElementChild: {}, classList: {
                add: name => classes.add(name), remove: name => classes.delete(name),
                contains: name => classes.has(name)},
            addEventListener(name, callback) { listeners[name] = callback; },
            removeAttribute(name) { this[name] = ''; }, getContext() { return null; }};
    }
    for (const id of ['go2rtc-frame', 'video-feed-fallback', 'stream-loading-static', 'stream-loading-status']) {
        elements[id] = element();
    }
    const embed = element();
    const container = element();
    container.parentElement = {clientWidth: 1200};
    const timers = new Map();
    let sequence = 0;
    const documentEvents = {};
    const windowEvents = {};
    const document = {hidden: false, fullscreenElement: null, getElementById: id => elements[id],
        querySelector: selector => selector === '.stream-embed' ? embed : container,
        addEventListener: (name, cb) => documentEvents[name] = cb};
    const window = {location: {hostname: 'camera.local'}, innerHeight: 1000,
        addEventListener: (name, cb) => windowEvents[name] = cb,
        setInterval: () => ++sequence, clearInterval() {}};
    const probes = [];
    const context = {window, document, console, AbortController, Date,
        setTimeout: cb => { timers.set(++sequence, cb); return sequence; },
        clearTimeout: id => timers.delete(id),
        fetch: async () => ({ok: true, json: async () => ({healthy: true})}),
        getComputedStyle: () => ({maxWidth: '1200px'}),
        Image: function () { probes.push(this); }};
    const source = scripts.find(script => script.includes('function reconnectStream'))
        .replace(/{% if stream_source_mode[\s\S]*?%}([\s\S]*?){% else %}([\s\S]*?){% endif %}/,
            (_, yes, no) => relay ? yes : no);
    vm.runInNewContext(source, context);
    return {elements, windowEvents, documentEvents, document, probes, embed, context,
        flush() { const pending = [...timers.values()]; timers.clear(); pending.forEach(cb => cb()); }};
}

test('initial pageshow and visibility during preflight cannot load the parent URL', async () => {
    const state = setup();
    state.windowEvents.pageshow({persisted: false});
    state.documentEvents.visibilitychange();
    state.flush();
    assert.equal(state.elements['go2rtc-frame'].src, '');
    await new Promise(resolve => setImmediate(resolve));
    assert.match(state.elements['go2rtc-frame'].src, /:1984\/stream.html\?src=camera/);
    assert.ok(state.elements['stream-loading-static'].classList.contains('is-hidden'));
});

test('restored go2rtc page reconnects to the configured player', async () => {
    const state = setup();
    await new Promise(resolve => setImmediate(resolve));
    state.elements['go2rtc-frame'].src = 'about:blank';
    state.windowEvents.pageshow({persisted: true});
    state.flush();
    assert.match(state.elements['go2rtc-frame'].src, /:1984\/stream.html/);
});

test('go2rtc visibility changes preserve the loaded player and fullscreen', async () => {
    const state = setup();
    await new Promise(resolve => setImmediate(resolve));
    const frame = state.elements['go2rtc-frame'];
    assert.equal(frame.sourceWrites.length, 1);
    for (const fullscreen of [null, frame]) {
        state.document.fullscreenElement = fullscreen;
        for (const hidden of [false, true, false]) {
            state.document.hidden = hidden;
            state.documentEvents.visibilitychange();
            state.flush();
        }
    }
    assert.equal(frame.sourceWrites.length, 1, 'the embedded player owns visibility recovery');
});

test('a restored page does not navigate a fullscreen player', async () => {
    const state = setup();
    await new Promise(resolve => setImmediate(resolve));
    const frame = state.elements['go2rtc-frame'];
    state.document.fullscreenElement = frame;
    state.windowEvents.pageshow({persisted: true});
    state.flush();
    assert.equal(frame.sourceWrites.length, 1);
});

test('fullscreen entered after scheduling a reconnect cancels the navigation', async () => {
    const state = setup();
    await new Promise(resolve => setImmediate(resolve));
    const frame = state.elements['go2rtc-frame'];
    state.windowEvents.pageshow({persisted: true});
    state.document.fullscreenElement = frame;
    state.flush();
    assert.equal(frame.sourceWrites.length, 1);
});

test('MJPEG reconnects after returning to a visible page', () => {
    const state = setup(false);
    const fallback = state.elements['video-feed-fallback'];
    state.document.hidden = true;
    state.documentEvents.visibilitychange();
    state.flush();
    assert.equal(fallback.sourceWrites.length, 1);
    state.document.hidden = false;
    state.documentEvents.visibilitychange();
    state.flush();
    assert.equal(fallback.sourceWrites.length, 2);
    assert.match(fallback.src, /^\/video_feed\?reconnect=/);
});

test('MJPEG overlay waits for an image and reconnect is cancelled on pagehide', () => {
    const state = setup(false);
    const fallback = state.elements['video-feed-fallback'];
    const overlay = state.elements['stream-loading-static'];
    fallback.listeners.load();
    assert.equal(overlay.classList.contains('is-hidden'), false);
    fallback.naturalWidth = 640;
    fallback.listeners.load();
    assert.equal(overlay.classList.contains('is-hidden'), true);
    state.windowEvents.pageshow({persisted: true});
    state.flush();
    assert.match(fallback.src, /^\/video_feed\?reconnect=/);
    assert.equal(overlay.classList.contains('is-hidden'), false);
    const source = fallback.src;
    state.documentEvents.visibilitychange();
    state.windowEvents.pagehide();
    state.flush();
    assert.equal(fallback.src, source);
});

test('snapshot sets a 4:3 aspect ratio even when MJPEG never loads', () => {
    const state = setup();
    vm.runInNewContext(scripts.find(script => script.includes('function resizeStreamPane')), state.context);
    assert.match(state.probes[0].src, /^\/api\/snapshot\?aspect_probe=/);
    Object.assign(state.probes[0], {naturalWidth: 640, naturalHeight: 480});
    state.probes[0].onload();
    assert.equal(state.embed.style['--stream-aspect'], String(4 / 3));
});

test('aspect probe retries once when the snapshot endpoint has no frame yet', () => {
    const state = setup();
    vm.runInNewContext(scripts.find(script => script.includes('function resizeStreamPane')), state.context);
    assert.equal(state.probes.length, 1);
    state.probes[0].onerror();
    state.flush();
    assert.equal(state.probes.length, 2, 'a 503 must schedule exactly one retry');
    Object.assign(state.probes[1], {naturalWidth: 640, naturalHeight: 480});
    state.probes[1].onload();
    assert.equal(state.embed.style['--stream-aspect'], String(4 / 3));
    state.probes[1].onerror();
    state.flush();
    assert.equal(state.probes.length, 2, 'no further probe once an aspect is known');
});
