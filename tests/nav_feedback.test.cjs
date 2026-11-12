const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

const source = fs.readFileSync('assets/js/nav_feedback.js', 'utf8');
const ORIGIN = 'http://example.test:8050';

function classes() {
    const names = new Set();
    return {
        add: (...items) => items.forEach(item => names.add(item)),
        remove: (...items) => items.forEach(item => names.delete(item)),
        contains: name => names.has(name),
    };
}

function element(extra = {}) {
    const attributes = {...(extra.attributes || {})};
    return {
        classList: classes(),
        dataset: extra.dataset || {},
        textContent: extra.textContent || '',
        hidden: false,
        getAttribute: name => (name in attributes ? attributes[name] : null),
        hasAttribute: name => name in attributes,
        setAttribute: (name, value) => { attributes[name] = String(value); },
        removeAttribute: name => { delete attributes[name]; },
        attributes,
        ...extra.props,
    };
}

function link(path, extra = {}) {
    const node = element(extra);
    node.href = new URL(path, ORIGIN).href;
    node.target = node.getAttribute('target') || '';
    node.closest = selector => (selector === 'a[href]' ? node : null);
    return node;
}

function harness(page = '/gallery') {
    const root = element();
    const notice = element();
    notice.hidden = true;
    const noticeText = element();
    const noticeElapsed = element();
    const announcement = element();
    const listeners = {document: {}, window: {}};
    const timers = [];
    const tickerCalls = [];
    let now = 1000;
    const context = vm.createContext({
        URL,
        Date: {now: () => now},
        document: {
            documentElement: root,
            getElementById: id => ({
                navNotice: notice,
                navNoticeText: noticeText,
                navNoticeElapsed: noticeElapsed,
                navAnnouncement: announcement,
            })[id] || null,
            addEventListener: (name, handler) => { listeners.document[name] = handler; },
        },
        window: {
            location: new URL(ORIGIN + page),
            addEventListener: (name, handler) => { listeners.window[name] = handler; },
            wmbLedTicker: {setNotice: (top, bottom) => tickerCalls.push([top, bottom])},
        },
        setTimeout: (fn, ms) => { timers.push({fn, due: now + ms, live: true}); return timers.length; },
        clearTimeout: id => { if (timers[id - 1]) timers[id - 1].live = false; },
        setInterval: fn => { timers.push({fn, live: true, repeat: true}); return timers.length; },
        clearInterval: id => { if (timers[id - 1]) timers[id - 1].live = false; },
    });
    vm.runInContext(source, context);
    const click = (target, extra = {}) => {
        const event = {
            target,
            button: 0,
            defaultPrevented: false,
            prevented: false,
            preventDefault() { this.prevented = true; },
            ...extra,
        };
        listeners.document.click(event);
        return event;
    };
    const submit = (form, extra = {}) => {
        const event = {
            target: form,
            submitter: null,
            defaultPrevented: false,
            prevented: false,
            preventDefault() { this.prevented = true; },
            ...extra,
        };
        listeners.document.submit(event);
        return event;
    };
    const run = ms => {
        now += ms;
        for (const timer of [...timers]) {
            if (timer.live && !timer.repeat && timer.due <= now) { timer.live = false; timer.fn(); }
        }
        timers.filter(timer => timer.live && timer.repeat).forEach(timer => timer.fn());
    };
    const pending = () => root.classList.contains('nav-pending');
    return {notice, noticeText, noticeElapsed, announcement, listeners, tickerCalls, click, submit, run, pending};
}

test('a page link marks the page and the clicked element as loading', () => {
    const page = harness();
    const species = link('/species', {textContent: '\n   Species\n'});
    const event = page.click(species);
    assert.equal(event.prevented, false);
    assert.equal(page.pending(), true);
    assert.equal(species.classList.contains('is-opening'), true);
    assert.equal(species.attributes['aria-busy'], 'true');
});

test('repeated clicks on the element being opened are swallowed', () => {
    const page = harness();
    const species = link('/species');
    page.click(species);
    assert.equal(page.click(species).prevented, true);
    assert.equal(page.click(species).prevented, true);
    assert.equal(page.pending(), true);
});

test('a click on another link moves the feedback there', () => {
    const page = harness();
    const species = link('/species');
    const report = link('/analytics');
    page.click(species);
    const event = page.click(report);
    assert.equal(event.prevented, false);
    assert.equal(species.classList.contains('is-opening'), false);
    assert.equal(species.attributes['aria-busy'], undefined);
    assert.equal(report.classList.contains('is-opening'), true);
});

for (const [name, target, extra] of [
    ['a download link', link('/api/image/download/7', {attributes: {download: ''}}), {}],
    ['a file route without the download attribute', link('/api/canonical-dataset/exports/abc/download'), {}],
    ['a link opening a new tab', link('/species', {attributes: {target: '_blank'}}), {}],
    ['a link that opted out', link('/species', {dataset: {navFeedback: 'off'}}), {}],
    ['another site', link('https://en.wikipedia.org/wiki/Great_tit'), {}],
    ['a blob with the same origin', link('blob:' + ORIGIN + '/example'), {}],
    ['an anchor on the same page', link('/gallery#detection-4'), {}],
    ['a link used as a button', link('/gallery#'), {}],
    ['a click with a modifier key', link('/species'), {ctrlKey: true}],
    ['a click with the command key', link('/species'), {metaKey: true}],
    ['a middle click', link('/species'), {button: 1}],
    ['a click another script already handled', link('/species'), {defaultPrevented: true}],
    ['a click outside any link', {closest: () => null}, {}],
]) {
    test(`${name} shows no loading feedback`, () => {
        const page = harness();
        const event = page.click(target, extra);
        assert.equal(page.pending(), false);
        assert.equal(event.prevented, false);
    });
}

test('feedback escalates: status bar after one second, notice after four', () => {
    const page = harness();
    page.click(link('/species', {textContent: '\n  Kohlmeise\n  Parus major\n'}));
    assert.deepEqual(page.tickerCalls, [[null, undefined]]);
    assert.equal(page.notice.hidden, true);
    page.run(1000);
    assert.deepEqual(page.tickerCalls.at(-1), ['OPENING', 'Kohlmeise']);
    assert.equal(page.notice.hidden, true);
    page.run(3000);
    assert.equal(page.notice.hidden, false);
    assert.equal(page.noticeText.textContent, 'Still loading Kohlmeise');
    assert.equal(page.noticeElapsed.textContent, ' · 4 s');
    assert.equal(page.announcement.textContent, 'Still loading Kohlmeise');
    page.run(2000);
    assert.equal(page.noticeElapsed.textContent, ' · 6 s');
    assert.equal(page.announcement.textContent, 'Still loading Kohlmeise');
});

test('a link without a short name still gets a generic notice', () => {
    const page = harness();
    page.click(link('/species', {textContent: 'x'.repeat(80)}));
    page.run(4000);
    assert.deepEqual(page.tickerCalls.at(-1), ['OPENING', 'PAGE']);
    assert.equal(page.noticeText.textContent, 'Still loading');
});

test('showing the page again clears every trace, also after going back', () => {
    const page = harness();
    const species = link('/species', {attributes: {'aria-busy': 'false'}});
    page.click(species);
    page.run(4000);
    page.listeners.window.pageshow({persisted: true});
    assert.equal(page.pending(), false);
    assert.equal(page.notice.hidden, true);
    assert.equal(page.announcement.textContent, '');
    assert.equal(species.classList.contains('is-opening'), false);
    assert.equal(species.attributes['aria-busy'], 'false');
    assert.deepEqual(page.tickerCalls.at(-1), [null, undefined]);
    assert.equal(page.click(species).prevented, false);
});

test('feedback gives up after 45 seconds so a click is never lost for good', () => {
    const page = harness();
    const species = link('/species');
    page.click(species);
    page.run(45000);
    assert.equal(page.pending(), false);
    assert.equal(page.click(species).prevented, false);
});

test('a submitted form gets the same feedback and no double submit', () => {
    const page = harness();
    const form = element({props: {action: ORIGIN + '/species', method: 'get'}});
    const submitter = element({textContent: 'Apply'});
    const first = page.submit(form, {submitter});
    assert.equal(first.prevented, false);
    assert.equal(page.pending(), true);
    assert.equal(form.attributes['aria-busy'], 'true');
    assert.equal(page.submit(form, {submitter}).prevented, true);
    page.listeners.document.input({target: {form}});
    assert.equal(page.submit(form, {submitter}).prevented, false);
    assert.equal(page.submit(form, {submitter: element({textContent: 'Reset'})}).prevented, false);
});

for (const [name, form, extra] of [
    ['a form a script handles itself', element(), {defaultPrevented: true}],
    ['a form opening a new tab', element({props: {target: '_blank'}}), {}],
    ['a form with a target override', element(), {
        submitter: element({attributes: {formtarget: '_blank'}, props: {formTarget: '_blank'}}),
    }],
    ['a form with a download action override', element(), {
        submitter: element({attributes: {formaction: '/download/archive'}, props: {formAction: ORIGIN + '/download/archive'}}),
    }],
    ['a dialog form', element({props: {method: 'dialog'}}), {}],
    ['a form submitting to another site', element({props: {action: 'https://example.org'}}), {}],
    ['a form that opted out', element({dataset: {navFeedback: 'off'}}), {}],
]) {
    test(`${name} shows no loading feedback`, () => {
        const page = harness();
        page.submit(form, extra);
        assert.equal(page.pending(), false);
    });
}

test('touch devices get pressed states', () => {
    const page = harness();
    assert.equal(typeof page.listeners.document.touchstart, 'function');
});
