(function () {
    'use strict';

    var STATUS_AFTER_MS = 1000;
    var NOTICE_AFTER_MS = 4000;
    // A response that turns out to be a file never replaces the page, and no
    // event reports that. Give up eventually so the element works again.
    var GIVE_UP_AFTER_MS = 45000;
    var NAME_LIMIT = 24;

    var root = document.documentElement;
    var notice = document.getElementById('navNotice');
    var noticeText = document.getElementById('navNoticeText');
    var noticeElapsed = document.getElementById('navNoticeElapsed');
    var announcement = document.getElementById('navAnnouncement');
    var pending = null;
    var timers = [];
    var ticking = null;

    function statusBar(top, bottom) {
        var ticker = window.wmbLedTicker;
        if (ticker && typeof ticker.setNotice === 'function') ticker.setNotice(top, bottom);
    }

    function reset() {
        timers.forEach(clearTimeout);
        timers = [];
        if (ticking !== null) clearInterval(ticking);
        ticking = null;
        if (pending) {
            pending.element.classList.remove('is-opening');
            if (pending.previousBusy === null) {
                pending.element.removeAttribute('aria-busy');
            } else {
                pending.element.setAttribute('aria-busy', pending.previousBusy);
            }
        }
        pending = null;
        root.classList.remove('nav-pending');
        if (notice) notice.hidden = true;
        if (announcement) announcement.textContent = '';
        statusBar(null);
    }

    function nameOf(element) {
        var lines = String((element && element.textContent) || '').split('\n');
        for (var index = 0; index < lines.length; index += 1) {
            var line = lines[index].trim();
            if (line) return line.length <= NAME_LIMIT ? line : '';
        }
        return '';
    }

    function begin(element, name, submitter) {
        reset();
        var startedAt = Date.now();
        pending = {
            element: element,
            previousBusy: element.getAttribute('aria-busy'),
            submitter: submitter,
            edited: false,
        };
        element.classList.add('is-opening');
        element.setAttribute('aria-busy', 'true');
        root.classList.add('nav-pending');
        timers.push(setTimeout(function () {
            statusBar('OPENING', name || 'PAGE');
        }, STATUS_AFTER_MS));
        timers.push(setTimeout(function () {
            function updateElapsed() {
                if (noticeElapsed) {
                    noticeElapsed.textContent = ' · ' + Math.floor((Date.now() - startedAt) / 1000) + ' s';
                }
            }
            updateElapsed();
            var message = 'Still loading' + (name ? ' ' + name : '');
            if (noticeText) noticeText.textContent = message;
            if (notice) notice.hidden = false;
            if (announcement) announcement.textContent = message;
            ticking = setInterval(updateElapsed, 1000);
        }, NOTICE_AFTER_MS));
        timers.push(setTimeout(reset, GIVE_UP_AFTER_MS));
    }

    function pageUrl(href, target) {
        if (target && target.toLowerCase() !== '_self') return null;
        var url;
        try {
            url = new URL(href, window.location.href);
        } catch (_error) {
            return null;
        }
        if (url.protocol !== window.location.protocol || url.origin !== window.location.origin) return null;
        return /(^|\/)download(\/|$)/.test(url.pathname) ? null : url;
    }

    document.addEventListener('click', function (event) {
        if (event.defaultPrevented || event.button !== 0) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        var link = event.target && event.target.closest ? event.target.closest('a[href]') : null;
        if (!link || typeof link.href !== 'string' || link.hasAttribute('download') || link.dataset.navFeedback === 'off') return;
        var url = pageUrl(link.href, link.target);
        var here = window.location;
        if (!url || (url.pathname === here.pathname && url.search === here.search && url.href.includes('#'))) return;
        if (pending && pending.element === link) {
            event.preventDefault();
            return;
        }
        begin(link, nameOf(link));
    });

    document.addEventListener('submit', function (event) {
        if (event.defaultPrevented) return;
        var form = event.target;
        var submitter = event.submitter;
        var target = submitter && submitter.hasAttribute('formtarget') ? submitter.formTarget : form.target;
        var method = submitter && submitter.hasAttribute('formmethod') ? submitter.formMethod : form.method;
        var action = submitter && submitter.hasAttribute('formaction') ? submitter.formAction : form.action;
        if (form.dataset.navFeedback === 'off' || method === 'dialog' || !pageUrl(action, target)) return;
        if (pending && pending.element === form && pending.submitter === submitter && !pending.edited) {
            event.preventDefault();
            return;
        }
        begin(form, nameOf(submitter), submitter);
    });

    document.addEventListener('input', function (event) {
        // An edited form can submit new values while the previous request is pending.
        if (pending && event.target.form === pending.element) pending.edited = true;
    });

    // Also fires when the page returns from the back/forward cache.
    window.addEventListener('pageshow', reset);

    // Some touch browsers apply :active only when a touch listener exists.
    document.addEventListener('touchstart', function () {}, { passive: true });
})();
