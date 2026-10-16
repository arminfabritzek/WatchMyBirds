(function () {
    'use strict';
    const root = document.querySelector('[data-report-state]');
    if (!root) return;
    const progress = document.getElementById('report-progress');
    const modals = document.querySelector('[data-report-modals]');
    let timer;
    let requestId = 0;
    let waiting = root.dataset.reportState !== 'ready' && root.dataset.reportState !== 'error';

    function interactionOpen() {
        return document.querySelector('.modal.show, dialog[open]');
    }

    function showLocalTime() {
        root.querySelectorAll('[data-report-time]').forEach(function (node) {
            const stamp = new Date(node.dateTime);
            node.textContent = 'as of ' + stamp.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
            node.title = stamp.toLocaleString();
        });
    }

    function setOutdated(visible) {
        root.querySelectorAll('[data-report-outdated]').forEach(function (notice) {
            notice.hidden = !visible;
        });
    }

    function setRefreshing(refreshing) {
        root.querySelectorAll('[data-report-refresh]').forEach(function (button) {
            button.disabled = refreshing;
            button.textContent = refreshing ? 'Refreshing…' : 'Refresh';
        });
    }

    function stop() {
        requestId++;
        waiting = false;
        window.clearTimeout(timer);
        setRefreshing(false);
    }

    async function applyReport(request) {
        const url = new URL(window.location.href);
        url.searchParams.set('content', '1');
        const response = await fetch(url, {cache: 'no-store'});
        if (!response.ok || response.redirected) throw new Error('Report unavailable');
        const page = new DOMParser().parseFromString(await response.text(), 'text/html');
        const nextRoot = page.querySelector('[data-report-state]');
        const nextReport = page.querySelector('[data-current-report]');
        const nextModals = page.querySelector('[data-report-modals]');
        if (!nextRoot || !nextReport || !nextModals) throw new Error('Report unavailable');
        if (!waiting || request !== requestId || interactionOpen() || nextRoot.dataset.reportState !== 'ready') return false;
        const previous = root.querySelector('[data-current-report]');
        const x = window.scrollX;
        const y = window.scrollY;
        const offsets = previous ? Array.from(previous.querySelectorAll('.station-report__table-scroll'), node => node.scrollLeft) : [];
        const details = previous ? Array.from(previous.querySelectorAll('details[data-report-details][open]'), node => node.dataset.reportDetails) : [];
        modals.querySelectorAll('.gallery-modal').forEach(function (modal) {
            if (modal._wmBirdEditorDispose) modal._wmBirdEditorDispose();
            if (window.bootstrap) {
                const instance = window.bootstrap.Modal.getInstance(modal);
                if (instance) instance.dispose();
            }
        });
        modals.replaceChildren(...nextModals.childNodes);
        if (previous) previous.replaceWith(nextReport);
        else root.appendChild(nextReport);
        nextReport.querySelectorAll('.station-report__table-scroll').forEach(function (node, index) {
            node.scrollLeft = offsets[index] || 0;
        });
        nextReport.querySelectorAll('details[data-report-details]').forEach(function (node) {
            node.open = details.includes(node.dataset.reportDetails);
        });
        root.dataset.reportRevision = nextRoot.dataset.reportRevision;
        root.dataset.reportState = 'ready';
        setOutdated(false);
        progress.hidden = true;
        showLocalTime();
        if (window.observeDeferredViewers) window.observeDeferredViewers(nextReport);
        window.scrollTo(x, y);
        document.dispatchEvent(new CustomEvent('wmb:station-report-updated'));
        return true;
    }

    async function poll() {
        if (!waiting) return;
        const request = requestId;
        try {
            const response = await fetch('/api/station-report' + window.location.search, {cache: 'no-store'});
            if (!response.ok) throw new Error('Report unavailable');
            const status = await response.json();
            if (!waiting || request !== requestId) return;
            if (status.state === 'error') throw new Error('Report unavailable');
            if (status.state === 'ready' && !interactionOpen() && await applyReport(request)) {
                stop();
                return;
            }
        } catch (_) {
            if (request !== requestId) return;
            progress.hidden = false;
            progress.textContent = 'The report could not be calculated. Use Refresh to try again.';
            stop();
            return;
        }
        if (waiting && request === requestId) timer = window.setTimeout(poll, 500);
    }

    root.addEventListener('click', async function (event) {
        const button = event.target.closest('[data-report-refresh]');
        if (!button || button.disabled) return;
        stop();
        const request = requestId;
        setRefreshing(true);
        progress.hidden = true;
        try {
            const token = document.querySelector('meta[name="csrf-token"]');
            const response = await fetch('/api/station-report' + window.location.search, {
                method: 'POST', headers: {'X-CSRF-Token': token ? token.content : ''}
            });
            if (!response.ok) throw new Error('Report unavailable');
            if (request !== requestId) return;
            waiting = true;
            poll();
        } catch (_) {
            if (request !== requestId) return;
            progress.hidden = false;
            progress.textContent = 'The report could not be calculated. Use Refresh to try again.';
            stop();
        }
    });

    document.addEventListener('wmb:bird-objects-updated', function (event) {
        const detail = event.detail || {};
        if (!detail.filename) return;
        stop();
        progress.hidden = true;
        setOutdated(true);
        root.querySelectorAll('[data-current-report] article[data-image-filename]').forEach(function (tile) {
            if (tile.dataset.imageFilename !== detail.filename) return;
            const object = (detail.objects || []).find(function (item) {
                return String(item.detection_id) === tile.dataset.detectionId;
            });
            if (!object && !detail.imageAbsent) return;
            const removed = object && object.object_status === 'rejected';
            const absent = detail.imageAbsent || object.human_review_state === 'reviewed_negative';
            const name = removed ? 'Moved to trash' : absent ? 'No bird' : object.common_name;
            const label = tile.querySelector('.wm-tile__name');
            if (label) label.textContent = name || 'Bird · species unknown';
            const latin = tile.querySelector('.wm-tile__latin');
            if (latin) latin.textContent = absent || removed ? '' : (object.species_key || '').replaceAll('_', ' ');
            const status = tile.querySelector('.station-card-status');
            if (status) status.textContent = removed ? 'Moved to trash' : absent ? 'No bird · Answered' : 'Answer updated';
            const image = tile.querySelector('img');
            if (image) image.alt = name || 'Bird · species unknown';
        });
    });
    showLocalTime();
    if (waiting) {
        setRefreshing(true);
        timer = window.setTimeout(poll, 500);
    }
}());
