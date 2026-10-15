(function () {
    'use strict';
    const root = document.querySelector('[data-report-state]');
    if (!root) return;
    let revision = root.dataset.reportRevision;
    const progress = document.getElementById('report-progress');
    const modals = document.querySelector('[data-report-modals]');
    let timer;
    let polling = false;
    let viewerOpen = false;
    let changes = 0;
    let appliedChanges = 0;

    function interactionOpen() {
        return viewerOpen || document.querySelector('.modal.show, dialog[open]');
    }

    function hideReport(message) {
        const report = root.querySelector('[data-current-report]');
        if (report) {
            report.style.visibility = 'hidden';
            report.inert = true;
            report.setAttribute('aria-hidden', 'true');
        }
        progress.style.position = report ? 'absolute' : '';
        progress.hidden = false;
        progress.textContent = message;
    }

    function schedule(delay) {
        window.clearTimeout(timer);
        timer = window.setTimeout(poll, delay);
    }

    async function updateReport() {
        const pendingChanges = changes;
        const response = await fetch(window.location.pathname + window.location.search, {cache: 'no-store'});
        if (!response.ok || response.redirected) throw new Error('Report unavailable');
        const page = new DOMParser().parseFromString(await response.text(), 'text/html');
        const nextRoot = page.querySelector('[data-report-state]');
        const nextReport = page.querySelector('[data-current-report]');
        const nextModals = page.querySelector('[data-report-modals]');
        if (!nextRoot || !nextModals) throw new Error('Report unavailable');
        if (nextRoot.dataset.reportState !== 'ready' || !nextReport ||
            interactionOpen() || changes !== pendingChanges) return;

        const previous = root.querySelector('[data-current-report]');
        const scrollX = window.scrollX;
        const scrollY = window.scrollY;
        const offsets = previous ? Array.from(previous.querySelectorAll('.station-report__table-scroll'), node => node.scrollLeft) : [];
        const openDetails = previous ? Array.from(previous.querySelectorAll('details[data-report-details][open]'), node => node.dataset.reportDetails) : [];
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
            node.open = openDetails.includes(node.dataset.reportDetails);
        });
        revision = nextRoot.dataset.reportRevision;
        root.dataset.reportRevision = revision;
        root.dataset.reportState = 'ready';
        appliedChanges = pendingChanges;
        progress.hidden = true;
        if (window.observeDeferredViewers) window.observeDeferredViewers(nextReport);
        window.scrollTo(scrollX, scrollY);
        document.dispatchEvent(new CustomEvent('wmb:station-report-updated'));
    }

    async function poll() {
        if (polling) return;
        polling = true;
        try {
            const response = await fetch('/api/station-report' + window.location.search, {cache: 'no-store'});
            if (!response.ok) throw new Error('Report unavailable');
            const current = await response.json();
            if (current.state !== 'ready' || current.revision !== revision ||
                root.dataset.reportState !== 'ready' || changes !== appliedChanges) {
                hideReport(current.state === 'error' ? 'The update failed. Retrying automatically.' : 'Updating the station history automatically…');
                if (current.state === 'ready' && !interactionOpen()) await updateReport();
            } else {
                const report = root.querySelector('[data-current-report]');
                if (report) {
                    report.style.visibility = '';
                    report.inert = false;
                    report.removeAttribute('aria-hidden');
                }
                progress.hidden = true;
            }
        } catch (_) {
            hideReport('Current data could not be checked. Retrying automatically.');
        } finally {
            polling = false;
            schedule(2000);
        }
    }

    document.addEventListener('show.bs.modal', function () { viewerOpen = true; });
    document.addEventListener('hidden.bs.modal', function () {
        viewerOpen = Boolean(document.querySelector('.modal.show'));
        if (!polling) schedule(0);
    });
    document.addEventListener('wmb:bird-objects-updated', function () {
        changes++;
        hideReport('Updating the station history automatically…');
        if (!polling) schedule(0);
    });
    schedule(500);
}());
