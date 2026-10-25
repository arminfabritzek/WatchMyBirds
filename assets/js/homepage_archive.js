/* Archive cards load once after the live page; failed loads have an explicit retry. */
(function () {
    'use strict';
    const section = document.getElementById('homepageArchive');
    if (!section) return;
    const status = section.querySelector('[data-archive-status]');
    const retry = section.querySelector('[data-archive-retry]');
    let inFlight = false;
    let loaded = false;

    async function loadArchive() {
        if (inFlight || loaded) return;
        inFlight = true;
        section.setAttribute('aria-busy', 'true');
        status.textContent = 'Loading your archive…';
        retry.hidden = true;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 120000);
        try {
            const response = await fetch(section.dataset.archiveUrl, {
                credentials: 'same-origin', cache: 'no-store', signal: controller.signal
            });
            if (!response.ok || response.redirected) throw new Error('Archive unavailable');
            const documentCopy = new DOMParser().parseFromString(await response.text(), 'text/html');
            const result = documentCopy.querySelector('[data-homepage-archive-result]');
            const modals = documentCopy.querySelector('[data-homepage-archive-modals]');
            if (!result || !modals) throw new Error('Incomplete archive response');
            // Insert the shared dialogs before their card buttons become usable.
            document.body.appendChild(modals);
            section.replaceChildren(result);
            const total = Number(result.dataset.totalSpecies);
            if (Number.isInteger(total) && total >= 0 && window.__ledTickerStats) {
                window.__ledTickerStats.total = total;
                window.dispatchEvent(new CustomEvent('wmb:archive-total-ready'));
            }
            loaded = true;
        } catch (error) {
            status.textContent = 'Your archive could not be loaded. Please try again.';
            retry.hidden = false;
        } finally {
            clearTimeout(timeout);
            inFlight = false;
            section.setAttribute('aria-busy', 'false');
        }
    }
    retry.addEventListener('click', loadArchive);
    loadArchive();
})();
