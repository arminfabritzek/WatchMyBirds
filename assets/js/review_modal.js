(function () {
    'use strict';

    const pending = new Map();
    let openRequest = 0;
    let navigating = false;

    function tileFor(trigger) {
        const tile = trigger.closest('.review-grid__tile');
        return tile && tile.querySelector('[data-tile-image][data-modal-target]');
    }

    async function load(tile) {
        const target = tile.dataset.modalTarget;
        const existing = document.querySelector(target);
        if (existing) return existing;
        if (pending.has(target)) return pending.get(target);
        const url = safeSameOriginPath(tile.dataset.modalUrl);
        if (!url) throw new Error('Image details are unavailable.');
        const request = (async function () {
            const response = await fetch(url, { credentials: 'same-origin' });
            if (response.redirected || response.status === 401) {
                redirectToLogin();
                throw new Error('Please sign in again.');
            }
            if (!response.ok) {
                throw new Error(response.status === 404
                    ? 'This image is no longer in Review. Reload the page.'
                    : 'Image details could not be loaded. Please try again.');
            }
            const fragment = document.createElement('template');
            fragment.innerHTML = await response.text();
            const modal = fragment.content.querySelector('.gallery-modal');
            if (!modal || '#' + modal.id !== target) {
                throw new Error('Image details could not be loaded. Please try again.');
            }
            modal.dataset.reviewLazy = 'true';
            document.body.appendChild(modal);
            return modal;
        })();
        pending.set(target, request);
        tile.setAttribute('aria-busy', 'true');
        try {
            return await request;
        } finally {
            pending.delete(target);
            tile.removeAttribute('aria-busy');
        }
    }

    function report(error) {
        if (window.wmToast) window.wmToast(error.message, 'error', 5000);
    }

    async function open(trigger) {
        const tile = tileFor(trigger);
        if (!tile) return;
        const requestId = ++openRequest;
        try {
            const modal = await load(tile);
            if (requestId !== openRequest || !tile.isConnected) return;
            bootstrap.Modal.getOrCreateInstance(modal).show();
        } catch (error) {
            report(error);
        }
    }

    async function navigate(current, direction) {
        if (navigating || modalNavigationInFlight) return;
        const tile = Array.from(document.querySelectorAll('[data-tile-image][data-modal-target]'))
            .find(function (item) { return item.dataset.modalTarget === '#' + current.id; });
        const card = tile && tile.closest('[data-review-grid-card]');
        if (!card) return;
        const tiles = Array.from(card.querySelectorAll('[data-tile-image][data-modal-target]'));
        const index = tiles.indexOf(tile);
        const filename = tile.closest('.review-grid__tile').dataset.imageFilename;
        const step = direction === 'next' ? 1 : -1;
        let next = null;
        for (let offset = 1; offset < tiles.length; offset++) {
            const candidate = tiles[(index + step * offset + tiles.length) % tiles.length];
            if (candidate.closest('.review-grid__tile').dataset.imageFilename !== filename) {
                next = candidate;
                break;
            }
        }
        if (!next) return;
        navigating = true;
        current.setAttribute('aria-busy', 'true');
        try {
            const modal = await load(next);
            if (current.classList.contains('show') && next.isConnected) {
                showModalTransition(current, modal);
            }
        } catch (error) {
            report(error);
        } finally {
            navigating = false;
            current.removeAttribute('aria-busy');
        }
    }

    let decisionsChanged = false;
    document.addEventListener('wmb:bird-objects-updated', function () {
        decisionsChanged = true;
    });
    document.addEventListener('hidden.bs.modal', function () {
        requestAnimationFrame(function () {
            if (decisionsChanged && !navigating && !modalNavigationInFlight
                && !document.querySelector('.gallery-modal.show')) window.location.reload();
        });
    });

    window.WmReviewModal = { open: open, navigate: navigate, tileFor: tileFor };
})();
