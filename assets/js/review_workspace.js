(function () {
    'use strict';

    let noBirdConfirmed = localStorage.getItem('noBirdConfirmed') === 'true'
        || localStorage.getItem('reviewTrashConfirmed') === 'true';
    let reviewMetricsExpanded = localStorage.getItem('reviewMetricsExpanded') !== 'false';
    const REVIEW_PENDING_SPECIES_KEY = 'reviewPendingSpeciesV1';
    const reviewQueueDataEl = document.getElementById('review-queue-data');
    const reviewQueueData = reviewQueueDataEl ? JSON.parse(reviewQueueDataEl.textContent) : [];
    const reviewEventDataEl = document.getElementById('review-event-data');
    const reviewEventData = reviewEventDataEl ? JSON.parse(reviewEventDataEl.textContent) : [];
    const REVIEW_PANEL_CACHE_LIMIT = 24;
    const reviewPanelCache = new Map();
    let reviewPanelLoadToken = 0;
    let latestDecisionStats = null;
    const reviewPanelPrefetchInFlight = new Set();
    const reviewImagePrefetchInFlight = new Set();
    const reviewQueueIndex = new Map(reviewQueueData.map(item => [item.item_key, item]));
    const reviewEventIndex = new Map(reviewEventData.map(event => [event.event_key, event]));

    function getReviewItemKey(itemKind, itemId) {
        return `${itemKind}:${itemId}`;
    }

    function getReviewEventCacheKey(eventKey) {
        return `event:${eventKey}`;
    }

    function getCachedReviewPanel(itemKey) {
        if (!reviewPanelCache.has(itemKey)) return null;
        const html = reviewPanelCache.get(itemKey);
        reviewPanelCache.delete(itemKey);
        reviewPanelCache.set(itemKey, html);
        return html;
    }

    function setCachedReviewPanel(itemKey, html) {
        if (!itemKey) return;
        if (reviewPanelCache.has(itemKey)) {
            reviewPanelCache.delete(itemKey);
        }
        reviewPanelCache.set(itemKey, html);
        while (reviewPanelCache.size > REVIEW_PANEL_CACHE_LIMIT) {
            const oldestKey = reviewPanelCache.keys().next().value;
            if (!oldestKey) break;
            reviewPanelCache.delete(oldestKey);
        }
    }

    function readJsonStorage(storage, key) {
        try {
            const raw = storage.getItem(key);
            return raw ? JSON.parse(raw) : null;
        } catch (_) {
            return null;
        }
    }

    function writeJsonStorage(storage, key, value) {
        try {
            storage.setItem(key, JSON.stringify(value));
        } catch (_) {
            // ignore storage failures
        }
    }

    function getPendingReviewSpeciesMap() {
        return readJsonStorage(sessionStorage, REVIEW_PENDING_SPECIES_KEY) || {};
    }

    function getPendingReviewSpecies(itemKey) {
        return getPendingReviewSpeciesMap()[itemKey] || '';
    }

    function setPendingReviewSpecies(itemKey, species) {
        if (!itemKey) return;
        const map = getPendingReviewSpeciesMap();
        if (species) {
            map[itemKey] = species;
        } else {
            delete map[itemKey];
        }
        writeJsonStorage(sessionStorage, REVIEW_PENDING_SPECIES_KEY, map);
    }

    function clearPendingReviewSpecies(itemKey) {
        if (!itemKey) return;
        const map = getPendingReviewSpeciesMap();
        delete map[itemKey];
        writeJsonStorage(sessionStorage, REVIEW_PENDING_SPECIES_KEY, map);
    }

    function initReviewDefaultBboxes(img) {
        const viewer = img.closest('.wm-image-viewer');
        if (!viewer || viewer.classList.contains('wm-image-viewer--fallback')) return;

        const bx = parseFloat(viewer.dataset.bboxX);
        const by = parseFloat(viewer.dataset.bboxY);
        const bw = parseFloat(viewer.dataset.bboxW);
        const bh = parseFloat(viewer.dataset.bboxH);
        if (isNaN(bx) || isNaN(by) || isNaN(bw) || isNaN(bh) || bw <= 0 || bh <= 0) return;

        const canvas = viewer.querySelector('.bbox-overlay');
        if (!canvas || typeof drawBoundingBoxes !== 'function') return;

        canvas.style.display = 'block';
        drawBoundingBoxes(canvas, img, [{
            x: bx,
            y: by,
            w: bw,
            h: bh,
            name: viewer.dataset.bboxName || 'Detection',
            isCurrent: true
        }], null);
    }

    async function refreshDeepScanStatus() {
        const statusEl = document.getElementById('deep-scan-status');
        if (!statusEl) return;

        try {
            const response = await fetch('/api/status', { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            const active = data.deep_scan_active === true;
            const pending = Number.isFinite(data.deep_scan_queue_pending) ? data.deep_scan_queue_pending : 0;
            const remaining = Number.isFinite(data.deep_scan_candidates_remaining) ? data.deep_scan_candidates_remaining : 0;
            statusEl.textContent = `Deep Scan: Manual only${active ? ' · Running' : ''} | Queue: ${pending} | Remaining: ${remaining}`;
            statusEl.hidden = false;
        } catch (error) {
            statusEl.textContent = '';
            statusEl.hidden = true;
        }
    }

    function getReviewStagePanel() {
        return document.getElementById('reviewActivePanel');
    }

    function getReviewPanel(itemKey) {
        const panel = getReviewStagePanel();
        if (!panel) return null;
        if (!itemKey) return panel;
        return panel.dataset.itemKey === itemKey ? panel : null;
    }

    function getReviewControls(itemKey) {
        const panel = getReviewPanel(itemKey);
        return panel?.querySelector('[data-review-controls]') || null;
    }

    function getReviewItem(itemKey) {
        return Array.from(document.querySelectorAll('.review-queue__item[data-review-item]'))
            .find(item => item.dataset.itemKey === itemKey) || null;
    }

    function getReviewItemRecord(itemKey) {
        // Index-backed fallback so drill-down from an event panel works
        // even though the new event-only side rail no longer renders
        // .review-queue__item nodes. The event surface still needs to
        // reach the per-detection queue panel via Open in Queue.
        if (!itemKey) return null;
        const record = reviewQueueIndex.get(itemKey);
        if (!record) return null;
        return {
            item_key: record.item_key || itemKey,
            item_kind: record.item_kind,
            item_id: String(record.item_id),
            filename: record.filename || '',
            reason: record.review_reason || ''
        };
    }

    function getReviewEvent(eventKey) {
        return Array.from(document.querySelectorAll('.review-event-card[data-review-event]'))
            .find(item => item.dataset.eventKey === eventKey) || null;
    }

    function getVisibleReviewItems() {
        return Array.from(document.querySelectorAll('.review-queue__item[data-review-item]'))
            .filter(item => item.style.display !== 'none');
    }

    function getVisibleReviewEvents() {
        return Array.from(document.querySelectorAll('.review-event-card[data-review-event]'))
            .filter(item => item.style.display !== 'none');
    }

    function getActiveReviewItem() {
        return document.querySelector('.review-queue__item.is-active');
    }

    function getActiveReviewEvent() {
        return document.querySelector('.review-event-card.is-active');
    }

    function isStageOnQueueItem() {
        const panel = getReviewStagePanel();
        return !!(panel && panel.dataset.panelType === 'queue');
    }

    function updateReviewMeta() {
        const focusEl = document.getElementById('review-focus-meta');
        const countEl = document.getElementById('review-visible-count');

        if (isStageOnQueueItem()) {
            const items = getVisibleReviewItems();
            if (items.length > 0) {
                const active = getActiveReviewItem();
                if (countEl) countEl.textContent = items.length;
                if (!focusEl) return;

                if (!active) {
                    focusEl.textContent = `Queue ${items.length}`;
                    return;
                }

                const index = items.findIndex(item => item === active);
                focusEl.textContent = `${index + 1} / ${items.length} · ${(active.dataset.reason || '').replace('_', ' ')}`;
                return;
            }

            // Rail-less drill-down from an event panel (or orphan-only
            // without a visible rail): drive the counter off the JSON
            // index + the stage-panel dataset instead of the DOM.
            const panel = getReviewStagePanel();
            const totalQueue = reviewQueueIndex.size;
            if (countEl) countEl.textContent = totalQueue;
            if (!focusEl) return;

            const currentItemKey = panel?.dataset.itemKey || '';
            if (!currentItemKey || totalQueue === 0) {
                focusEl.textContent = `Queue ${totalQueue}`;
                return;
            }
            const keys = Array.from(reviewQueueIndex.keys());
            const currentIndex = keys.indexOf(currentItemKey);
            const reason = (panel?.dataset.reason || '').replace('_', ' ');
            if (currentIndex === -1) {
                focusEl.textContent = `Queue ${totalQueue}`;
                return;
            }
            focusEl.textContent = `${currentIndex + 1} / ${totalQueue}${reason ? ' · ' + reason : ''}`;
            return;
        }

        const events = getVisibleReviewEvents();
        const activeEvent = getActiveReviewEvent();
        if (countEl) countEl.textContent = events.length;
        if (!focusEl) return;

        if (!activeEvent) {
            focusEl.textContent = `${events.length} event${events.length === 1 ? '' : 's'}`;
            return;
        }

        const index = events.findIndex(item => item === activeEvent);
        const tone = (activeEvent.dataset.eligibility || '').replace('event_', '').replace('_', ' ');
        focusEl.textContent = `${index + 1} / ${events.length} · ${tone || 'event'}`;
    }

    function applyReviewMetricsState(root = document) {
        root.querySelectorAll('[data-review-facts-toggle]').forEach(function (button) {
            button.setAttribute('aria-expanded', reviewMetricsExpanded ? 'true' : 'false');
        });
        root.querySelectorAll('[data-review-facts-panel]').forEach(function (panel) {
            panel.hidden = !reviewMetricsExpanded;
        });
    }

    function applyDecisionStatsToReviewMetrics(root = document) {
        if (!latestDecisionStats) return;

        const values = {
            auto_accepted: latestDecisionStats.states?.['null'] || 0,
            manual_confirmed: latestDecisionStats.manual_confirmed_count || 0,
            policy_confirmed: latestDecisionStats.states?.confirmed || 0,
            uncertain: latestDecisionStats.states?.uncertain || 0,
            unknown: latestDecisionStats.states?.unknown || 0,
            rejected: latestDecisionStats.states?.rejected || 0,
            review_queue: latestDecisionStats.review_queue_count || 0
        };

        root.querySelectorAll('[data-review-global-metric]').forEach(function (el) {
            const metric = el.dataset.reviewGlobalMetric || '';
            if (Object.prototype.hasOwnProperty.call(values, metric)) {
                el.textContent = String(values[metric]);
            }
        });
    }

    async function hydrateReviewSpeciesThumbs() {
        return;
    }

    function ensurePanelImageLoaded() {
        const panel = getReviewStagePanel();
        const img = panel?.querySelector('.review-stage-panel__image');
        if (!img) return;

        if (img.complete && img.naturalWidth > 0) {
            initReviewDefaultBboxes(img);
        }
    }

    function syncViewerScopePreferences(root = document) {
        if (!root || typeof applySmartZoomPreferenceToScope !== 'function') return;
        const run = function () {
            root.querySelectorAll('.wm-viewer-scope').forEach(function (scope) {
                applySmartZoomPreferenceToScope(scope);
                if (typeof redrawBboxOverlay === 'function') {
                    scope.querySelectorAll('.bbox-toggle.active').forEach(function (btn) {
                        redrawBboxOverlay(btn);
                    });
                }
            });
        };

        if (typeof window.requestAnimationFrame === 'function') {
            window.requestAnimationFrame(run);
        } else {
            run();
        }
    }

    async function loadReviewPanel(itemKind, itemId, options = {}) {
        const panel = getReviewStagePanel();
        if (!panel) return false;
        const itemKey = getReviewItemKey(itemKind, itemId);

        if (!options.force && panel.dataset.itemKey === itemKey && panel.innerHTML.trim()) {
            ensurePanelImageLoaded();
            hydrateReviewControls(itemKey);
            syncViewerScopePreferences(panel);
            return true;
        }

        const loadToken = ++reviewPanelLoadToken;
        panel.classList.add('is-loading');

        try {
            let html = getCachedReviewPanel(itemKey);
            if (!html || options.force) {
                const response = await fetch(`/api/review/panel/${encodeURIComponent(itemKind)}/${encodeURIComponent(itemId)}`, {
                    cache: 'no-store',
                    credentials: 'same-origin'
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                html = await response.text();
                setCachedReviewPanel(itemKey, html);
            }

            if (loadToken !== reviewPanelLoadToken) return false;

            panel.innerHTML = html;
            panel.dataset.panelType = 'queue';
            panel.removeAttribute('data-event-key');
            panel.dataset.itemKind = itemKind;
            panel.dataset.itemId = itemId;
            panel.dataset.itemKey = itemKey;
            const queueRecord = reviewQueueIndex.get(itemKey);
            panel.dataset.filename = queueRecord?.filename || '';
            panel.dataset.reason = getReviewItem(itemKey)?.dataset.reason || queueRecord?.review_reason || '';
            ensurePanelImageLoaded();
            applyReviewMetricsState(panel);
            applyDecisionStatsToReviewMetrics(panel);
            hydrateReviewControls(itemKey);
            hydrateReviewSpeciesThumbs(panel);
            syncViewerScopePreferences(panel);
            return true;
        } catch (error) {
            console.error('Review panel load error:', error);
            return false;
        } finally {
            if (loadToken === reviewPanelLoadToken) {
                panel.classList.remove('is-loading');
            }
        }
    }

    async function prefetchReviewPanel(itemKey) {
        if (!itemKey || reviewPanelCache.has(itemKey) || reviewPanelPrefetchInFlight.has(itemKey)) {
            return;
        }

        const item = reviewQueueIndex.get(itemKey);
        if (!item) return;

        reviewPanelPrefetchInFlight.add(itemKey);
        try {
            const response = await fetch(`/api/review/panel/${encodeURIComponent(item.item_kind)}/${encodeURIComponent(item.item_id)}`, {
                cache: 'no-store',
                credentials: 'same-origin'
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const html = await response.text();
            setCachedReviewPanel(itemKey, html);
        } catch (error) {
            console.error('Review panel prefetch error:', error);
        } finally {
            reviewPanelPrefetchInFlight.delete(itemKey);
        }
    }

    async function loadReviewEventPanel(eventKey, options = {}) {
        const panel = getReviewStagePanel();
        if (!panel || !eventKey) return false;
        const cacheKey = getReviewEventCacheKey(eventKey);

        if (!options.force && panel.dataset.panelType === 'event' && panel.dataset.eventKey === eventKey && panel.innerHTML.trim()) {
            applyReviewMetricsState(panel);
            applyDecisionStatsToReviewMetrics(panel);
            hydrateReviewEventControls(eventKey);
            syncViewerScopePreferences(panel);
            return true;
        }

        const loadToken = ++reviewPanelLoadToken;
        panel.classList.add('is-loading');

        try {
            let html = getCachedReviewPanel(cacheKey);
            if (!html || options.force) {
                const response = await fetch(`/api/review/event-panel/${encodeURIComponent(eventKey)}`, {
                    cache: 'no-store',
                    credentials: 'same-origin'
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                html = await response.text();
                setCachedReviewPanel(cacheKey, html);
            }

            if (loadToken !== reviewPanelLoadToken) return false;

            panel.innerHTML = html;
            panel.dataset.panelType = 'event';
            panel.dataset.eventKey = eventKey;
            panel.removeAttribute('data-item-kind');
            panel.removeAttribute('data-item-id');
            panel.removeAttribute('data-item-key');
            panel.removeAttribute('data-filename');
            panel.removeAttribute('data-reason');
            applyReviewMetricsState(panel);
            applyDecisionStatsToReviewMetrics(panel);
            hydrateReviewEventControls(eventKey);
            syncViewerScopePreferences(panel);
            if (typeof window.observeDeferredViewers === 'function') {
                window.observeDeferredViewers(panel);
            }
            return true;
        } catch (error) {
            console.error('Review event panel load error:', error);
            return false;
        } finally {
            if (loadToken === reviewPanelLoadToken) {
                panel.classList.remove('is-loading');
            }
        }
    }

    function renderEmptyEventStage() {
        const panel = getReviewStagePanel();
        if (!panel) return;
        panel.dataset.panelType = 'event';
        panel.removeAttribute('data-event-key');
        panel.removeAttribute('data-item-kind');
        panel.removeAttribute('data-item-id');
        panel.removeAttribute('data-item-key');
        panel.removeAttribute('data-filename');
        panel.removeAttribute('data-reason');
        panel.innerHTML = `
            <div class="review-stage-panel__empty">
                <h3>No events available</h3>
                <p>As soon as unresolved detections form a biological observation (same species, gap ≤ 30 min) they appear here for one-click confirmation.</p>
            </div>
        `;
    }

    function prefetchReviewImage(itemKey) {
        if (!itemKey || reviewImagePrefetchInFlight.has(itemKey)) return;
        const item = reviewQueueIndex.get(itemKey);
        const imageUrl = item?.full_url || item?.thumb_url || '';
        if (!imageUrl) return;

        reviewImagePrefetchInFlight.add(itemKey);
        const img = new Image();
        img.decoding = 'async';
        img.loading = 'eager';
        img.onload = function () {
            reviewImagePrefetchInFlight.delete(itemKey);
        };
        img.onerror = function () {
            reviewImagePrefetchInFlight.delete(itemKey);
        };
        img.src = imageUrl;
    }

    function scheduleReviewPrefetch(itemKey) {
        if (!itemKey) return;
        const nextItemKey = getNextReviewItemKey(itemKey, 1);
        const prevItemKey = getNextReviewItemKey(itemKey, -1);
        const runPrefetch = function () {
            prefetchReviewImage(itemKey);
            prefetchReviewPanel(nextItemKey);
            prefetchReviewPanel(prevItemKey);
            prefetchReviewImage(nextItemKey);
            prefetchReviewImage(prevItemKey);
        };

        if (typeof window.requestIdleCallback === 'function') {
            window.requestIdleCallback(runPrefetch, { timeout: 900 });
            return;
        }

        window.setTimeout(runPrefetch, 120);
    }

    async function selectReviewItem(itemKey, options = {}) {
        // Two drill-down modes:
        //   1. The legacy rail mode, where a .review-queue__item node
        //      exists (still the bootstrap path on orphan-only pages).
        //   2. The event-surface drill-down, where there is no rail
        //      node and we resolve the queue record directly from
        //      reviewQueueIndex. This is what "Open in Queue" inside
        //      an event panel uses.
        const item = getReviewItem(itemKey);
        if (item) {
            if (item.style.display === 'none') return;
            document.querySelectorAll('.review-queue__item.is-active').forEach(el => el.classList.remove('is-active'));
            item.classList.add('is-active');
            updateReviewMeta();

            if (options.scroll !== false) {
                item.scrollIntoView({ block: 'nearest', behavior: options.instant ? 'auto' : 'smooth' });
            }

            const loaded = await loadReviewPanel(item.dataset.itemKind, item.dataset.itemId, options);
            if (loaded) {
                scheduleReviewPrefetch(itemKey);
            }
            return;
        }

        const record = getReviewItemRecord(itemKey);
        if (!record) return;

        document.querySelectorAll('.review-event-card.is-active').forEach(el => {
            el.classList.remove('is-active');
            el.removeAttribute('aria-current');
        });
        updateReviewMeta();

        const loaded = await loadReviewPanel(record.item_kind, record.item_id, options);
        if (loaded) {
            scheduleReviewPrefetch(itemKey);
        }
    }

    async function selectReviewEvent(eventKey, options = {}) {
        const card = getReviewEvent(eventKey);
        if (!card || card.style.display === 'none') return;

        document.querySelectorAll('.review-event-card.is-active').forEach(el => {
            el.classList.remove('is-active');
            el.removeAttribute('aria-current');
        });
        card.classList.add('is-active');
        card.setAttribute('aria-current', 'true');
        updateReviewMeta();

        if (options.scroll !== false) {
            card.scrollIntoView({ block: 'nearest', behavior: options.instant ? 'auto' : 'smooth' });
        }

        await loadReviewEventPanel(eventKey, options);
    }

    function getNextReviewItemKey(itemKey, direction = 1) {
        // Prefer the rail (DOM order, honours style.display filters).
        const items = getVisibleReviewItems();
        if (items.length > 0) {
            const current = getReviewItem(itemKey);
            const index = items.findIndex(item => item === current);
            if (index === -1) return items[0]?.dataset.itemKey || null;
            const nextIndex = (index + direction + items.length) % items.length;
            return items[nextIndex]?.dataset.itemKey || null;
        }

        // Index fallback: the page is event-only (or orphan-only and
        // the rail has not been rendered). Iterate the JSON order.
        if (reviewQueueIndex.size === 0) return null;
        const keys = Array.from(reviewQueueIndex.keys());
        const currentIndex = keys.indexOf(itemKey);
        if (currentIndex === -1) return keys[0] || null;
        const nextIndex = (currentIndex + direction + keys.length) % keys.length;
        return keys[nextIndex] || null;
    }

    function getNextReviewEventKey(eventKey, direction = 1) {
        const items = getVisibleReviewEvents();
        const current = getReviewEvent(eventKey);
        const index = items.findIndex(item => item === current);
        if (index === -1 || items.length === 0) return null;
        const nextIndex = (index + direction + items.length) % items.length;
        return items[nextIndex]?.dataset.eventKey || null;
    }

    async function stepReviewItem(direction) {
        // Receipts are scoped to the currently active item. Clear them
        // before the new panel loads so no "from X to Y" strip leaks
        // from the previous item into the next one.
        clearReviewSpeciesReceipts();

        if (isStageOnQueueItem()) {
            // Rail-first, stage-panel dataset as the index fallback so
            // orphan-only or drill-down-from-event pages can still
            // navigate with the arrow keys.
            const active = getActiveReviewItem();
            const panel = getReviewStagePanel();
            const currentKey = active?.dataset.itemKey || panel?.dataset.itemKey || '';
            if (!currentKey) return;
            const nextItemKey = getNextReviewItemKey(currentKey, direction);
            if (nextItemKey) await selectReviewItem(nextItemKey);
            return;
        }

        const activeEvent = getActiveReviewEvent();
        if (!activeEvent) return;
        const nextEventKey = getNextReviewEventKey(activeEvent.dataset.eventKey, direction);
        if (nextEventKey) await selectReviewEvent(nextEventKey);
    }

    function applyReviewBboxUi(controls, bboxReview) {
        if (!controls) return;
        controls.dataset.bboxReview = bboxReview || '';
        controls.querySelectorAll('[data-bbox-review-toggle]').forEach(function (btn) {
            btn.dataset.bboxReviewValue = bboxReview || '';
            btn.classList.toggle('is-correct', bboxReview === 'correct');
            btn.classList.toggle('is-wrong', bboxReview === 'wrong');
            const copy = btn.querySelector('[data-bbox-review-copy]');
            if (copy) {
                copy.textContent = bboxReview === 'wrong' ? 'Wrong' : 'Correct';
            }
        });
        updateReviewApproveState(controls);
    }

    function getNextReviewBboxState(currentState) {
        return currentState === 'wrong' ? 'correct' : 'wrong';
    }

    function applyReviewSpeciesUi(controls, species, options = {}) {
        if (!controls) return;
        const isEventScope = controls.hasAttribute('data-review-event-controls');
        const itemKey = controls.dataset.itemKey || '';
        const originalSpecies = controls.dataset.originalSpecies || '';
        const selectedOrigin = species
            ? (options.origin !== undefined ? options.origin : (controls.dataset.selectedSpeciesOrigin || ''))
            : '';
        const selectedCommon = species
            ? (options.commonName || lookupSpeciesCommonName(controls, species) || '')
            : '';
        const persistPending = options.persistPending !== false;
        controls.dataset.selectedSpecies = species || '';
        controls.dataset.selectedSpeciesCommon = selectedCommon;
        controls.dataset.selectedSpeciesOrigin = selectedOrigin;
        controls.dataset.selectedSpeciesRefImageUrl = species
            ? (options.refImageUrl || lookupSpeciesRefImageUrl(controls, species) || '')
            : '';
        if (isEventScope) {
            controls.dataset.species = species || '';
        }
        controls.querySelectorAll('.review-stage-panel__species-btn').forEach(btn => {
            const isSelected = (btn.dataset.species || '') === (species || '');
            btn.classList.toggle('is-selected', isSelected);
            if (isEventScope) {
                btn.dataset.reviewPanelAction = 'select_event_species';
            } else {
                btn.dataset.reviewPanelAction = isSelected ? 'confirm_species' : 'select_species';
            }
        });
        controls.querySelectorAll('[data-current-species]').forEach(btn => {
            btn.dataset.currentSpecies = species || '';
        });
        if (persistPending) {
            if (!species || (originalSpecies && species === originalSpecies)) {
                clearPendingReviewSpecies(itemKey);
            } else {
                setPendingReviewSpecies(itemKey, species || '');
            }
        }
        if (isEventScope) {
            updateReviewEventSpeciesSummary(controls, species, selectedOrigin);
            // Post fixed-5 (2026-04-08): mirror the event-level species
            // choice onto every auto-origin cell label in the grid so the
            // label under each Review frame tracks the picker. Cells with
            // `data-species-is-manual="1"` keep their per-frame manual
            // override — manual wins.
            //
            // Colour slot: if the caller passes a fresh slot (from the
            // clicked species-btn's data-species-colour), propagate it
            // through to every auto-origin cell. Without this, cells
            // kept their stale data-species-colour and the bbox
            // overlay redraw picked up the old slot.
            mirrorEventSpeciesToAutoCells(
                species,
                selectedCommon,
                controls.dataset.selectedSpeciesRefImageUrl || '',
                options.speciesColour !== undefined ? options.speciesColour : null
            );
        }
        updateReviewSpeciesReceipt(controls, species, selectedOrigin);
        updateReviewApproveState(controls);
    }

    function syncReviewCellSpeciesArtifacts(cell, speciesKey, commonName, refImageUrl, speciesColourSlot) {
        if (!cell) return;
        const nextKey = speciesKey || '';
        const nextCommon = commonName || nextKey.replace(/_/g, ' ') || '';
        const nextRefImageUrl = refImageUrl || '';
        // speciesColourSlot is null when the caller has no fresh value
        // (e.g. per-frame picker flows that don't expose the slot). In
        // that case we leave the cell's existing data-species-colour
        // alone. When it is a valid slot we write it through so the
        // --cell-species-colour custom property + bbox draw pick it up.
        const hasColour = Number.isFinite(speciesColourSlot);

        if (hasColour) {
            cell.dataset.speciesColour = String(speciesColourSlot);
            cell.style.setProperty(
                '--cell-species-colour',
                'var(--species-colour-' + String(speciesColourSlot) + ')'
            );
        }

        const changeSpeciesAction = cell.querySelector('.wm-toolbox__item[data-action="change-species"]');
        if (changeSpeciesAction) {
            changeSpeciesAction.dataset.currentSpecies = nextKey;
        }

        const viewer = cell.querySelector('.wm-image-viewer');
        if (viewer) {
            viewer.dataset.bboxName = nextCommon || nextKey;
        }

        const bboxBtn = cell.querySelector('.bbox-toggle');
        if (bboxBtn) {
            try {
                const currentBbox = JSON.parse(bboxBtn.dataset.currentBbox || '{}');
                if (currentBbox && typeof currentBbox === 'object' && Object.keys(currentBbox).length > 0) {
                    currentBbox.name = nextCommon || nextKey;
                    if (hasColour) {
                        currentBbox.speciesColour = speciesColourSlot;
                    }
                    bboxBtn.dataset.currentBbox = JSON.stringify(currentBbox);
                }
            } catch (error) {
                console.warn('[review] failed to sync bbox label payload', error);
            }

            if (bboxBtn.classList.contains('active') && typeof redrawBboxOverlay === 'function') {
                redrawBboxOverlay(bboxBtn);
            }
        }

        const media = cell.querySelector('.review-event-panel__cell-media');
        const currentRef = cell.querySelector('.review-species-ref');
        if (media) {
            let nextRef = currentRef;
            if (nextRefImageUrl) {
                if (!nextRef || !nextRef.matches('img.review-species-ref')) {
                    nextRef = document.createElement('img');
                    nextRef.className = 'review-species-ref';
                    nextRef.alt = '';
                    nextRef.setAttribute('aria-hidden', 'true');
                    if (currentRef) {
                        currentRef.replaceWith(nextRef);
                    } else {
                        media.appendChild(nextRef);
                    }
                }
                nextRef.setAttribute('src', nextRefImageUrl);
                nextRef.setAttribute('title', nextCommon || nextKey);
            } else {
                if (!nextRef || !nextRef.classList.contains('review-species-ref--initial')) {
                    nextRef = document.createElement('span');
                    nextRef.className = 'review-species-ref review-species-ref--initial';
                    nextRef.setAttribute('aria-hidden', 'true');
                    if (currentRef) {
                        currentRef.replaceWith(nextRef);
                    } else {
                        media.appendChild(nextRef);
                    }
                }
                nextRef.setAttribute('title', nextCommon || nextKey);
                nextRef.textContent = (nextCommon || nextKey || '?').charAt(0);
            }
        }
    }

    function mirrorEventSpeciesToAutoCells(species, commonName, refImageUrl, speciesColour) {
        const panel = getReviewStagePanel();
        if (!panel) return;
        const grid = panel.querySelector('[data-review-event-grid]');
        if (!grid) return;
        const nextCommon = commonName || '';
        const nextKey = species || '';
        const nextRefImageUrl = refImageUrl || '';
        // Normalise the next colour slot: accept numeric or string,
        // coerce to a number in range [0..7], fall back to null (leave
        // the cell's slot untouched) when the value is missing/invalid.
        let nextColourSlot = null;
        if (speciesColour !== undefined && speciesColour !== null && speciesColour !== '') {
            const parsed = Number(speciesColour);
            if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 7) {
                nextColourSlot = parsed;
            }
        }
        grid.querySelectorAll('.review-event-panel__cell').forEach(function (cell) {
            if (cell.dataset.contextOnly === '1') return;
            if (cell.dataset.speciesIsManual === '1') return;
            const labelBtn = cell.querySelector('.review-event-panel__cell-species[data-review-cell-relabel]');
            if (labelBtn) {
                if (nextCommon) labelBtn.textContent = nextCommon;
                labelBtn.dataset.currentSpecies = nextKey;
                labelBtn.dataset.currentSpeciesCommon = nextCommon;
            } else {
                const labelSpan = cell.querySelector('span.review-event-panel__cell-species');
                if (labelSpan && nextCommon) labelSpan.textContent = nextCommon;
            }
            syncReviewCellSpeciesArtifacts(cell, nextKey, nextCommon, nextRefImageUrl, nextColourSlot);
        });
    }

    function updateReviewEventSpeciesSummary(controls, species, origin) {
        if (!controls) return;
        const nameEl = controls.querySelector('[data-review-event-species-name]');
        const metaEl = controls.querySelector('[data-review-event-species-meta]');
        const originalSpecies = controls.dataset.originalSpecies || '';
        const nextSpecies = species || originalSpecies || '';
        const nextCommon = controls.dataset.selectedSpeciesCommon
            || lookupSpeciesCommonName(controls, nextSpecies)
            || controls.dataset.originalSpeciesCommon
            || nextSpecies
            || 'Unknown species';

        if (nameEl) {
            nameEl.textContent = nextCommon;
        }
        if (!metaEl) return;

        let nextMeta = controls.dataset.originalSpeciesMeta || '';
        if (nextSpecies && nextSpecies !== originalSpecies) {
            nextMeta = 'Pending event choice';
        } else if (origin === 'manual') {
            nextMeta = 'Manual';
        } else if (origin === 'cls') {
            nextMeta = 'CLS';
        } else if (origin === 'default') {
            nextMeta = 'Default';
        }
        metaEl.textContent = nextMeta;
    }

    /**
     * Rebuild the species change-receipt slot in the same DOM mutation
     * as the quick-pick `is-selected` toggle so the operator sees a
     * clear "previous -> new" diff with an Undo affordance the moment
     * they click a different species.
     *
     * The slot lives at `[data-review-species-receipt]` inside the
     * Species section (orphan modal and event panel controls aside).
     * `data-original-species` + `data-original-species-common` on the
     * controls root anchor the "previous" side. These values are set
     * once at initial render and never mutated.
     *
     * All DOM construction uses createElement + textContent so user
     * content never touches innerHTML. XSS-safe by construction.
     */
    function updateReviewSpeciesReceipt(controls, nextSpecies, origin) {
        if (!controls) return;
        const slot = controls.querySelector('[data-review-species-receipt]');
        if (!slot) return;

        // Clear any previous receipt content via DOM node removal —
        // no innerHTML anywhere in this function.
        while (slot.firstChild) slot.removeChild(slot.firstChild);

        const originalKey = controls.dataset.originalSpecies || '';
        const originalCommon = controls.dataset.originalSpeciesCommon || originalKey || '';
        const nextKey = nextSpecies || '';
        if (!originalKey || !nextKey || originalKey === nextKey) {
            slot.classList.remove('is-active');
            return;
        }
        const nextCommon = controls.dataset.selectedSpeciesCommon
            || lookupSpeciesCommonName(controls, nextKey)
            || nextKey;
        const originLabel = origin === 'manual' || origin === 'pending'
            ? 'Manually changed'
            : origin === 'cls' ? 'From CLS suggestion' : '';

        slot.classList.add('is-active');

        const receipt = document.createElement('div');
        receipt.className = 'review-species-receipt';
        receipt.setAttribute('role', 'status');
        receipt.setAttribute('aria-live', 'polite');

        const prev = document.createElement('span');
        prev.className = 'review-species-receipt__prev';
        prev.title = 'Previous species';
        prev.textContent = originalCommon;
        receipt.appendChild(prev);

        const arrow = document.createElement('span');
        arrow.className = 'review-species-receipt__arrow';
        arrow.setAttribute('aria-hidden', 'true');
        arrow.textContent = '→';
        receipt.appendChild(arrow);

        const next = document.createElement('span');
        next.className = 'review-species-receipt__new';
        next.title = 'New species';
        next.textContent = nextCommon;
        receipt.appendChild(next);

        if (originLabel) {
            const originEl = document.createElement('span');
            originEl.className = 'review-species-receipt__origin';
            originEl.textContent = originLabel;
            receipt.appendChild(originEl);
        }

        const undo = document.createElement('button');
        undo.type = 'button';
        undo.className = 'review-species-receipt__undo';
        undo.dataset.reviewPanelAction = 'undo_species_change';
        undo.title = 'Revert species change';
        undo.textContent = 'Undo';
        receipt.appendChild(undo);

        slot.appendChild(receipt);
    }

    function lookupSpeciesCommonName(controls, speciesKey) {
        if (!controls || !speciesKey) return '';
        const btn = controls.querySelector(
            `.review-stage-panel__species-btn[data-species="${speciesKey.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"]`
        );
        if (btn) {
            const nameEl = btn.querySelector('.review-stage-panel__species-name');
            if (nameEl && nameEl.textContent) return nameEl.textContent.trim();
        }
        return '';
    }

    function lookupSpeciesRefImageUrl(controls, speciesKey) {
        if (!controls || !speciesKey) return '';
        const btn = controls.querySelector(
            `.review-stage-panel__species-btn[data-species="${speciesKey.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"]`
        );
        if (!btn) return '';
        return btn.dataset.speciesRefImageUrl
            || btn.querySelector('.review-stage-panel__species-image')?.getAttribute('src')
            || '';
    }

    function clearReviewSpeciesReceipts() {
        document
            .querySelectorAll('[data-review-species-receipt]')
            .forEach(function (slot) {
                slot.classList.remove('is-active');
                while (slot.firstChild) slot.removeChild(slot.firstChild);
            });
    }

    function updateReviewApproveState(controls) {
        if (!controls) return;
        const approveBtn = controls.querySelector('[data-review-panel-action="approve_review"]');
        const bboxReview = controls.dataset.bboxReview || '';
        if (approveBtn) {
            const selectedSpecies = controls.dataset.selectedSpecies || '';
            approveBtn.disabled = !selectedSpecies || !['correct', 'wrong'].includes(bboxReview);
        }

        const approveEventBtn = controls.querySelector('[data-review-panel-action="approve_event"]');
        if (approveEventBtn) {
            // Post fixed-5: `data-can-approve` is `1` whenever the event
            // carries any species at all (candidate or manually selected),
            // regardless of the cluster-time `event_eligible` / `event_ineligible`
            // split. Ineligible events get routed through /api/review/event-resolve
            // at submit time — see reviewApproveEvent. The gate here only
            // enforces the UI precondition: species + bbox review must both be
            // set before the button is clickable.
            const canApprove = controls.dataset.canApprove === '1';
            const selectedSpecies = controls.dataset.species || controls.dataset.selectedSpecies || '';
            approveEventBtn.disabled = !canApprove || !selectedSpecies || !['correct', 'wrong'].includes(bboxReview);
        }
    }

    function applyReviewEventBboxUi(controls, bboxReview) {
        if (!controls) return;
        controls.dataset.bboxReview = bboxReview || '';
        controls.querySelectorAll('[data-review-event-bbox-toggle]').forEach(function (btn) {
            btn.dataset.bboxReviewValue = bboxReview || '';
            btn.classList.toggle('is-correct', bboxReview === 'correct');
            btn.classList.toggle('is-wrong', bboxReview === 'wrong');
            const copy = btn.querySelector('[data-review-event-bbox-copy]');
            if (copy) {
                copy.textContent = bboxReview === 'wrong' ? 'Wrong' : 'Correct';
            }
        });
        updateReviewApproveState(controls);
    }

    function toggleReviewFacts(button) {
        const shell = button.closest('.review-stage-panel__viewer-shell');
        const panel = shell?.querySelector('[data-review-facts-panel]');
        if (!shell || !panel) return;

        const expanded = button.getAttribute('aria-expanded') === 'true';
        reviewMetricsExpanded = !expanded;
        localStorage.setItem('reviewMetricsExpanded', reviewMetricsExpanded ? 'true' : 'false');
        applyReviewMetricsState(shell);
    }

    function hydrateReviewControls(itemKey) {
        const controls = getReviewControls(itemKey);
        if (!controls) return;
        const pendingSpecies = getPendingReviewSpecies(itemKey);
        applyReviewSpeciesUi(controls, pendingSpecies || controls.dataset.selectedSpecies || '', {
            origin: pendingSpecies ? 'pending' : (controls.dataset.selectedSpeciesOrigin || ''),
            persistPending: Boolean(pendingSpecies)
        });
        applyReviewBboxUi(controls, controls.dataset.bboxReview || '');
    }

    function hydrateReviewEventControls(eventKey) {
        const panel = getReviewStagePanel();
        if (!panel || panel.dataset.eventKey !== eventKey) return;
        const controls = panel.querySelector('[data-review-event-controls]');
        if (!controls) return;
        const pendingSpecies = getPendingReviewSpecies(controls.dataset.itemKey || eventKey);
        applyReviewSpeciesUi(controls, pendingSpecies || controls.dataset.species || controls.dataset.originalSpecies || '', {
            origin: pendingSpecies ? 'pending' : (controls.dataset.selectedSpeciesOrigin || controls.dataset.originalSpeciesOrigin || ''),
            persistPending: Boolean(pendingSpecies)
        });
        applyReviewEventBboxUi(controls, controls.dataset.bboxReview || 'correct');
        updateReviewMultiSelectUi(panel);
    }

    function getReviewEventGrid(panel = getReviewStagePanel()) {
        return panel?.querySelector('[data-review-event-grid]') || null;
    }

    /**
     * Post fixed-5 (2026-04-08): fan out a bbox-overlay toggle click to
     * every cell inside the same `.wm-viewer-scope` (the event grid).
     *
     * The user clicks the bbox toggle on any one cell. We compute the
     * target state from that one button (flip its current `active`
     * state), then walk every sibling host in the scope and delegate to
     * the shared `toggleBboxOverlay` — but only on hosts whose current
     * button state diverges from the target, so a cell that's already in
     * the desired state doesn't get double-toggled back off.
     *
     * LocalStorage (`wmb_review_bbox_pref`) is already scope-global, so
     * this just makes the DOM follow the pref the moment the user
     * clicks, instead of waiting for a re-render to resync the other
     * cells. Gallery/Stream/Trash surfaces still call `toggleBboxOverlay`
     * directly on single-host scopes and are unaffected.
     */
    function toggleBboxOverlayForReviewScope(clickedBtn) {
        if (typeof toggleBboxOverlay !== 'function') return;
        const scope = clickedBtn.closest('.wm-viewer-scope');
        if (!scope) {
            toggleBboxOverlay(clickedBtn);
            return;
        }
        const targetActive = !clickedBtn.classList.contains('active');
        const buttons = Array.from(
            scope.querySelectorAll('[data-review-viewer-tool="bbox"]')
        );
        if (buttons.length === 0) {
            toggleBboxOverlay(clickedBtn);
            return;
        }
        buttons.forEach(function (btn) {
            const isActive = btn.classList.contains('active');
            if (isActive !== targetActive) {
                toggleBboxOverlay(btn);
            }
        });
    }

    function getReviewMultiSelectCheckboxes(panel = getReviewStagePanel()) {
        return Array.from(panel?.querySelectorAll('[data-review-multi-select-checkbox]') || []);
    }

    function getReviewMultiSelectToggle(panel = getReviewStagePanel()) {
        return panel?.querySelector('[data-review-multi-select-toggle]') || null;
    }

    function getReviewMultiSelectFooter(panel = getReviewStagePanel()) {
        return panel?.querySelector('[data-review-multi-select-footer]') || null;
    }

    function isReviewMultiSelectMode(panel = getReviewStagePanel()) {
        return getReviewEventGrid(panel)?.dataset.multiSelectMode === '1';
    }

    function setReviewCellSelected(checkbox, checked) {
        if (!checkbox) return;
        checkbox.checked = checked;
        const cell = checkbox.closest('.review-event-panel__cell');
        if (cell) {
            cell.classList.toggle('is-selected', checked);
        }
    }

    function toggleReviewMultiSelectRange(checkboxes, start, end, checked) {
        const min = Math.min(start, end);
        const max = Math.max(start, end);
        for (let i = min; i <= max; i++) {
            setReviewCellSelected(checkboxes[i], checked);
        }
    }

    function updateReviewMultiSelectUi(panel = getReviewStagePanel()) {
        const grid = getReviewEventGrid(panel);
        if (!grid) return;

        const mode = isReviewMultiSelectMode(panel);
        const checkboxes = getReviewMultiSelectCheckboxes(panel);
        const selectedCount = checkboxes.filter(function (checkbox) {
            return checkbox.checked;
        }).length;
        const toggle = getReviewMultiSelectToggle(panel);
        const footer = getReviewMultiSelectFooter(panel);
        const hint = panel?.querySelector('[data-review-multi-select-hint]');
        const counter = panel?.querySelector('#reviewEventBatchCounter');
        const relabelBtn = panel?.querySelector('#reviewEventBatchRelabelBtn');
        const trashBtn = panel?.querySelector('#reviewEventBatchTrashBtn');

        grid.classList.toggle('is-multi-select', mode);
        if (toggle) {
            toggle.classList.toggle('is-active', mode);
            toggle.setAttribute('aria-pressed', mode ? 'true' : 'false');
            toggle.textContent = mode ? 'Exit multi-select' : 'Select multiple';
        }
        if (hint) {
            hint.hidden = !mode;
        }
        if (footer) {
            footer.hidden = !mode;
        }
        if (counter) {
            counter.textContent = selectedCount + ' selected';
        }
        if (relabelBtn) {
            relabelBtn.disabled = selectedCount === 0;
        }
        if (trashBtn) {
            trashBtn.disabled = selectedCount === 0;
        }

        panel.querySelectorAll('[data-review-cell-relabel], [data-review-frame-decision]').forEach(function (btn) {
            btn.disabled = mode;
        });

        // Mode-aware species-block feedback. In multi-select, the species
        // strip no longer stages a pending event-wide choice — a click
        // commits a bulk relabel immediately and can split the event.
        // Flip the section label, hint copy, and species-strip tooltips so
        // the operator sees the changed semantics before the click.
        const speciesSection = panel?.querySelector('[data-review-event-species-section]');
        const speciesLabel = panel?.querySelector('[data-review-event-species-label]');
        const speciesHint = panel?.querySelector('[data-review-event-species-hint]');
        if (speciesSection) {
            const selectionActive = mode && selectedCount > 0;
            speciesSection.classList.toggle('is-multi-select', mode);
            speciesSection.classList.toggle('is-multi-select-armed', selectionActive);
            if (speciesLabel) {
                if (selectionActive) {
                    speciesLabel.textContent = 'Relabel ' + selectedCount + ' selected frame'
                        + (selectedCount === 1 ? '' : 's');
                } else if (mode) {
                    speciesLabel.textContent = 'Select frames to relabel';
                } else {
                    speciesLabel.textContent = 'Species';
                }
            }
            if (speciesHint) {
                if (selectionActive) {
                    speciesHint.textContent = 'Clicking a species below will IMMEDIATELY relabel the '
                        + selectedCount + ' selected frame'
                        + (selectedCount === 1 ? '' : 's')
                        + '. If the new species differs from the event species, the event will be split into separate events (one per species).';
                } else if (mode) {
                    speciesHint.textContent = 'Pick frames with the checkboxes first. Then clicking a species will relabel just those frames.';
                } else {
                    speciesHint.textContent = 'Event-wide species choice stays local until you approve the event. With selected frames, a quick-pick relabels them immediately.';
                }
            }
        }
        const speciesStrip = panel?.querySelectorAll('[data-review-panel-action="select_event_species"]');
        if (speciesStrip) {
            speciesStrip.forEach(function (btn) {
                const originalTitle = btn.dataset._tipOriginal || btn.getAttribute('title') || '';
                if (!btn.dataset._tipOriginal) {
                    btn.dataset._tipOriginal = originalTitle;
                }
                if (mode && selectedCount > 0) {
                    btn.setAttribute('title',
                        'Relabel ' + selectedCount + ' selected frame'
                        + (selectedCount === 1 ? '' : 's')
                        + ' to this species (immediate; may split the event)');
                } else if (mode) {
                    btn.setAttribute('title', 'Pick frames first, then click a species to relabel them');
                } else if (btn.dataset._tipOriginal) {
                    btn.setAttribute('title', btn.dataset._tipOriginal);
                }
            });
        }
    }

    function clearReviewMultiSelect(panel = getReviewStagePanel()) {
        getReviewMultiSelectCheckboxes(panel).forEach(function (checkbox) {
            setReviewCellSelected(checkbox, false);
        });
        const grid = getReviewEventGrid(panel);
        if (grid) {
            delete grid.dataset.lastMultiSelectIndex;
        }
        updateReviewMultiSelectUi(panel);
    }

    function setReviewMultiSelectMode(enabled, panel = getReviewStagePanel()) {
        const grid = getReviewEventGrid(panel);
        if (!grid) return;
        grid.dataset.multiSelectMode = enabled ? '1' : '0';
        if (!enabled) {
            clearReviewMultiSelect(panel);
            return;
        }
        updateReviewMultiSelectUi(panel);
    }

    function toggleReviewMultiSelectMode() {
        const panel = getReviewStagePanel();
        if (!panel) return;
        setReviewMultiSelectMode(!isReviewMultiSelectMode(panel), panel);
    }

    function handleReviewMultiSelect(checkbox, shiftKey) {
        const panel = getReviewStagePanel();
        const grid = getReviewEventGrid(panel);
        if (!panel || !grid || !checkbox) return;

        const checkboxes = getReviewMultiSelectCheckboxes(panel);
        const currentIndex = Number(checkbox.dataset.index);
        const lastClickedIndex = grid.dataset.lastMultiSelectIndex === undefined
            ? null
            : Number(grid.dataset.lastMultiSelectIndex);

        setReviewCellSelected(checkbox, checkbox.checked);

        if (shiftKey && lastClickedIndex !== null && Number.isFinite(currentIndex)) {
            toggleReviewMultiSelectRange(checkboxes, lastClickedIndex, currentIndex, checkbox.checked);
        }

        if (Number.isFinite(currentIndex)) {
            grid.dataset.lastMultiSelectIndex = String(currentIndex);
        }
        updateReviewMultiSelectUi(panel);
    }

    function collectReviewMultiSelectPayload(panel = getReviewStagePanel()) {
        if (typeof window.WmBatchActions === 'undefined') {
            alert('Batch actions are not available.');
            return null;
        }
        return window.WmBatchActions.getExplicitSelection('[data-review-multi-select-checkbox]');
    }

    function getSelectedReviewMultiSelectDetectionIds(panel = getReviewStagePanel()) {
        return getReviewMultiSelectCheckboxes(panel)
            .filter(function (checkbox) {
                return checkbox.checked;
            })
            .map(function (checkbox) {
                return Number(checkbox.dataset.detectionId || 0);
            })
            .filter(function (detectionId) {
                return Number.isFinite(detectionId) && detectionId > 0;
            });
    }

    function getSelectedReviewMultiSelectCells(panel = getReviewStagePanel()) {
        return getReviewMultiSelectCheckboxes(panel)
            .filter(function (checkbox) {
                return checkbox.checked;
            })
            .map(function (checkbox) {
                return checkbox.closest('.review-event-panel__cell');
            })
            .filter(Boolean);
    }

    function setReviewCellRelabelPending(cell, pending) {
        if (!cell) return;
        const label = cell.querySelector('.review-event-panel__cell-species');
        if (!label) return;
        label.classList.toggle('is-relabel-pending', Boolean(pending));
    }

    function updateReviewCellSpeciesDisplay(cell, speciesPayload) {
        if (!cell || !speciesPayload) return;
        const speciesKey = speciesPayload.species || '';
        const commonName = speciesPayload.commonName || speciesKey.replace(/_/g, ' ') || '';
        const label = cell.querySelector('.review-event-panel__cell-species');
        if (label) {
            label.textContent = commonName || speciesKey;
            if (label.dataset) {
                label.dataset.currentSpecies = speciesKey;
                label.dataset.currentSpeciesCommon = commonName || speciesKey;
            }
            label.classList.add('is-relabel-applied');
        }

        cell.dataset.speciesRelabel = speciesKey;
        cell.dataset.speciesIsManual = '1';
        if (speciesPayload.speciesColour !== undefined && speciesPayload.speciesColour !== null && speciesPayload.speciesColour !== '') {
            const colour = String(speciesPayload.speciesColour);
            cell.dataset.speciesColour = colour;
            cell.style.setProperty('--cell-species-colour', 'var(--species-colour-' + colour + ')');
        }
        syncReviewCellSpeciesArtifacts(
            cell,
            speciesKey,
            commonName,
            speciesPayload.refImageUrl || ''
        );
    }

    async function quickRelabelSelectedReviewFrames(actionBtn, panel = getReviewStagePanel()) {
        if (!panel) return false;

        const detectionIds = getSelectedReviewMultiSelectDetectionIds(panel);
        const cells = getSelectedReviewMultiSelectCells(panel);
        const species = actionBtn.dataset.species || '';
        const commonName = actionBtn.querySelector('.review-stage-panel__species-name')?.textContent?.trim()
            || species.replace(/_/g, ' ');
        const speciesColour = actionBtn.dataset.speciesColour || '';
        const refImageUrl = actionBtn.dataset.speciesRefImageUrl || '';
        const eventKey = panel.dataset.eventKey || '';
        if (!species || detectionIds.length === 0 || cells.length === 0) return false;

        // Detect whether this relabel will split the event. The event
        // builder in core/events.py groups strictly by species — so any
        // subset relabel to a different species creates a second event.
        // We read the current event species off the controls aside; if
        // the target species differs AND we are not relabelling every
        // actionable frame in the event, the event will split and the
        // stale event_key in the panel becomes invalid. In that case a
        // full page reload is the safest way to pick up the new events
        // in the rail.
        const controls = panel.querySelector('[data-review-event-controls]');
        const eventSpecies = controls?.dataset.species || controls?.dataset.originalSpecies || '';
        const actionableIds = (controls?.dataset.actionableDetectionIds || '')
            .split(',')
            .map(function (value) { return Number(value); })
            .filter(function (value) { return Number.isFinite(value) && value > 0; });
        const selectedSet = new Set(detectionIds);
        const actionableSet = new Set(actionableIds);
        const relabelCoversAllActionable = actionableIds.length > 0
            && actionableIds.every(function (id) { return selectedSet.has(id); });
        const speciesDiffers = Boolean(eventSpecies) && eventSpecies !== species;
        const willSplitEvent = speciesDiffers && !relabelCoversAllActionable;
        // A partial-coverage relabel to the SAME species is a no-op for
        // event shape but we still log it; no split toast needed.
        const partialSelection = actionableIds.length > 0
            && !actionableIds.every(function (id) { return selectedSet.has(id); })
            && Array.from(selectedSet).every(function (id) { return actionableSet.has(id); });

        cells.forEach(function (cell) {
            setReviewCellRelabelPending(cell, true);
            updateReviewCellSpeciesDisplay(cell, {
                species: species,
                commonName: commonName,
                speciesColour: speciesColour,
                refImageUrl: refImageUrl,
            });
        });

        try {
            if (typeof window.WmBatchActions !== 'undefined' && typeof window.WmBatchActions.executeBatchAction === 'function') {
                await window.WmBatchActions.executeBatchAction('/api/moderation/bulk/relabel', {
                    detection_ids: detectionIds,
                    species: species,
                });
            } else {
                const response = await fetch('/api/moderation/bulk/relabel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        detection_ids: detectionIds,
                        species: species,
                    }),
                });
                const data = await response.json().catch(function () { return {}; });
                if (!response.ok || data.status !== 'success') {
                    throw new Error(data.message || 'Relabel failed');
                }
            }

            cells.forEach(function (cell) {
                setReviewCellRelabelPending(cell, false);
            });
            reviewPanelCache.clear();
            clearReviewMultiSelect(panel);

            if (willSplitEvent) {
                // Event got split server-side: the remaining frames keep the
                // old species and old event_key, the relabelled frames form
                // a new event with a new key. Neither the rail data nor the
                // event panel in DOM know about the new event yet — reload
                // so the operator can see and approve both events.
                const keepCount = Math.max(0, actionableIds.length - detectionIds.length);
                const splitMsg = detectionIds.length + ' frame'
                    + (detectionIds.length === 1 ? '' : 's')
                    + ' → ' + (commonName || species)
                    + '. Event was split into two events ('
                    + keepCount + '× original species + '
                    + detectionIds.length + '× ' + (commonName || species)
                    + '). Reloading so you can approve each event separately…';
                if (window.wmToast) {
                    window.wmToast(splitMsg, 'warning', 6000);
                }
                window.setTimeout(function () {
                    window.location.reload();
                }, 1800);
                return true;
            }

            if (window.wmToast) {
                const baseMsg = detectionIds.length + ' selected frame'
                    + (detectionIds.length === 1 ? '' : 's')
                    + ' relabelled to ' + (commonName || species) + '.';
                const suffix = partialSelection && !speciesDiffers
                    ? ' (same species, event stays intact.)'
                    : '';
                window.wmToast(baseMsg + suffix, 'success', 2800);
            }
            return true;
        } catch (err) {
            console.error('[review] quick multi-select relabel error', err);
            if (eventKey) {
                reviewPanelCache.delete(getReviewEventCacheKey(eventKey));
                await loadReviewEventPanel(eventKey, { force: true });
            }
            alert(err.message || 'Relabel failed.');
            return false;
        } finally {
            cells.forEach(function (cell) {
                setReviewCellRelabelPending(cell, false);
            });
        }
    }

    async function reviewEventBatchRelabelSelected() {
        const panel = getReviewStagePanel();
        const selection = collectReviewMultiSelectPayload(panel);
        if (!selection) return;
        const result = await window.WmBatchActions.runBatchRelabel(selection, {
            pickerTitle: '🏷️ Relabel selected frames',
            mountEl: document.body
        });
        if (result?.success) {
            setReviewMultiSelectMode(false, panel);
        }
    }

    async function reviewEventBatchTrashSelected() {
        const panel = getReviewStagePanel();
        const selection = collectReviewMultiSelectPayload(panel);
        if (!selection) return;
        const result = await window.WmBatchActions.runBatchAction(
            'Move Selected to Trash',
            selection,
            '/api/moderation/bulk/reject'
        );
        if (!result?.success) return;

        const eventKey = panel?.dataset.eventKey || '';
        const nextEventKey = eventKey ? getNextReviewEventKey(eventKey, 1) : '';
        const removedIds = selection.ids || [];
        reviewPanelCache.clear();
        removedIds.forEach(function (detectionId) {
            removeReviewNodes(getReviewItemKey('detection', detectionId));
        });

        if (eventKey) {
            const remaining = getReviewMultiSelectCheckboxes(panel).filter(function (checkbox) {
                return removedIds.indexOf(Number(checkbox.dataset.detectionId || 0)) === -1;
            });
            if (remaining.length === 0) {
                await applyReviewEventDomRemoval(eventKey, nextEventKey);
            } else {
                await loadReviewEventPanel(eventKey, { force: true });
            }
        }

        updateReviewMeta();
        if (window.wmToast) {
            const count = (result.data?.rejected || removedIds.length || 0);
            window.wmToast(count + ' selected frame' + (count === 1 ? '' : 's') + ' moved to Trash.', 'success', 3200);
        }
    }

    function removeReviewNodes(itemKey) {
        const item = getReviewItem(itemKey);
        if (item) item.remove();
        reviewPanelCache.delete(itemKey);
        reviewQueueIndex.delete(itemKey);
        clearPendingReviewSpecies(itemKey);

        const panel = getReviewStagePanel();
        if (panel && panel.dataset.itemKey === itemKey) {
            panel.innerHTML = '';
            panel.removeAttribute('data-item-kind');
            panel.removeAttribute('data-item-id');
            panel.removeAttribute('data-item-key');
            panel.removeAttribute('data-reason');
            panel.removeAttribute('data-filename');
        }
    }

    function removeReviewEventNode(eventKey) {
        const card = getReviewEvent(eventKey);
        if (card) card.remove();
        reviewPanelCache.delete(getReviewEventCacheKey(eventKey));
        reviewEventIndex.delete(eventKey);

        const panel = getReviewStagePanel();
        if (panel && panel.dataset.panelType === 'event' && panel.dataset.eventKey === eventKey) {
            panel.innerHTML = '';
            panel.removeAttribute('data-event-key');
        }
    }

    async function applyReviewDomRemoval(itemKeys, nextItemKey = null) {
        itemKeys.forEach(removeReviewNodes);
        updateReviewMeta();

        const remaining = getVisibleReviewItems();
        if (remaining.length === 0) {
            location.reload();
            return;
        }

        await selectReviewItem(nextItemKey || remaining[0].dataset.itemKey, { scroll: false });
    }

    async function applyReviewEventDomRemoval(eventKey, nextEventKey = null) {
        removeReviewEventNode(eventKey);
        updateReviewMeta();

        const remaining = getVisibleReviewEvents();
        if (remaining.length > 0) {
            await selectReviewEvent(nextEventKey || remaining[0].dataset.eventKey, { scroll: false });
            return;
        }

        // No events left. If orphans are still open on this page we
        // drill straight into the first one so the user keeps working
        // instead of seeing a misleading "No events" empty stage.
        const firstRailItem = getVisibleReviewItems()[0];
        if (firstRailItem) {
            await selectReviewItem(firstRailItem.dataset.itemKey, { scroll: false });
            return;
        }
        const firstIndexKey = reviewQueueIndex.keys().next().value;
        if (firstIndexKey) {
            await selectReviewItem(firstIndexKey, { scroll: false });
            return;
        }

        renderEmptyEventStage();
    }

    function removeStaleEventsForDetectionIds(detectionIds) {
        if (!Array.isArray(detectionIds) || detectionIds.length === 0) return;
        const targetIds = new Set(detectionIds.map(function (id) { return Number(id); }));
        const staleEventKeys = [];

        reviewEventIndex.forEach(function (event, eventKey) {
            const eventDetectionIds = Array.isArray(event.detection_ids) ? event.detection_ids : [];
            const intersects = eventDetectionIds.some(function (id) {
                return targetIds.has(Number(id));
            });
            if (intersects) staleEventKeys.push(eventKey);
        });

        staleEventKeys.forEach(removeReviewEventNode);
    }

    async function withNextReviewItem(itemKey, work) {
        const nextItemKey = getNextReviewItemKey(itemKey, 1);
        const success = await work();
        if (success) {
            if (itemKey.indexOf('detection:') === 0) {
                removeStaleEventsForDetectionIds([Number(itemKey.split(':')[1] || 0)]);
            }
            await applyReviewDomRemoval([itemKey], nextItemKey);
        }
        return success;
    }

    async function sendReviewDecision(filenames, action) {
        try {
            const response = await fetch('/api/review/decision', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames, action })
            });
            const data = await response.json();
            if (data.status === 'success') return true;
            alert('Error: ' + (data.message || 'Unknown error'));
        } catch (error) {
            console.error('Review decision error:', error);
            alert('Network error. Please try again.');
        }
        return false;
    }

    async function persistReviewBboxState(filename, detectionId, bboxReview) {
        const response = await fetch('/api/review/bbox-review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                detection_id: detectionId,
                bbox_review: bboxReview
            })
        });

        const data = await response.json();
        if (data.status !== 'success') throw new Error(data.message || 'BBox review update failed');
    }

    async function setReviewBboxState(button, bboxReview) {
        const controls = button.closest('[data-review-controls]');
        if (!controls) return;

        const filename = controls.dataset.filename;
        const detectionId = Number(controls.dataset.detectionId || 0);
        const previous = controls.dataset.bboxReview || '';
        const previousExplicit = controls.dataset.bboxReviewExplicit || '0';
        applyReviewBboxUi(controls, bboxReview);
        controls.dataset.bboxReviewExplicit = '1';

        try {
            await persistReviewBboxState(filename, detectionId, bboxReview);
        } catch (error) {
            console.error('BBox review update error:', error);
            controls.dataset.bboxReviewExplicit = previousExplicit;
            applyReviewBboxUi(controls, previous);
            alert(error.message || 'BBox review update failed.');
        }
    }

    async function reviewQuickSpecies(itemKey, filename, detectionId, species) {
        const controls = getReviewControls(itemKey);
        const bboxReview = controls?.dataset.bboxReviewExplicit === '1'
            ? (controls.dataset.bboxReview || null)
            : null;

        try {
            const response = await fetch('/api/review/quick-species', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: filename,
                    detection_id: detectionId,
                    species: species,
                    bbox_review: bboxReview
                })
            });
            const data = await response.json();
            if (data.status !== 'success') throw new Error(data.message || 'Quick species update failed');
            clearPendingReviewSpecies(itemKey);
            await loadReviewPanel(controls.dataset.itemKind, controls.dataset.itemId, { force: true });
            if (window.wmToast) window.wmToast('Species confirmed. Approve when the review is complete.', 'success', 3000);
            return true;
        } catch (error) {
            console.error('Quick species update error:', error);
            alert(error.message || 'Quick species update failed.');
            return false;
        }
    }

    async function reviewApprove(itemKey, filename) {
        const controls = getReviewControls(itemKey);
        const detectionId = Number(controls?.dataset.detectionId || 0);
        const selectedSpecies = controls?.dataset.selectedSpecies || '';
        const bboxReview = controls?.dataset.bboxReview || '';
        if (!detectionId) {
            alert('Review controls not available.');
            return false;
        }
        if (!selectedSpecies || !bboxReview) {
            alert('Review selection is incomplete.');
            return false;
        }

        return withNextReviewItem(itemKey, async () => {
            try {
                const response = await fetch('/api/review/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: filename,
                        detection_id: detectionId,
                        species: selectedSpecies,
                        bbox_review: bboxReview
                    })
                });
                const data = await response.json();
                if (data.status !== 'success') throw new Error(data.message || 'Review approval failed');
                if (window.wmToast) {
                    const tone = data.gallery_visible ? 'success' : 'warning';
                    window.wmToast(
                        data.message || (data.gallery_visible
                            ? 'Approved. The image is now visible in the gallery.'
                            : 'Detection approved. The image stays out of the gallery until remaining open detections on the same photo are resolved.'),
                        tone,
                        4500
                    );
                }
                clearPendingReviewSpecies(itemKey);
                return true;
            } catch (error) {
                console.error('Review approval error:', error);
                alert(error.message || 'Review approval failed.');
                return false;
            }
        });
    }

    // --- Per-cell species relabel (V1, ad-hoc 2026-04-08) ----------------
    //
    // Clicking a cell's species label in the event grid opens the shared
    // WmSpeciesPicker for that single detection. The pick persists
    // immediately via /api/moderation/bulk/relabel (same endpoint the
    // Edit Page + Gallery inline-edit use), so the correction survives
    // page reloads. The event-resolve backend respects the resulting
    // `manual_species_override` and does NOT overwrite it with the
    // event-level species when the operator hits Approve Event —
    // per-frame wins over event-level.
    async function openReviewCellRelabel(btn) {
        if (!window.WmSpeciesPicker || typeof window.WmSpeciesPicker.pickSpecies !== 'function') {
            console.error('[review] WmSpeciesPicker missing');
            return;
        }
        const detectionId = Number(btn.dataset.detectionId || 0);
        if (!detectionId) return;

        const currentSpecies = btn.dataset.currentSpecies || '';
        let picked;
        try {
            picked = await window.WmSpeciesPicker.pickSpecies({
                title: '🏷️ Relabel this frame',
                currentSpecies: currentSpecies,
                detectionId: detectionId,
                mountEl: document.body,
            });
        } catch (err) {
            console.error('[review] species picker failed', err);
            return;
        }
        if (!picked || !picked.scientific) return;
        if (picked.scientific === currentSpecies) return;

        btn.disabled = true;
        btn.classList.add('is-relabel-pending');
        try {
            const response = await fetch('/api/moderation/bulk/relabel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detection_ids: [detectionId],
                    species: picked.scientific,
                }),
            });
            const data = await response.json();
            if (data.status !== 'success') {
                throw new Error(data.message || 'Relabel failed');
            }

            // Update the cell DOM so the operator immediately sees the
            // new species + colour. The backend has already committed.
            const cell = btn.closest('.review-event-panel__cell');
            if (cell) {
                const controls = getReviewStagePanel()?.querySelector('[data-review-event-controls]');
                updateReviewCellSpeciesDisplay(cell, {
                    species: picked.scientific,
                    commonName: picked.common || picked.scientific,
                    refImageUrl: lookupSpeciesRefImageUrl(controls, picked.scientific),
                });
            } else {
                btn.dataset.currentSpecies = picked.scientific;
                btn.dataset.currentSpeciesCommon = picked.common || picked.scientific;
                btn.textContent = picked.common || picked.scientific;
                btn.classList.add('is-relabel-applied');
            }

            if (window.wmToast) {
                window.wmToast(
                    'Frame relabelled to ' + (picked.common || picked.scientific)
                    + '. Save takes effect on Approve Event.',
                    'success',
                    3200
                );
            }
        } catch (err) {
            console.error('[review] relabel error', err);
            alert(err.message || 'Relabel failed.');
        } finally {
            btn.disabled = false;
            btn.classList.remove('is-relabel-pending');
        }
    }

    // --- Mixed-event per-frame decisions ---------------------------------
    //
    // Frame-level `Keep` / `Trash` decisions are held client-side until the
    // operator presses `Approve Event`. Default per frame is `Keep`, so a
    // homogeneous all-correct event still approves in one click. When any
    // frame is marked `Trash`, `Approve Event` switches to the mixed-event
    // resolve endpoint which commits Keep→Gallery and Trash→Trash in one
    // transaction.
    function getFrameDecisionButtons() {
        const panel = getReviewStagePanel();
        if (!panel) return [];
        return Array.from(panel.querySelectorAll('[data-review-frame-decision]'));
    }

    function readFrameDecisions() {
        const result = { keep: [], trash: [] };
        getFrameDecisionButtons().forEach(function (btn) {
            const detectionId = Number(btn.dataset.detectionId || 0);
            if (!detectionId) return;
            const state = btn.dataset.frameState === 'trash' ? 'trash' : 'keep';
            result[state].push(detectionId);
        });
        return result;
    }

    function setFrameDecisionVisual(btn, state) {
        const next = state === 'trash' ? 'trash' : 'keep';
        btn.dataset.frameState = next;
        btn.classList.toggle('is-keep', next === 'keep');
        btn.classList.toggle('is-trash', next === 'trash');
        btn.setAttribute('aria-pressed', next === 'trash' ? 'true' : 'false');
        const copy = btn.querySelector('[data-frame-decision-copy]');
        const icon = btn.querySelector('.review-event-panel__cell-action-icon');
        if (next === 'trash') {
            btn.title = 'Trash this frame. Click to mark as Keep.';
            if (copy) copy.textContent = 'Trash this frame';
            if (icon) icon.textContent = '🗑';
        } else {
            btn.title = 'Keep this frame. Click to mark as Trash.';
            if (copy) copy.textContent = 'Keep this frame';
            if (icon) icon.textContent = '✓';
        }
        const cell = btn.closest('.review-event-panel__cell');
        if (cell) {
            cell.classList.toggle('is-frame-trash', next === 'trash');
            cell.classList.toggle('is-frame-keep', next === 'keep');
        }
    }

    function toggleFrameDecision(btn) {
        if (!btn) return;
        const current = btn.dataset.frameState === 'trash' ? 'trash' : 'keep';
        setFrameDecisionVisual(btn, current === 'trash' ? 'keep' : 'trash');
    }

    function handleStaleEventRaceError(error) {
        // The event was split or rehashed by an earlier relabel. Both
        // `_load_single_review_event` → "no longer exists" and the
        // id-parity branch → "changed and must be reloaded" mean the
        // panel's event_key is stale. Reload so the rail shows the new
        // events and the operator can act on each one separately.
        const msg = error && error.message ? String(error.message) : '';
        const isStale = msg.indexOf('changed and must be reloaded') !== -1
            || msg.indexOf('no longer exists') !== -1;
        if (!isStale) return false;
        if (window.wmToast) {
            window.wmToast(
                'This event was split by an earlier relabel. Reloading so the new events show up…',
                'warning',
                4500
            );
        }
        window.setTimeout(function () {
            window.location.reload();
        }, 1500);
        return true;
    }

    async function reviewResolveEvent(eventKey, controls, decisions) {
        const species = controls.dataset.species || '';
        const bboxReview = controls.dataset.bboxReview || '';
        if (!species || !bboxReview) {
            alert('Event review selection is incomplete.');
            return false;
        }
        if (decisions.keep.length === 0) {
            alert('At least one frame must be kept. Use "Move Event to Trash" to reject every frame at once.');
            return false;
        }

        const nextEventKey = getNextReviewEventKey(eventKey, 1);
        try {
            const response = await fetch('/api/review/event-resolve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    event_key: eventKey,
                    keep_detection_ids: decisions.keep,
                    trash_detection_ids: decisions.trash,
                    species: species,
                    bbox_review: bboxReview
                })
            });
            const data = await response.json();
            if (data.status !== 'success') throw new Error(data.message || 'Event resolve failed');

            clearPendingReviewSpecies(controls.dataset.itemKey || eventKey);
            reviewPanelCache.clear();
            const touchedIds = decisions.keep.concat(decisions.trash);
            touchedIds.forEach(function (detectionId) {
                removeReviewNodes(getReviewItemKey('detection', detectionId));
            });
            await applyReviewEventDomRemoval(eventKey, nextEventKey);
            updateReviewMeta();

            if (window.wmToast) {
                const keptCount = decisions.keep.length;
                const trashedCount = decisions.trash.length;
                window.wmToast(
                    data.message || ('Event resolved — ' + keptCount + ' kept, ' + trashedCount + ' trashed.'),
                    'success',
                    4500
                );
            }
            return true;
        } catch (error) {
            console.error('Event resolve error:', error);
            if (handleStaleEventRaceError(error)) return false;
            alert(error.message || 'Event resolve failed.');
            return false;
        }
    }

    async function reviewApproveEvent(eventKey) {
        const panel = getReviewStagePanel();
        const controls = panel?.querySelector('[data-review-event-controls]');
        if (!controls) {
            alert('Event review controls not available.');
            return false;
        }

        const species = controls.dataset.species || '';
        const bboxReview = controls.dataset.bboxReview || '';
        const detectionIds = (controls.dataset.detectionIds || '')
            .split(',')
            .map(function (value) { return Number(value); })
            .filter(function (value) { return Number.isFinite(value) && value > 0; });

        if (!species || !bboxReview || detectionIds.length === 0) {
            alert('Event review selection is incomplete.');
            return false;
        }

        // Event-ineligible path: the server refuses /api/review/event-approve
        // when an event_key is supplied AND the event's cluster-time
        // eligibility is not `event_eligible` (multi-bird ambiguity, bbox-jump,
        // species-ambiguity). Those events used to be approvable via the
        // now-retired continuity-batch panel, which POSTed to the exact same
        // endpoint but deliberately WITHOUT an event_key — that skips the
        // eligibility gate (see web/blueprints/review.py:2359). After the
        // fixed-5 rewrite we re-use that no-event_key fastpath for ineligible
        // events, limited to non-context (actionable) detection ids so
        // Gallery anchors stay read-only.
        const isEventIneligible = controls.dataset.eventIneligible === '1';
        const actionableIds = (controls.dataset.actionableDetectionIds || '')
            .split(',')
            .map(function (value) { return Number(value); })
            .filter(function (value) { return Number.isFinite(value) && value > 0; });

        // Mixed-event path: if any frame is marked Trash, resolve via the
        // event-resolve endpoint. Homogeneous Keep events take the fast
        // single-call event-approve path (with or without event_key depending
        // on eligibility).
        const decisions = readFrameDecisions();
        if (decisions.trash.length > 0) {
            return reviewResolveEvent(eventKey, controls, decisions);
        }

        const nextEventKey = getNextReviewEventKey(eventKey, 1);

        // Build the fastpath payload. The approval write path always targets
        // actionable detection ids only. Gallery anchors remain visible in
        // the event grid, but they are read-only context and must never be
        // re-confirmed through event approval.
        let submitDetectionIds;
        const fastpathBody = {
            species: species,
            bbox_review: bboxReview,
        };
        if (isEventIneligible) {
            if (actionableIds.length === 0) {
                alert('No actionable frames in this event.');
                return false;
            }
            submitDetectionIds = actionableIds;
            fastpathBody.detection_ids = actionableIds;
        } else {
            submitDetectionIds = actionableIds.length > 0 ? actionableIds : detectionIds;
            fastpathBody.event_key = eventKey;
            fastpathBody.detection_ids = submitDetectionIds;
        }

        try {
            const response = await fetch('/api/review/event-approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(fastpathBody)
            });
            const data = await response.json();
            if (data.status !== 'success') throw new Error(data.message || 'Event approval failed');

            clearPendingReviewSpecies(controls.dataset.itemKey || eventKey);
            reviewPanelCache.clear();
            submitDetectionIds.forEach(function (detectionId) {
                removeReviewNodes(getReviewItemKey('detection', detectionId));
            });
            await applyReviewEventDomRemoval(eventKey, nextEventKey);
            updateReviewMeta();

            if (window.wmToast) {
                const allVisible = Array.isArray(data.gallery_visible_filenames)
                    && Array.isArray(data.filenames)
                    && data.gallery_visible_filenames.length === data.filenames.length;
                window.wmToast(
                    data.message || (allVisible
                        ? 'Event approved. All touched images are now visible in the gallery.'
                        : 'Event approved. Some images remain hidden until other detections are resolved.'),
                    allVisible ? 'success' : 'warning',
                    4500
                );
            }
            return true;
        } catch (error) {
            console.error('Event approval error:', error);
            if (handleStaleEventRaceError(error)) return false;
            alert(error.message || 'Event approval failed.');
            return false;
        }
    }

    async function reviewTrashEvent(eventKey) {
        const panel = getReviewStagePanel();
        const controls = panel?.querySelector('[data-review-event-controls]');
        if (!controls) {
            alert('Event review controls not available.');
            return false;
        }

        // Only reject actionable (non-context) detections. Gallery-anchor
        // context frames were already approved earlier — they stay in the
        // Gallery and must not be touched by Move Event to Trash. The
        // server mirrors this filter, but the client sends the right
        // shape up front to avoid any stale payload race.
        const actionableIds = (controls.dataset.actionableDetectionIds || '')
            .split(',')
            .map(function (value) { return Number(value); })
            .filter(function (value) { return Number.isFinite(value) && value > 0; });
        const allEventIds = (controls.dataset.detectionIds || '')
            .split(',')
            .map(function (value) { return Number(value); })
            .filter(function (value) { return Number.isFinite(value) && value > 0; });
        const detectionIds = actionableIds.length > 0 ? actionableIds : allEventIds;

        if (detectionIds.length === 0) {
            alert('Event detections are not available.');
            return false;
        }

        if (!noBirdConfirmed) {
            if (!confirm('Reject every review detection in this event?\n\nGallery-anchor frames (the ones with the "In Gallery" badge) are not touched — they stay in the Gallery.\n\nImages that only had review detections go to Trash. Images that also have detections in other events keep those and stay.\n\n(This confirmation appears only once per session.)')) {
                return false;
            }
            noBirdConfirmed = true;
            localStorage.setItem('noBirdConfirmed', 'true');
            localStorage.setItem('reviewTrashConfirmed', 'true');
        }

        const nextEventKey = getNextReviewEventKey(eventKey, 1);

        try {
            const response = await fetch('/api/review/event-trash', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    event_key: eventKey,
                    detection_ids: detectionIds
                })
            });
            const data = await response.json();
            if (data.status !== 'success') throw new Error(data.message || 'Event trash failed');

            clearPendingReviewSpecies(controls.dataset.itemKey || eventKey);
            reviewPanelCache.clear();
            detectionIds.forEach(function (detectionId) {
                removeReviewNodes(getReviewItemKey('detection', detectionId));
            });
            await applyReviewEventDomRemoval(eventKey, nextEventKey);
            updateReviewMeta();

            if (window.wmToast) {
                const movedCount = Array.isArray(data.trash_filenames) ? data.trash_filenames.length : 0;
                const tone = movedCount > 0 ? 'warning' : 'info';
                window.wmToast(data.message || 'Event moved to Trash.', tone, 5200);
            }
            return true;
        } catch (error) {
            console.error('Event trash error:', error);
            if (handleStaleEventRaceError(error)) return false;
            alert(error.message || 'Event trash failed.');
            return false;
        }
    }

    // --- Continuity batch -------------------------------------------------

    // Batch-local selection state keyed by batch_key. Holds the currently
    // applied species (if any) so Undo can revert in one DOM transaction
    // and Approve Batch is gated on convergence.
    const reviewBatchState = new Map();

    function getBatchPanelEl() {
        const panel = getReviewStagePanel();
        return panel ? panel.querySelector('[data-review-batch-panel]') : null;
    }

    function getBatchCells(batchPanel) {
        if (!batchPanel) return [];
        return Array.from(batchPanel.querySelectorAll('.review-batch-cell--review'));
    }

    function getActionableDetectionIds(batchPanel) {
        if (!batchPanel) return [];
        return (batchPanel.dataset.batchReviewDetectionIds || '')
            .split(',')
            .map(function (value) { return Number(value); })
            .filter(function (value) { return Number.isFinite(value) && value > 0; });
    }

    function updateBatchApproveGate(batchPanel) {
        if (!batchPanel) return;
        const approveBtn = batchPanel.querySelector('[data-review-panel-action="approve_batch"]');
        if (!approveBtn) return;
        const batchKey = batchPanel.dataset.batchKey || '';
        const state = reviewBatchState.get(batchKey);
        const cells = getBatchCells(batchPanel);
        const expectedCount = cells.length;
        if (
            !state
            || !state.species
            || expectedCount === 0
            || Object.keys(state.byDetection || {}).length !== expectedCount
        ) {
            approveBtn.disabled = true;
            return;
        }
        // All actionable frames must converge on the same species.
        const converged = Object.values(state.byDetection).every(function (value) {
            return value === state.species;
        });
        approveBtn.disabled = !converged;
    }

    function renderBatchReceipt(batchPanel, state) {
        if (!batchPanel) return;
        const receipt = batchPanel.querySelector('[data-review-batch-receipt]');
        if (!receipt) return;
        if (!state || !state.species) {
            receipt.hidden = true;
            receipt.innerHTML = '';
            return;
        }
        const count = Object.keys(state.byDetection || {}).length;
        const label = state.speciesCommon || state.species;
        receipt.hidden = false;
        receipt.innerHTML =
            '<span class="review-batch-panel__receipt-text">'
            + count + ' review frame' + (count === 1 ? '' : 's') + ' → ' + label
            + '</span>'
            + '<button type="button" class="review-batch-panel__receipt-undo"'
            + ' data-review-panel-action="undo_batch_species_change"'
            + ' data-batch-key="' + (batchPanel.dataset.batchKey || '') + '"'
            + ' title="Revert this batch species change">Undo</button>';
    }

    function applyBatchSpeciesToCells(batchPanel, species, speciesCommon) {
        if (!batchPanel) return null;
        const batchKey = batchPanel.dataset.batchKey || '';
        if (!batchKey || !species) return null;
        const cells = getBatchCells(batchPanel);
        if (cells.length === 0) return null;

        const previous = reviewBatchState.get(batchKey) || null;
        const snapshot = previous
            ? {
                species: previous.species,
                speciesCommon: previous.speciesCommon,
                byDetection: Object.assign({}, previous.byDetection || {}),
                cellLabels: Object.assign({}, previous.cellLabels || {})
            }
            : null;

        const byDetection = {};
        const cellLabels = {};
        cells.forEach(function (cell) {
            const detectionId = Number(cell.dataset.detectionId || 0);
            if (!detectionId) return;
            const labelEl = cell.querySelector('[data-batch-cell-species]');
            if (labelEl) {
                if (!snapshot || !(detectionId in snapshot.cellLabels)) {
                    cellLabels[detectionId] = labelEl.textContent || '';
                } else {
                    cellLabels[detectionId] = snapshot.cellLabels[detectionId];
                }
                labelEl.textContent = speciesCommon || species;
            }
            byDetection[detectionId] = species;
            cell.classList.add('is-batch-pending');
            cell.dataset.batchSpecies = species;
        });

        const nextState = {
            species: species,
            speciesCommon: speciesCommon || species,
            byDetection: byDetection,
            cellLabels: snapshot ? snapshot.cellLabels : cellLabels,
            previous: snapshot
        };
        // Preserve the first-seen original labels so repeated applies still
        // know what the pre-batch label was.
        if (!snapshot) {
            nextState.cellLabels = cellLabels;
        }
        reviewBatchState.set(batchKey, nextState);

        renderBatchReceipt(batchPanel, nextState);
        updateBatchApproveGate(batchPanel);
        return nextState;
    }

    function undoBatchSpeciesChange(batchPanel) {
        if (!batchPanel) return;
        const batchKey = batchPanel.dataset.batchKey || '';
        if (!batchKey) return;
        const state = reviewBatchState.get(batchKey);
        if (!state) return;
        const cells = getBatchCells(batchPanel);
        cells.forEach(function (cell) {
            const detectionId = Number(cell.dataset.detectionId || 0);
            const labelEl = cell.querySelector('[data-batch-cell-species]');
            if (labelEl && state.cellLabels && detectionId in state.cellLabels) {
                labelEl.textContent = state.cellLabels[detectionId];
            }
            cell.classList.remove('is-batch-pending');
            delete cell.dataset.batchSpecies;
        });
        reviewBatchState.delete(batchKey);
        renderBatchReceipt(batchPanel, null);
        updateBatchApproveGate(batchPanel);
    }

    function handleApplyBatchSpecies(actionBtn) {
        const batchPanel = getBatchPanelEl();
        if (!batchPanel) return;
        const species = actionBtn.dataset.species || '';
        const speciesCommon = actionBtn.dataset.speciesCommon || species;
        if (!species) return;
        const nextState = applyBatchSpeciesToCells(batchPanel, species, speciesCommon);
        if (!nextState) return;
        if (window.wmToast) {
            const count = Object.keys(nextState.byDetection).length;
            window.wmToast(
                'Applied ' + speciesCommon + ' to ' + count + ' review frame'
                + (count === 1 ? '' : 's') + '. Click Approve Batch to commit.',
                'success',
                3200
            );
        }
    }

    async function reviewApproveBatch(batchPanel) {
        if (!batchPanel) return false;
        const batchKey = batchPanel.dataset.batchKey || '';
        const state = reviewBatchState.get(batchKey);
        if (!state || !state.species) {
            alert('Select a species for the batch before approving.');
            return false;
        }

        const actionableIds = getActionableDetectionIds(batchPanel);
        if (actionableIds.length === 0) {
            alert('No actionable review frames in this batch.');
            return false;
        }

        // Hard guard mirroring the server-side refusal: never submit the
        // context anchor detection ids exposed in the dataset.
        const contextIds = new Set(
            (batchPanel.dataset.batchContextDetectionIds || '')
                .split(',')
                .map(function (value) { return Number(value); })
                .filter(function (value) { return Number.isFinite(value) && value > 0; })
        );
        const submitIds = actionableIds.filter(function (id) { return !contextIds.has(id); });
        if (submitIds.length !== actionableIds.length) {
            console.error('Batch submit blocked: context anchors leaked into actionable set');
            alert('Refusing to submit: Gallery anchors detected in batch payload.');
            return false;
        }

        try {
            const response = await fetch('/api/review/event-approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detection_ids: submitIds,
                    species: state.species,
                    bbox_review: 'correct'
                })
            });
            const data = await response.json();
            if (data.status !== 'success') {
                throw new Error(data.message || 'Batch approval failed');
            }

            reviewBatchState.delete(batchKey);

            // Remove every affected event from the rail and the cache in
            // one pass, then rotate to whatever event is next.
            const affectedEventKeys = Array.from(new Set(
                getBatchCells(batchPanel)
                    .map(function (cell) { return cell.dataset.batchEventKey || ''; })
                    .filter(Boolean)
            ));
            submitIds.forEach(function (detectionId) {
                removeReviewNodes(getReviewItemKey('detection', detectionId));
            });
            let nextEventKey = null;
            affectedEventKeys.forEach(function (key) {
                if (!nextEventKey) {
                    nextEventKey = getNextReviewEventKey(key, 1);
                }
                removeReviewEventNode(key);
            });
            await applyReviewEventDomRemoval(
                batchPanel.dataset.activeEventKey || affectedEventKeys[0] || '',
                nextEventKey
            );
            updateReviewMeta();

            if (window.wmToast) {
                const count = submitIds.length;
                window.wmToast(
                    'Batch approved. ' + count + ' review frame'
                    + (count === 1 ? '' : 's') + ' moved to the gallery.',
                    'success',
                    4500
                );
            }
            return true;
        } catch (error) {
            console.error('Batch approval error:', error);
            alert(error.message || 'Batch approval failed.');
            return false;
        }
    }

    async function reviewOpenSpeciesPicker(itemKey, filename, detectionId, currentSpecies) {
        if (typeof WmSpeciesPicker === 'undefined') {
            alert('Species picker not available.');
            return;
        }

        const controls = getReviewControls(itemKey);
        const selectedSpecies = controls?.dataset.selectedSpecies || currentSpecies || '';
        const choice = await WmSpeciesPicker.pickSpecies({
            currentSpecies: selectedSpecies,
            detectionId: detectionId,
            mountEl: document.body,
            title: 'Set species for review'
        });

        if (choice && controls) {
            applyReviewSpeciesUi(controls, choice.scientific, {
                origin: 'pending',
                commonName: choice.common
            });
            if (window.wmToast) window.wmToast('Species selected. Click again to confirm.', 'success', 2800);
        }
    }

    async function reviewOpenEventSpeciesPicker(eventKey, detectionId, currentSpecies) {
        if (typeof WmSpeciesPicker === 'undefined') {
            alert('Species picker not available.');
            return;
        }

        const panel = getReviewStagePanel();
        const controls = panel?.querySelector('[data-review-event-controls]');
        if (!controls) return;

        const selectedSpecies = controls.dataset.species || currentSpecies || '';
        const choice = await WmSpeciesPicker.pickSpecies({
            currentSpecies: selectedSpecies,
            detectionId: detectionId,
            mountEl: document.body,
            title: 'Set species for this event'
        });

        if (choice) {
            applyReviewSpeciesUi(controls, choice.scientific, {
                origin: 'pending',
                commonName: choice.common
            });
            if (window.wmToast) {
                window.wmToast('Species selected for this event. Approve Event to commit.', 'success', 2800);
            }
        }
    }

    async function confirmReviewSpeciesSelection(actionBtn, panel, itemKey, filename) {
        const detectionId = Number(actionBtn.dataset.detectionId || panel.querySelector('[data-review-controls]')?.dataset.detectionId || 0);
        const species = actionBtn.dataset.species || '';
        if (!detectionId || !species) return;
        await reviewQuickSpecies(itemKey, filename, detectionId, species);
    }

    async function analyzeAction(event, filename) {
        if (event) event.stopPropagation();

        const btn = event ? event.currentTarget : null;
        const originalHtml = btn ? btn.innerHTML : '';

        if (btn) {
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
            btn.disabled = true;
        }

        try {
            const response = await fetch(`/api/review/analyze/${encodeURIComponent(filename)}?force=1`, { method: 'POST' });
            const data = await response.json();

            if (data.status === 'success') {
                if (btn) {
                    btn.classList.add('is-queued');
                    btn.innerHTML = '<span class="review-stage-panel__action-icon">⏳</span><span>Queued</span>';
                }
                waitForReviewDetectionControls(filename);
            } else {
                throw new Error(data.message || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            alert(error.message || 'Network error.');
            if (btn) {
                btn.innerHTML = originalHtml;
                btn.disabled = false;
            }
        }
    }

    async function waitForReviewDetectionControls(itemKey, filename) {
        const maxAttempts = 12;
        const initialDelayMs = 750;
        const maxDelayMs = 6000;
        for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
            const delayMs = Math.min(initialDelayMs * (2 ** attempt), maxDelayMs);
            await new Promise(resolve => setTimeout(resolve, delayMs));

            const active = getActiveReviewItem();
            if (!active || active.dataset.itemKey !== itemKey) return;

            const loaded = await loadReviewPanel(active.dataset.itemKind, active.dataset.itemId, { force: true });
            if (!loaded) continue;

            if (getReviewControls(itemKey)) {
                if (window.wmToast) {
                    window.wmToast('Deep Scan finished. BBox and species review are now available.', 'success', 4500);
                }
                return;
            }
        }
    }

    async function rejectReviewDetection(detectionId) {
        const response = await fetch('/api/detections/reject', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: [Number(detectionId)] })
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || data.message || 'Failed to reject detection');
        }
        return true;
    }

    function modalAction(itemKey, filename, action, detectionId) {
        if ((action === 'trash' || action === 'no_bird') && !noBirdConfirmed) {
            if (!confirm('Move this image to Trash?\n\nIt can still be restored there first and deleted permanently later.\n\n(This confirmation appears only once per session.)')) {
                return;
            }
            noBirdConfirmed = true;
            localStorage.setItem('noBirdConfirmed', 'true');
            localStorage.setItem('reviewTrashConfirmed', 'true');
        }

        if (action === 'trash' && detectionId) {
            return withNextReviewItem(itemKey, async () => {
                try {
                    await rejectReviewDetection(detectionId);
                    return true;
                } catch (error) {
                    console.error('Reject detection error:', error);
                    alert(error.message || 'Reject detection failed.');
                    return false;
                }
            });
        }

        return withNextReviewItem(itemKey, () => sendReviewDecision([filename], action));
    }

    async function toggleOrphanFilter(checkbox) {
        const hide = checkbox.checked;
        localStorage.setItem('reviewHideOrphans', hide ? 'true' : 'false');
        document.querySelectorAll('[data-reason="orphan"]').forEach(el => {
            el.style.display = hide ? 'none' : '';
        });

        const active = getActiveReviewItem();
        if (active && active.style.display === 'none') {
            const firstVisible = getVisibleReviewItems()[0];
            if (firstVisible) await selectReviewItem(firstVisible.dataset.itemKey, { scroll: false, instant: true });
        }
        updateReviewMeta();
    }

    window.reviewEventBatchRelabelSelected = reviewEventBatchRelabelSelected;
    window.reviewEventBatchTrashSelected = reviewEventBatchTrashSelected;
    window.reviewEventBatchCancelMultiSelect = function () {
        setReviewMultiSelectMode(false);
    };

    document.addEventListener('keydown', function (event) {
        if (event.target.matches('input, textarea, select')) return;
        if (event.key === 'ArrowRight') stepReviewItem(1);
        if (event.key === 'ArrowLeft') stepReviewItem(-1);
    });

    document.addEventListener('click', function (event) {
        const copyBtn = event.target.closest('[data-review-event-copy]');
        if (copyBtn) {
            event.preventDefault();
            const label = copyBtn.closest('[data-review-event-label]');
            const text = label?.dataset.eventCopyText || '';
            if (!text) return;
            const finishToast = function (ok) {
                if (!window.wmToast) return;
                if (ok) {
                    window.wmToast('Event label copied to clipboard.', 'success', 1800);
                } else {
                    window.wmToast('Copy failed — please select the text manually.', 'warning', 2800);
                }
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(
                    function () { finishToast(true); },
                    function () { finishToast(false); }
                );
            } else {
                // Fallback: hidden textarea + execCommand, for HTTP or
                // older browsers where navigator.clipboard is gated.
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                let ok = false;
                try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
                document.body.removeChild(ta);
                finishToast(ok);
            }
            return;
        }

        const navBtn = event.target.closest('[data-review-nav]');
        if (navBtn) {
            event.preventDefault();
            stepReviewItem(Number(navBtn.dataset.reviewNav || 0));
            return;
        }

        const viewerToolBtn = event.target.closest('[data-review-viewer-tool]');
        if (viewerToolBtn) {
            event.preventDefault();
            const tool = viewerToolBtn.dataset.reviewViewerTool;
            if (tool === 'zoom' && typeof toggleSmartZoom === 'function') {
                toggleSmartZoom(viewerToolBtn);
            }
            if (tool === 'bbox' && typeof toggleBboxOverlay === 'function') {
                // Post fixed-5 (2026-04-08): the bbox overlay pref is
                // already scope-global in LocalStorage
                // (wmb_review_bbox_pref), so clicking the bbox toggle on
                // one cell should flip every cell in the same
                // wm-viewer-scope in sync — otherwise the cells diverge
                // from the pref and from each other. We compute the
                // target state from the clicked button, then delegate
                // toggleBboxOverlay for every host that's not already in
                // that state. Gallery/Stream/Trash continue to call
                // toggleBboxOverlay directly (single-host scopes), so
                // only the Review surface picks up the multi-host sync.
                // The shared single-host call is: toggleBboxOverlay(viewerToolBtn);
                toggleBboxOverlayForReviewScope(viewerToolBtn);
            }
            return;
        }

        const factsToggle = event.target.closest('[data-review-facts-toggle]');
        if (factsToggle) {
            event.preventDefault();
            toggleReviewFacts(factsToggle);
            return;
        }

        const bboxBtn = event.target.closest('[data-bbox-review-toggle]');
        if (bboxBtn) {
            event.preventDefault();
            setReviewBboxState(bboxBtn, getNextReviewBboxState(bboxBtn.dataset.bboxReviewValue || 'correct'));
            return;
        }

        const eventBboxBtn = event.target.closest('[data-review-event-bbox-toggle]');
        if (eventBboxBtn) {
            event.preventDefault();
            const controls = eventBboxBtn.closest('[data-review-event-controls]');
            if (!controls) return;
            applyReviewEventBboxUi(controls, getNextReviewBboxState(controls.dataset.bboxReview || 'correct'));
            return;
        }

        const reviewItem = event.target.closest('[data-review-item]');
        if (reviewItem) {
            event.preventDefault();
            selectReviewItem(reviewItem.dataset.itemKey);
            return;
        }

        const reviewEvent = event.target.closest('[data-review-event]');
        if (reviewEvent) {
            event.preventDefault();
            selectReviewEvent(reviewEvent.dataset.eventKey);
            return;
        }

        const multiSelectToggleBtn = event.target.closest('[data-review-multi-select-toggle]');
        if (multiSelectToggleBtn) {
            event.preventDefault();
            toggleReviewMultiSelectMode();
            return;
        }

        const multiSelectControl = event.target.closest('[data-review-multi-select-control]');
        if (multiSelectControl) {
            if (!isReviewMultiSelectMode()) return;
            const checkbox = multiSelectControl.querySelector('[data-review-multi-select-checkbox]');
            if (!checkbox) return;
            if (event.target !== checkbox) {
                event.preventDefault();
                checkbox.checked = !checkbox.checked;
            }
            handleReviewMultiSelect(checkbox, event.shiftKey);
            return;
        }

        const multiSelectCell = event.target.closest('.review-event-panel__cell');
        if (multiSelectCell && isReviewMultiSelectMode()) {
            event.preventDefault();
            const checkbox = multiSelectCell.querySelector('[data-review-multi-select-checkbox]');
            if (!checkbox) return;
            checkbox.checked = !checkbox.checked;
            handleReviewMultiSelect(checkbox, event.shiftKey);
            return;
        }

        const frameDecisionBtn = event.target.closest('[data-review-frame-decision]');
        if (frameDecisionBtn) {
            event.preventDefault();
            toggleFrameDecision(frameDecisionBtn);
            return;
        }

        const cellRelabelBtn = event.target.closest('[data-review-cell-relabel]');
        if (cellRelabelBtn) {
            event.preventDefault();
            openReviewCellRelabel(cellRelabelBtn);
            return;
        }

        const openQueueBtn = event.target.closest('[data-review-open-item]');
        if (openQueueBtn) {
            event.preventDefault();
            const itemKey = openQueueBtn.dataset.itemKey || '';
            if (!itemKey) return;
            selectReviewItem(itemKey);
            return;
        }

        const actionBtn = event.target.closest('[data-review-panel-action]');
        if (!actionBtn) return;

        const panel = actionBtn.closest('[data-review-panel]');
        if (!panel) return;

        const action = actionBtn.dataset.reviewPanelAction;
        if (action === 'approve_event') {
            event.preventDefault();
            reviewApproveEvent(actionBtn.dataset.eventKey || panel.dataset.eventKey || '');
            return;
        }

        if (action === 'trash_event') {
            event.preventDefault();
            reviewTrashEvent(actionBtn.dataset.eventKey || panel.dataset.eventKey || '');
            return;
        }

        if (action === 'apply_batch_species') {
            event.preventDefault();
            handleApplyBatchSpecies(actionBtn);
            return;
        }

        if (action === 'undo_batch_species_change') {
            event.preventDefault();
            undoBatchSpeciesChange(getBatchPanelEl());
            return;
        }

        if (action === 'approve_batch') {
            event.preventDefault();
            reviewApproveBatch(getBatchPanelEl());
            return;
        }

        if (action === 'undo_species_change') {
            event.preventDefault();
            // Locate the surrounding controls root (either orphan or
            // event scope). Both carry data-original-species as the
            // anchor the Undo reverts to.
            const controls = actionBtn.closest('[data-review-controls], [data-review-event-controls]');
            if (!controls) return;
            const originalKey = controls.dataset.originalSpecies || '';
            applyReviewSpeciesUi(controls, originalKey, {
                origin: controls.dataset.originalSpeciesOrigin || '',
                persistPending: true,
            });
            if (window.wmToast) window.wmToast('Species change reverted.', 'info', 1800);
            return;
        }

        if (action === 'select_event_species') {
            event.preventDefault();
            const controls = panel.querySelector('[data-review-event-controls]');
            const species = actionBtn.dataset.species || '';
            if (!controls || !species) return;
            if (isReviewMultiSelectMode(panel) && getSelectedReviewMultiSelectDetectionIds(panel).length > 0) {
                quickRelabelSelectedReviewFrames(actionBtn, panel);
                return;
            }
            if ((controls.dataset.species || '') === species) return;
            applyReviewSpeciesUi(controls, species, {
                origin: 'pending',
                commonName: actionBtn.querySelector('.review-stage-panel__species-name')?.textContent?.trim() || '',
                // Propagate the clicked button's colour slot + ref image
                // URL through so mirrorEventSpeciesToAutoCells updates
                // every Review-frame cell's data-species-colour and
                // currentBbox.speciesColour in one go. Without this, the
                // cell keeps its stale slot, and the bbox overlay draws
                // with the previous species' colour — which used to land
                // on slot 7 (pure black) in some events, making the
                // bounding box and its label unreadable.
                speciesColour: actionBtn.dataset.speciesColour || '',
                refImageUrl: actionBtn.dataset.speciesRefImageUrl || ''
            });
            if (window.wmToast) {
                window.wmToast('Species selected for this event. Approve Event to commit.', 'success', 2400);
            }
            return;
        }

        if (action === 'open_event_species_picker') {
            event.preventDefault();
            const detectionId = Number(actionBtn.dataset.detectionId || panel.querySelector('[data-review-event-controls]')?.dataset.detectionId || 0);
            const currentSpecies = actionBtn.dataset.currentSpecies || panel.querySelector('[data-review-event-controls]')?.dataset.species || '';
            if (!detectionId) return;
            reviewOpenEventSpeciesPicker(panel.dataset.eventKey || '', detectionId, currentSpecies);
            return;
        }

        const itemKey = actionBtn.dataset.itemKey || panel.dataset.itemKey || '';
        const filename = actionBtn.dataset.filename || panel.dataset.filename || '';
        if (!itemKey || !filename) return;
        if (action === 'approve_review') {
            event.preventDefault();
            reviewApprove(itemKey, filename);
            return;
        }

        if (action === 'select_species') {
            event.preventDefault();
            const controls = panel.querySelector('[data-review-controls]');
            const species = actionBtn.dataset.species || '';
            if (!controls || !species) return;
            const currentPendingSpecies = getPendingReviewSpecies(itemKey);
            if (currentPendingSpecies && currentPendingSpecies === species) {
                confirmReviewSpeciesSelection(actionBtn, panel, itemKey, filename);
                return;
            }
            applyReviewSpeciesUi(controls, species, { origin: 'pending' });
            if (window.wmToast) window.wmToast('Species selected. Click again to confirm.', 'success', 2400);
            return;
        }

        if (action === 'confirm_species') {
            event.preventDefault();
            confirmReviewSpeciesSelection(actionBtn, panel, itemKey, filename);
            return;
        }

        if (action === 'open_species_picker') {
            event.preventDefault();
            const detectionId = Number(actionBtn.dataset.detectionId || panel.querySelector('[data-review-controls]')?.dataset.detectionId || 0);
            const currentSpecies = actionBtn.dataset.currentSpecies || panel.querySelector('[data-review-controls]')?.dataset.selectedSpecies || '';
            if (!detectionId) return;
            reviewOpenSpeciesPicker(itemKey, filename, detectionId, currentSpecies);
            return;
        }

        if (action === 'trash' || action === 'no_bird') {
            event.preventDefault();
            const detectionId = Number(actionBtn.dataset.detectionId || panel.querySelector('[data-review-controls]')?.dataset.detectionId || 0);
            modalAction(itemKey, filename, action, detectionId);
            return;
        }

        if (action === 'deep_scan') {
            event.preventDefault();
            analyzeAction({ stopPropagation: function () {}, currentTarget: actionBtn }, filename);
            waitForReviewDetectionControls(itemKey, filename);
        }
    });

    document.addEventListener('dblclick', function (event) {
        const actionBtn = event.target.closest('.review-stage-panel__species-btn');
        if (!actionBtn) return;

        const panel = actionBtn.closest('[data-review-panel]');
        if (!panel) return;

        const itemKey = actionBtn.dataset.itemKey || panel.dataset.itemKey || '';
        const filename = actionBtn.dataset.filename || panel.dataset.filename || '';
        const species = actionBtn.dataset.species || '';
        if (!itemKey || !filename || !species) return;

        event.preventDefault();
        const controls = panel.querySelector('[data-review-controls]');
        if (controls && (controls.dataset.selectedSpecies || '') !== species) {
            applyReviewSpeciesUi(controls, species, { origin: 'pending' });
        }
        confirmReviewSpeciesSelection(actionBtn, panel, itemKey, filename);
    });

    document.addEventListener('change', function (event) {
        const hideToggle = event.target.closest('#hide-orphans');
        if (hideToggle) {
            toggleOrphanFilter(hideToggle);
        }
    });

    (function initDecisionStats() {
        fetch('/api/v1/analytics/decisions', { credentials: 'same-origin', cache: 'no-store' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.status !== 'success') return;
                latestDecisionStats = data;
                document.getElementById('dsAutoAccepted').textContent = 'Auto-accepted ' + (data.states?.['null'] || 0);
                document.getElementById('dsManualConfirmed').textContent = 'Manual confirmed ' + (data.manual_confirmed_count || 0);
                document.getElementById('dsConfirmed').textContent = 'Policy confirmed ' + (data.states?.confirmed || 0);
                document.getElementById('dsUncertain').textContent = 'Uncertain ' + (data.states?.uncertain || 0);
                document.getElementById('dsUnknown').textContent = 'Unknown ' + (data.states?.unknown || 0);
                document.getElementById('dsRejected').textContent = 'Rejected ' + (data.states?.rejected || 0);
                document.getElementById('dsReview').textContent = 'Review Queue ' + (data.review_queue_count || 0);
                applyDecisionStatsToReviewMetrics(document);
                document.getElementById('decisionStatsStrip').hidden = false;
            })
            .catch(function () { });
    })();

    (async function initReviewWorkspace() {
        applyReviewMetricsState(document);
        const saved = localStorage.getItem('reviewHideOrphans') === 'true';
        const hideToggle = document.getElementById('hide-orphans');
        if (hideToggle && saved) {
            hideToggle.checked = true;
        }

        const firstEvent = getVisibleReviewEvents()[0];
        if (firstEvent) {
            await selectReviewEvent(firstEvent.dataset.eventKey, { scroll: false, instant: true });
            updateReviewMeta();
            return;
        }

        // No events. Try the rail (if it was rendered for the
        // orphan-only case), then fall back to the queue JSON index
        // so drill-down bootstrap still works on an event-only page
        // that happens to be empty.
        const firstRailItem = getVisibleReviewItems()[0];
        if (firstRailItem) {
            await selectReviewItem(firstRailItem.dataset.itemKey, { scroll: false, instant: true });
            updateReviewMeta();
            return;
        }

        const firstIndexKey = reviewQueueIndex.keys().next().value;
        if (firstIndexKey) {
            await selectReviewItem(firstIndexKey, { scroll: false, instant: true });
            updateReviewMeta();
            return;
        }

        renderEmptyEventStage();
        updateReviewMeta();
    })();

    refreshDeepScanStatus();
    setInterval(refreshDeepScanStatus, 15000);

    window.initReviewDefaultBboxes = initReviewDefaultBboxes;
    window.analyzeAction = analyzeAction;
})();
