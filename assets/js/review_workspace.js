(function () {
    'use strict';

    let noBirdConfirmed = localStorage.getItem('noBirdConfirmed') === 'true'
        || localStorage.getItem('reviewTrashConfirmed') === 'true';
    let reviewMetricsExpanded = localStorage.getItem('reviewMetricsExpanded') !== 'false';
    const REVIEW_PENDING_SPECIES_KEY = 'reviewPendingSpeciesV1';
    const reviewQueueDataEl = document.getElementById('review-queue-data');
    const reviewQueueData = reviewQueueDataEl ? JSON.parse(reviewQueueDataEl.textContent) : [];
    const REVIEW_PANEL_CACHE_LIMIT = 24;
    const reviewPanelCache = new Map();
    let reviewPanelLoadToken = 0;
    let latestDecisionStats = null;
    const reviewPanelPrefetchInFlight = new Set();
    const reviewImagePrefetchInFlight = new Set();
    const reviewQueueIndex = new Map(reviewQueueData.map(item => [item.item_key, item]));

    function getReviewItemKey(itemKind, itemId) {
        return `${itemKind}:${itemId}`;
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

    function getVisibleReviewItems() {
        return Array.from(document.querySelectorAll('.review-queue__item[data-review-item]'))
            .filter(item => item.style.display !== 'none');
    }

    function getActiveReviewItem() {
        return document.querySelector('.review-queue__item.is-active');
    }

    function updateReviewMeta() {
        const items = getVisibleReviewItems();
        const active = getActiveReviewItem();
        const focusEl = document.getElementById('review-focus-meta');
        const countEl = document.getElementById('review-visible-count');

        if (countEl) countEl.textContent = items.length;
        if (!focusEl) return;

        if (!active) {
            focusEl.textContent = `Queue ${items.length}`;
            return;
        }

        const index = items.findIndex(item => item === active);
        focusEl.textContent = `${index + 1} / ${items.length} · ${active.dataset.reason.replace('_', ' ')}`;
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

    async function loadReviewPanel(itemKind, itemId, options = {}) {
        const panel = getReviewStagePanel();
        if (!panel) return false;
        const itemKey = getReviewItemKey(itemKind, itemId);

        if (!options.force && panel.dataset.itemKey === itemKey && panel.innerHTML.trim()) {
            ensurePanelImageLoaded();
            hydrateReviewControls(itemKey);
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
            panel.dataset.itemKind = itemKind;
            panel.dataset.itemId = itemId;
            panel.dataset.itemKey = itemKey;
            panel.dataset.filename = reviewQueueIndex.get(itemKey)?.filename || '';
            panel.dataset.reason = getReviewItem(itemKey)?.dataset.reason || '';
            ensurePanelImageLoaded();
            applyReviewMetricsState(panel);
            applyDecisionStatsToReviewMetrics(panel);
            hydrateReviewControls(itemKey);
            hydrateReviewSpeciesThumbs(panel);
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
        const item = getReviewItem(itemKey);
        if (!item || item.style.display === 'none') return;

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
    }

    function getNextReviewItemKey(itemKey, direction = 1) {
        const items = getVisibleReviewItems();
        const current = getReviewItem(itemKey);
        const index = items.findIndex(item => item === current);
        if (index === -1 || items.length === 0) return null;
        const nextIndex = (index + direction + items.length) % items.length;
        return items[nextIndex]?.dataset.itemKey || null;
    }

    async function stepReviewItem(direction) {
        const active = getActiveReviewItem();
        if (!active) return;
        const nextItemKey = getNextReviewItemKey(active.dataset.itemKey, direction);
        if (nextItemKey) await selectReviewItem(nextItemKey);
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
        const selectedOrigin = species
            ? (options.origin !== undefined ? options.origin : (controls.dataset.selectedSpeciesOrigin || ''))
            : '';
        const persistPending = options.persistPending !== false;
        controls.dataset.selectedSpecies = species || '';
        controls.dataset.selectedSpeciesOrigin = selectedOrigin;
        controls.querySelectorAll('.review-stage-panel__species-btn').forEach(btn => {
            const isSelected = (btn.dataset.species || '') === (species || '');
            btn.classList.toggle('is-selected', isSelected);
            btn.dataset.reviewPanelAction = isSelected ? 'confirm_species' : 'select_species';
        });
        controls.querySelectorAll('[data-current-species]').forEach(btn => {
            btn.dataset.currentSpecies = species || '';
        });
        if (persistPending) {
            setPendingReviewSpecies(controls.dataset.itemKey || '', species || '');
        }
        updateReviewApproveState(controls);
    }

    function updateReviewApproveState(controls) {
        if (!controls) return;
        const approveBtn = controls.querySelector('[data-review-panel-action="approve_review"]');
        if (!approveBtn) return;

        const bboxReview = controls.dataset.bboxReview || '';
        const selectedSpecies = controls.dataset.selectedSpecies || '';
        approveBtn.disabled = !selectedSpecies || !['correct', 'wrong'].includes(bboxReview);
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

    async function withNextReviewItem(itemKey, work) {
        const nextItemKey = getNextReviewItemKey(itemKey, 1);
        const success = await work();
        if (success) {
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
            applyReviewSpeciesUi(controls, choice.scientific, { origin: 'pending' });
            if (window.wmToast) window.wmToast('Species selected. Click again to confirm.', 'success', 2800);
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
        if (!active || active.style.display === 'none') {
            const firstVisible = getVisibleReviewItems()[0];
            if (firstVisible) await selectReviewItem(firstVisible.dataset.itemKey, { scroll: false, instant: true });
        }
        updateReviewMeta();
    }

    document.addEventListener('keydown', function (event) {
        if (event.target.matches('input, textarea, select')) return;
        if (event.key === 'ArrowRight') stepReviewItem(1);
        if (event.key === 'ArrowLeft') stepReviewItem(-1);
    });

    document.addEventListener('click', function (event) {
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
                toggleBboxOverlay(viewerToolBtn);
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

        const reviewItem = event.target.closest('[data-review-item]');
        if (reviewItem) {
            event.preventDefault();
            selectReviewItem(reviewItem.dataset.itemKey);
            return;
        }

        const actionBtn = event.target.closest('[data-review-panel-action]');
        if (!actionBtn) return;

        const panel = actionBtn.closest('[data-review-panel]');
        if (!panel) return;

        const itemKey = actionBtn.dataset.itemKey || panel.dataset.itemKey || '';
        const filename = actionBtn.dataset.filename || panel.dataset.filename || '';
        if (!itemKey || !filename) return;

        const action = actionBtn.dataset.reviewPanelAction;
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
        const panel = getReviewStagePanel();
        if (panel && panel.dataset.itemKey && panel.innerHTML.trim()) {
            setCachedReviewPanel(panel.dataset.itemKey, panel.innerHTML);
        }

        applyReviewMetricsState(document);
        const saved = localStorage.getItem('reviewHideOrphans') === 'true';
        const hideToggle = document.getElementById('hide-orphans');
        if (hideToggle && saved) {
            hideToggle.checked = true;
            await toggleOrphanFilter(hideToggle);
        } else {
            updateReviewMeta();
        }

        const firstVisible = getVisibleReviewItems()[0];
        if (firstVisible) {
            await selectReviewItem(firstVisible.dataset.itemKey, { scroll: false, instant: true });
        }
    })();

    refreshDeepScanStatus();
    setInterval(refreshDeepScanStatus, 15000);

    window.initReviewDefaultBboxes = initReviewDefaultBboxes;
    window.analyzeAction = analyzeAction;
})();
