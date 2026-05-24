/*
 * Review Grid (2026-05-23 redesign — plan 2026-05-23_UI_review-grid-redesign).
 *
 * Slice 2 wiring:
 *  - per-card tile-size stepper (1 / 2 / 4 / 8 / 16 per row), persisted
 *    to localStorage `wmb_review_tile_size` as the page-wide session
 *    default.
 *  - per-card Approve / Trash buttons hitting the existing event-scope
 *    endpoints (`/api/review/event-approve`, `/api/review/event-trash`).
 *
 * Slice 3 will add multi-select + bulk-footer wiring.
 * Slice 4 will add continuity-batch card support.
 *
 * Convention: delegated handlers on `data-review-grid-action` and
 * `data-review-grid-stepper`, not per-card listeners — keeps the JS
 * cheap and survives Slice 3's tile-replacement DOM churn.
 */
(function () {
    'use strict';

    /* ------------------------------------------------------------
     * Global tile size — Apple Photos pattern.
     *
     * One slider in the page header drives a single CSS variable
     * `--review-tile-size` on the .review-grid root. Every card's
     * grid uses `grid-template-columns: repeat(auto-fit,
     * minmax(var(--review-tile-size), 1fr))` so the layout reflows
     * naturally as the slider moves.
     *
     * Hotfix 8 (2026-05-23): the slider range depends on the
     * current view mode because the crop source is only 256×256 —
     * anything beyond ~320px would be visibly upscaled. We split
     * the range by mode and persist per-mode positions:
     *   crop mode  → 120..320 px,  default 240
     *   full mode  → 120..1200 px, default 480
     * Storage keys are mode-suffixed so switching back keeps the
     * size the operator last picked in that mode.
     * ---------------------------------------------------------- */
    const SIZE_STEP = 40;
    const CROP_MAX = 320;
    const FULL_MAX = 1200;
    const SIZE_MIN = 120;
    const CROP_DEFAULT = 240;
    const FULL_DEFAULT = 480;

    function modeSettings(mode) {
        if (mode === 'full') {
            return { min: SIZE_MIN, max: FULL_MAX, def: FULL_DEFAULT,
                     key: 'wmb_review_tile_size_full_px' };
        }
        return { min: SIZE_MIN, max: CROP_MAX, def: CROP_DEFAULT,
                 key: 'wmb_review_tile_size_crop_px' };
    }

    function clampSize(px, settings) {
        if (!Number.isFinite(px)) return settings.def;
        if (px < settings.min) return settings.min;
        if (px > settings.max) return settings.max;
        return Math.round(px / SIZE_STEP) * SIZE_STEP;
    }

    function readModeSize(mode) {
        const s = modeSettings(mode);
        const stored = parseInt(localStorage.getItem(s.key) || '', 10);
        return Number.isFinite(stored) ? clampSize(stored, s) : s.def;
    }

    function writeModeSize(mode, px) {
        try { localStorage.setItem(modeSettings(mode).key, String(px)); } catch (e) { /* private mode */ }
    }

    function applyGlobalSize(px) {
        const mode = currentViewMode();
        const value = clampSize(px, modeSettings(mode)) + 'px';
        const root = document.querySelector('.review-grid');
        if (root) root.style.setProperty('--review-tile-size', value);
        else document.documentElement.style.setProperty('--review-tile-size', value);
        const slider = document.getElementById('review-grid-size-slider');
        if (slider && parseInt(slider.value, 10) !== parseInt(value, 10)) {
            slider.value = String(parseInt(value, 10));
        }
    }

    function syncSliderToMode() {
        const mode = currentViewMode();
        const s = modeSettings(mode);
        const slider = document.getElementById('review-grid-size-slider');
        if (slider) {
            slider.min = String(s.min);
            slider.max = String(s.max);
            slider.step = String(SIZE_STEP);
        }
        const px = readModeSize(mode);
        applyGlobalSize(px);
    }

    function stepGlobalSize(direction) {
        const mode = currentViewMode();
        const current = readModeSize(mode);
        const next = clampSize(current + direction * SIZE_STEP, modeSettings(mode));
        applyGlobalSize(next);
        writeModeSize(mode, next);
    }

    function applySessionDefaultToAllCards() {
        syncSliderToMode();
    }

    function getCsrfToken() {
        // Hidden CSRF inputs are emitted by the base template on
        // protected pages. Fall back to an empty string — server-side
        // returns 403 with a clear message if the token is missing.
        const el = document.querySelector('input[name="_csrf_token"]');
        return el ? el.value : '';
    }

    async function postJson(url, body) {
        const headers = { 'Content-Type': 'application/json' };
        const csrf = getCsrfToken();
        if (csrf) headers['X-CSRF-Token'] = csrf;
        const res = await fetch(url, {
            method: 'POST',
            credentials: 'same-origin',
            headers: headers,
            body: JSON.stringify(body || {})
        });
        const data = await res.json().catch(function () { return {}; });
        if (!res.ok) {
            const err = new Error(data.message || ('HTTP ' + res.status));
            err.status = res.status;
            err.data = data;
            throw err;
        }
        return data;
    }

    function reloadReviewEventsSoon(message, tone) {
        if (window.wmToast && message) {
            window.wmToast(message, tone || 'info', 3500);
        }
        window.setTimeout(function () { window.location.reload(); }, 900);
    }

    function isStaleReviewEventError(err) {
        const message = (err && err.message ? err.message : '').toLowerCase();
        return message.indexOf('no longer exists') !== -1
            || message.indexOf('event changed') !== -1
            || message.indexOf('must be reloaded') !== -1;
    }

    function parseDetectionIds(rawValue) {
        return (rawValue || '')
            .split(',')
            .map(function (s) { return parseInt(s.trim(), 10); })
            .filter(function (n) { return Number.isFinite(n) && n > 0; });
    }

    function removeCard(card, label) {
        if (!card) return;
        card.style.transition = 'opacity 220ms ease, transform 220ms ease';
        card.style.opacity = '0';
        card.style.transform = 'scale(0.98)';
        setTimeout(function () { card.remove(); refreshVisibleCount(); }, 240);
        if (label && window.wmToast) {
            window.wmToast(label, 'success', 2500);
        }
    }

    function refreshVisibleCount() {
        const remaining = document.querySelectorAll('[data-review-grid-card]').length;
        const badge = document.getElementById('review-visible-count');
        if (badge) badge.textContent = String(remaining);
        const meta = document.getElementById('review-grid-meta');
        if (meta) meta.textContent = remaining + ' event' + (remaining === 1 ? '' : 's');
    }

    /* 2026-05-23 evening — Smart-Mode for card-header buttons.
     *
     * Card-header Approve / Relabel / Trash / Mark No Bird now respect
     * the operator's selection: if any tile checkbox in the card is
     * checked, the action targets only those frames. If nothing is
     * checked, it falls back to every actionable frame in the event
     * (the legacy event-wide behaviour). Removes the surprise where
     * a header click ignored visible checkboxes.
     *
     * Returns { ids, isSelectionScoped, count } so handlers can render
     * confirmation prompts + toasts in the right scope language.
     */
    function getCardScopeIds(card) {
        const selectedIds = Array.from(card.querySelectorAll('.review-grid__tile:not([data-review-grid-removing]) .review-grid__tile-checkbox:checked'))
            .map(function (cb) { return parseInt(cb.dataset.detectionId, 10); })
            .filter(function (n) { return Number.isFinite(n); });
        if (selectedIds.length > 0) {
            return { ids: selectedIds, isSelectionScoped: true, count: selectedIds.length };
        }
        const allIds = parseDetectionIds(card.dataset.actionableDetectionIds || '');
        return { ids: allIds, isSelectionScoped: false, count: allIds.length };
    }

    function readRemainingActionableIds(card) {
        if (!card) return [];
        return Array.from(card.querySelectorAll('.review-grid__tile:not(.review-grid__tile--context):not([data-review-grid-removing])'))
            .map(function (tile) { return parseInt(tile.dataset.detectionId, 10); })
            .filter(function (n) { return Number.isFinite(n) && n > 0; });
    }

    function syncCardActionableState(card) {
        if (!card) return;
        const ids = readRemainingActionableIds(card);
        card.dataset.actionableDetectionIds = ids.join(',');
        const countEl = card.querySelector('.review-grid__card-count');
        if (countEl) countEl.textContent = ids.length + ' frame' + (ids.length === 1 ? '' : 's');
        syncCardSelectionUi(card);
    }

    function updateTilesSpeciesMetadata(ids, chosen) {
        const species = chosen && chosen.scientific;
        if (!species) return;
        const label = chosen.common || species.replace(/_/g, ' ');
        ids.map(Number).forEach(function (id) {
            const tile = document.querySelector('.review-grid__tile[data-detection-id="' + id + '"]');
            if (!tile) return;
            tile.dataset.species = species;
            tile.querySelectorAll('[data-current-species]').forEach(function (el) {
                el.dataset.currentSpecies = species;
            });
            const img = tile.querySelector('.review-grid__tile-image');
            if (img) img.alt = label;
        });
    }

    /* Update the card-header action button labels + tooltips to reflect
     * the current selection state. Fires on every checkbox change.
     * Uses textContent throughout — no innerHTML — because labels come
     * from data-* attributes set in the Jinja template and we want to
     * keep this XSS-safe even if the macro author writes carelessly. */
    function refreshCardActionLabels(card) {
        if (!card) return;
        const scope = getCardScopeIds(card);
        const buttons = card.querySelectorAll('[data-review-grid-action][data-scope-label-event]');
        buttons.forEach(function (btn) {
            const eventLabel = btn.dataset.scopeLabelEvent || '';
            const selLabel = btn.dataset.scopeLabelSelection || '';
            const eventTip = btn.dataset.scopeTipEvent || btn.title;
            const selTip = btn.dataset.scopeTipSelection || btn.title;
            if (scope.isSelectionScoped) {
                btn.textContent = selLabel.replace('{N}', String(scope.count));
                btn.title = selTip.replace('{N}', String(scope.count));
            } else {
                btn.textContent = eventLabel;
                btn.title = eventTip;
            }
        });
        updateCardApproveState(card, scope);
    }

    function refreshAllCardActionLabels() {
        document.querySelectorAll('[data-review-grid-card]').forEach(refreshCardActionLabels);
    }

    function updateCardApproveState(card, scope) {
        const approveBtn = card.querySelector('[data-review-grid-action="approve_event"]');
        if (!approveBtn) return;

        const species = (card.dataset.species || '').trim();
        const blockedReason = card.dataset.approveBlockedReason || 'Select frames before approving this event';

        if (!species) {
            approveBtn.disabled = true;
            approveBtn.title = 'Pick a species before approving';
            return;
        }
        if (!scope || scope.count === 0) {
            approveBtn.disabled = true;
            approveBtn.title = 'No actionable frames to approve';
            return;
        }
        if (scope.isSelectionScoped) {
            approveBtn.disabled = false;
            return;
        }
        if (card.dataset.eventEligible !== '1') {
            approveBtn.disabled = true;
            approveBtn.title = blockedReason;
            return;
        }
        approveBtn.disabled = false;
        if (blockedReason) approveBtn.title = blockedReason;
    }

    async function approveEvent(card) {
        const eventKey = card.dataset.eventKey;
        const species = card.dataset.species;
        const scope = getCardScopeIds(card);
        const eventEligible = card.dataset.eventEligible === '1';
        const blockedReason = card.dataset.approveBlockedReason || 'Select frames before approving this event';
        if (!eventKey || !species || scope.count === 0) {
            if (window.wmToast) window.wmToast('Approve refused — missing event key, species, or detections', 'error', 3500);
            return;
        }
        if (!scope.isSelectionScoped && !eventEligible) {
            if (window.wmToast) window.wmToast(blockedReason, 'warning', 3500);
            return;
        }
        try {
            const payload = {
                species: species,
                bbox_review: 'correct',
                detection_ids: scope.ids
            };
            // Selection-scoped approval is intentionally ID-scoped.
            // Blind event-wide approval still goes through the
            // event-key safety gate.
            if (!scope.isSelectionScoped && eventEligible) {
                payload.event_key = eventKey;
            }
            await postJson('/api/review/event-approve', payload);
            if (scope.isSelectionScoped) {
                // Partial approval — remove the approved tiles but leave
                // the card if other actionable frames remain.
                removeTilesByDetectionIds(scope.ids);
                if (window.wmToast) window.wmToast(scope.count + ' frame(s) approved', 'success', 2500);
            } else {
                removeCard(card, 'Event approved');
            }
        } catch (err) {
            if (isStaleReviewEventError(err)) {
                reloadReviewEventsSoon('Review changed — reloading events…', 'info');
                return;
            }
            if (window.wmToast) window.wmToast('Approve failed: ' + err.message, 'error', 4000);
        }
    }

    async function trashEvent(card) {
        const eventKey = card.dataset.eventKey;
        if (!eventKey) {
            if (window.wmToast) window.wmToast('Trash refused — missing event key', 'error', 3500);
            return;
        }
        const scope = getCardScopeIds(card);
        if (scope.count === 0) {
            if (window.wmToast) window.wmToast('Nothing to trash.', 'warning', 2500);
            return;
        }
        const promptText = scope.isSelectionScoped
            ? 'Move ' + scope.count + ' selected frame(s) to Trash?'
            : 'Move every actionable frame in this event to Trash?';
        if (!confirm(promptText)) return;
        try {
            if (scope.isSelectionScoped) {
                // Per-detection bulk-reject — keeps the event open if
                // other frames remain.
                await postJson('/api/moderation/bulk/reject', { detection_ids: scope.ids });
                removeTilesByDetectionIds(scope.ids);
                if (window.wmToast) window.wmToast(scope.count + ' frame(s) moved to Trash', 'success', 2500);
            } else {
                await postJson('/api/review/event-trash', {
                    event_key: eventKey,
                    detection_ids: scope.ids
                });
                removeCard(card, 'Event moved to Trash');
            }
        } catch (err) {
            if (isStaleReviewEventError(err)) {
                removeCard(card, null);
                reloadReviewEventsSoon('Event already gone — reloading…', 'info');
                return;
            }
            if (window.wmToast) window.wmToast('Trash failed: ' + err.message, 'error', 4000);
        }
    }

    async function noBirdEvent(card) {
        const scope = getCardScopeIds(card);
        if (scope.count === 0) {
            if (window.wmToast) window.wmToast('Nothing to mark.', 'warning', 2500);
            return;
        }
        // Mark No Bird is per-image-filename, not per-detection. Resolve
        // the checked tiles back to their image filenames so the
        // /api/review/decision endpoint can flag the source images.
        const tilesInScope = scope.isSelectionScoped
            ? Array.from(card.querySelectorAll('.review-grid__tile-checkbox:checked'))
                .map(function (cb) { return cb.closest('.review-grid__tile'); })
                .filter(Boolean)
            : Array.from(card.querySelectorAll('.review-grid__tile[data-image-filename]:not([data-context-only])'));
        const filenames = Array.from(new Set(
            tilesInScope.map(function (t) { return t.dataset.imageFilename; }).filter(Boolean)
        ));
        if (!filenames.length) {
            if (window.wmToast) window.wmToast('No image filenames found.', 'error', 3500);
            return;
        }
        const promptText = scope.isSelectionScoped
            ? 'Mark ' + filenames.length + ' selected image(s) as No Bird? (Training signal — distinct from Trash)'
            : 'Mark every image in this event as No Bird? (Training signal — distinct from Trash)';
        if (!confirm(promptText)) return;
        try {
            await postJson('/api/review/decision', {
                filenames: filenames,
                action: 'no_bird'
            });
            const ids = scope.ids;
            if (scope.isSelectionScoped) {
                removeTilesByDetectionIds(ids);
                if (window.wmToast) window.wmToast(filenames.length + ' image(s) marked No Bird', 'success', 2500);
            } else {
                removeCard(card, filenames.length + ' image(s) marked No Bird');
            }
        } catch (err) {
            if (window.wmToast) window.wmToast('Mark No Bird failed: ' + err.message, 'error', 4000);
        }
    }

    /* Apple-Photos click semantics for tile images (Hotfix 5):
     *   single-click  → toggle the tile's selection checkbox
     *   shift+click   → range-select from last clicked
     *   double-click  → open detection modal at full size
     *
     * The single-click handler uses a small debounce so that a
     * double-click doesn't fire both an unwanted toggle + the modal
     * open. We schedule the toggle on a 220 ms timer that gets
     * cancelled by the dblclick handler.
     */
    let lastClickedTileIndex = null;
    const SINGLE_CLICK_DELAY = 220;
    const pendingSingleClick = new WeakMap();

    function tileCheckboxes(card) {
        return Array.from(card.querySelectorAll('.review-grid__tile:not([data-review-grid-removing]) .review-grid__tile-checkbox'));
    }

    function setTileSelectionState(checkbox, checked) {
        checkbox.checked = checked;
        const tile = checkbox.closest('.review-grid__tile');
        if (tile) tile.classList.toggle('is-selected', checked);
    }

    function syncCardSelectionUi(card) {
        updateCardSelectionCounter(card);
        refreshCardActionLabels(card);
    }

    function toggleTileSelection(tileImage, shiftKey) {
        const card = tileImage.closest('[data-review-grid-card]');
        const tile = tileImage.closest('.review-grid__tile');
        if (!card || !tile) return;
        const cb = tile.querySelector('.review-grid__tile-checkbox');
        if (!cb) return;
        const all = tileCheckboxes(card);
        const idx = all.indexOf(cb);
        if (shiftKey && lastClickedTileIndex !== null && idx !== -1) {
            const lo = Math.min(lastClickedTileIndex, idx);
            const hi = Math.max(lastClickedTileIndex, idx);
            const targetState = !cb.checked;
            for (let i = lo; i <= hi; i++) {
                setTileSelectionState(all[i], targetState);
            }
        } else {
            setTileSelectionState(cb, !cb.checked);
        }
        if (idx !== -1) lastClickedTileIndex = idx;
        syncCardSelectionUi(card);
    }

    function invertCardSelection(card) {
        const checkboxes = tileCheckboxes(card);
        if (!checkboxes.length) {
            if (window.wmToast) window.wmToast('No actionable frames to invert.', 'warning', 2500);
            return;
        }
        checkboxes.forEach(function (cb) {
            setTileSelectionState(cb, !cb.checked);
        });
        lastClickedTileIndex = null;
        syncCardSelectionUi(card);
    }

    function openTileModal(tileImage) {
        const target = tileImage.dataset.modalTarget;
        if (!target) return;
        const modalEl = document.querySelector(target);
        if (!modalEl) return;
        if (window.bootstrap && window.bootstrap.Modal) {
            const instance = window.bootstrap.Modal.getOrCreateInstance(modalEl);
            instance.show();
        }
    }

    document.addEventListener('click', function (event) {
        const tileImage = event.target.closest('[data-tile-image]');
        if (tileImage) {
            // Ignore clicks that bubble up from the toolbox or the
            // selection checkbox — those have their own handlers.
            if (event.target.closest('.wm-toolbox, .review-grid__tile-select')) return;
            event.preventDefault();
            // Cancel any pending single-click from a recent click,
            // then schedule a new one. The dblclick handler clears it.
            const prev = pendingSingleClick.get(tileImage);
            if (prev) clearTimeout(prev);
            const shiftKey = event.shiftKey;
            const timer = setTimeout(function () {
                pendingSingleClick.delete(tileImage);
                toggleTileSelection(tileImage, shiftKey);
            }, SINGLE_CLICK_DELAY);
            pendingSingleClick.set(tileImage, timer);
            return;
        }
    });

    document.addEventListener('dblclick', function (event) {
        const tileImage = event.target.closest('[data-tile-image]');
        if (!tileImage) return;
        if (event.target.closest('.wm-toolbox, .review-grid__tile-select')) return;
        event.preventDefault();
        // Cancel the pending single-click toggle from the first click
        // of the dblclick pair.
        const pending = pendingSingleClick.get(tileImage);
        if (pending) {
            clearTimeout(pending);
            pendingSingleClick.delete(tileImage);
        }
        openTileModal(tileImage);
    });

    document.addEventListener('click', function (event) {
        const sizeBtn = event.target.closest('[data-review-grid-size-action]');
        if (sizeBtn) {
            event.preventDefault();
            const direction = sizeBtn.dataset.reviewGridSizeAction === 'grow' ? +1 : -1;
            stepGlobalSize(direction);
            return;
        }

        const actionBtn = event.target.closest('[data-review-grid-action]');
        if (actionBtn) {
            event.preventDefault();
            if (actionBtn.disabled) return;
            const card = actionBtn.closest('[data-review-grid-card]');
            const verb = actionBtn.dataset.reviewGridAction;
            if (verb === 'approve_event' && card) {
                approveEvent(card);
            } else if (verb === 'trash_event' && card) {
                trashEvent(card);
            } else if (verb === 'relabel_event' && card) {
                relabelEvent(card, actionBtn);
            } else if (verb === 'no_bird_event' && card) {
                noBirdEvent(card);
            } else if (verb === 'invert_selection' && card) {
                invertCardSelection(card);
            } else if (verb === 'apply_batch_species') {
                applyBatchSpecies(actionBtn);
            } else if (verb === 'approve_batch') {
                approveBatch(actionBtn);
            }
        }
    });

    /* 2026-05-23 evening — wire checkbox change → card label refresh.
     * Smart-Mode buttons need to flip labels the moment selection
     * state changes. One delegated listener at document level catches
     * every .review-grid__tile-checkbox change without per-card hookup. */
    document.addEventListener('change', function (event) {
        const cb = event.target.closest('.review-grid__tile-checkbox');
        if (!cb) return;
        const card = cb.closest('[data-review-grid-card]');
        if (card) refreshCardActionLabels(card);
    });

    // Initial pass on page load (selection might be empty, but this
    // also ensures the data-scope-label-event default labels stick).
    document.addEventListener('DOMContentLoaded', refreshAllCardActionLabels);

    /* Hotfix 3b — Relabel Event from the card header.
     * Opens WmSpeciesPicker for the event's cover detection, then
     * POSTs /api/moderation/bulk/relabel for every actionable
     * detection_id in the card. The card's data-species + visible
     * species pill update in-place so the next Approve picks up the
     * new species without a page reload.
     */
    async function relabelEvent(card, btn) {
        if (!window.WmSpeciesPicker || typeof window.WmSpeciesPicker.pickSpecies !== 'function') {
            if (window.wmToast) window.wmToast('Species picker unavailable on this page.', 'error', 3500);
            return;
        }
        const scope = getCardScopeIds(card);
        if (!scope.count) {
            if (window.wmToast) window.wmToast('No actionable frames to relabel.', 'warning', 2500);
            return;
        }
        const currentSpecies = btn.dataset.currentSpecies || card.dataset.species || '';
        let chosen = null;
        try {
            chosen = await window.WmSpeciesPicker.pickSpecies({
                currentSpecies: currentSpecies,
                detectionId: scope.ids[0],
                title: scope.isSelectionScoped
                    ? 'Relabel ' + scope.count + ' selected frame(s)'
                    : 'Relabel event'
            });
        } catch (e) {
            return;
        }
        if (!chosen || !chosen.scientific) return;
        if (chosen.scientific === currentSpecies) {
            if (window.wmToast) window.wmToast('Same species — no change.', 'info', 2500);
            return;
        }
        try {
            await postJson('/api/moderation/bulk/relabel', {
                detection_ids: scope.ids,
                species: chosen.scientific
            });
            updateTilesSpeciesMetadata(scope.ids, chosen);

            const allIds = parseDetectionIds(card.dataset.actionableDetectionIds || '');
            const selectedSet = new Set(scope.ids.map(Number));
            const relabelCoversAllActionable = allIds.length > 0
                && allIds.every(function (id) { return selectedSet.has(id); });
            const relabelChangesEventKey = chosen.scientific !== currentSpecies;

            // Update card-level species when relabeling the whole event.
            // A species change also changes the server-side event_key,
            // so the honest UI update is an automatic reload after the
            // operator sees the receipt.
            if (!scope.isSelectionScoped || relabelCoversAllActionable) {
                card.dataset.species = chosen.scientific;
                const speciesEl = card.querySelector('.review-grid__card-species');
                if (speciesEl) speciesEl.textContent = chosen.common || chosen.scientific.replace(/_/g, ' ');
                btn.dataset.currentSpecies = chosen.scientific;
                // Re-enable Approve if it was disabled for missing species.
                const approveBtn = card.querySelector('[data-review-grid-action="approve_event"]');
                if (approveBtn) refreshCardActionLabels(card);
            }
            if (relabelChangesEventKey) {
                reloadReviewEventsSoon(
                    scope.count + ' frame(s) relabeled to ' + (chosen.common || chosen.scientific)
                    + '. Reloading review events…',
                    'success'
                );
                return;
            }
            if (window.wmToast) window.wmToast(scope.count + ' frame(s) relabeled to ' + (chosen.common || chosen.scientific), 'success', 3500);
        } catch (err) {
            if (window.wmToast) window.wmToast('Relabel failed: ' + err.message, 'error', 4000);
        }
    }

    /* ------------------------------------------------------------
     * Slice 4 — Continuity batch actions.
     *
     * Apply species: POST /api/moderation/bulk/relabel with the batch's
     *   actionable detection ids (every member of every event in the
     *   batch). Then update each card's data-species so subsequent
     *   Approve Batch / Approve Event picks the new species.
     *
     * Approve batch: POST /api/review/event-approve with **no**
     *   event_key (UI_STANDARD §6c rule 7) and the union of
     *   review_detection_ids, after filtering context-only ids on the
     *   client (rule 5 mirror). Each batch-member card is then removed.
     * ---------------------------------------------------------- */

    function batchCards(batchKey) {
        return Array.from(document.querySelectorAll(
            '[data-review-grid-card][data-continuity-batch-key="' + batchKey + '"]'
        ));
    }

    function batchReviewDetectionIds(batchKey) {
        // Merge per-card data-batch-review-detection-ids. They are
        // identical across siblings of the same batch but reading them
        // from the DOM keeps the JS state-free.
        const cards = batchCards(batchKey);
        const ids = new Set();
        cards.forEach(function (card) {
            (card.dataset.batchReviewDetectionIds || '').split(',').forEach(function (raw) {
                const n = parseInt(raw, 10);
                if (n > 0) ids.add(n);
            });
        });
        return Array.from(ids);
    }

    async function applyBatchSpecies(btn) {
        const batchKey = btn.dataset.batchKey;
        const species = btn.dataset.batchRecommendedSpecies;
        if (!batchKey || !species) return;
        const ids = batchReviewDetectionIds(batchKey);
        if (!ids.length) {
            if (window.wmToast) window.wmToast('Batch has no actionable frames.', 'error', 3500);
            return;
        }
        if (!confirm('Relabel ' + ids.length + ' frame(s) to ' + species.replace(/_/g, ' ') + '?')) return;
        try {
            await postJson('/api/moderation/bulk/relabel', {
                detection_ids: ids,
                species: species
            });
            // Update every card in the batch so Approve picks the new
            // species.
            batchCards(batchKey).forEach(function (card) {
                card.dataset.species = species;
                const speciesEl = card.querySelector('.review-grid__card-species');
                if (speciesEl) speciesEl.textContent = species.replace(/_/g, ' ');
            });
            if (window.wmToast) window.wmToast(ids.length + ' frame(s) relabeled · Approve Batch is now ready', 'success', 3500);
        } catch (err) {
            if (window.wmToast) window.wmToast('Apply Species failed: ' + err.message, 'error', 4000);
        }
    }

    async function approveBatch(btn) {
        const batchKey = btn.dataset.batchKey;
        if (!batchKey) return;
        const cards = batchCards(batchKey);
        if (!cards.length) {
            if (window.wmToast) window.wmToast('Batch has no rendered cards.', 'error', 3500);
            return;
        }
        // UI_STANDARD §6c rule 6: convergence on a single species.
        const speciesSet = new Set(cards.map(function (c) { return c.dataset.species || ''; }).filter(Boolean));
        if (speciesSet.size !== 1) {
            if (window.wmToast) window.wmToast('Batch events disagree on species — relabel first.', 'warning', 3500);
            return;
        }
        const species = Array.from(speciesSet)[0];
        const reviewIds = batchReviewDetectionIds(batchKey);
        if (!reviewIds.length) {
            if (window.wmToast) window.wmToast('Batch has no actionable frames left.', 'info', 3500);
            return;
        }
        if (!confirm('Approve every actionable frame in this batch (' + reviewIds.length + ' frame(s)) as ' + species.replace(/_/g, ' ') + '?')) return;
        try {
            // §6c rule 7: POST without event_key.
            await postJson('/api/review/event-approve', {
                species: species,
                bbox_review: 'correct',
                detection_ids: reviewIds
            });
            cards.forEach(function (card) { removeCard(card, null); });
            if (window.wmToast) window.wmToast('Batch approved · ' + reviewIds.length + ' frame(s) confirmed', 'success', 3500);
        } catch (err) {
            if (window.wmToast) window.wmToast('Approve Batch failed: ' + err.message, 'error', 4000);
        }
    }

    /* ------------------------------------------------------------
     * Slice 3 — multi-select + bulk-footer wiring.
     *
     * Selection is page-scoped: a Trash Selected can span multiple
     * events. Approve Selected groups picks by event_key and posts
     * one /api/review/event-approve per event. Mark No Bird Selected
     * resolves the picks to image filenames via
     * /api/moderation/resolve-selection, then POSTs /api/review/decision
     * with action=no_bird. Relabel Selected reuses the existing
     * runBatchRelabel helper.
     * ---------------------------------------------------------- */

    function removeTilesByDetectionIds(ids) {
        const set = new Set(ids.map(Number));
        const affectedCards = new Set();
        set.forEach(function (id) {
            const tile = document.querySelector('.review-grid__tile[data-detection-id="' + id + '"]');
            if (tile) {
                const card = tile.closest('[data-review-grid-card]');
                if (card) affectedCards.add(card);
                tile.dataset.reviewGridRemoving = '1';
                const cb = tile.querySelector('.review-grid__tile-checkbox');
                if (cb) cb.checked = false;
                tile.style.transition = 'opacity 200ms ease';
                tile.style.opacity = '0';
                setTimeout(function () {
                    tile.remove();
                    if (card) syncCardActionableState(card);
                    pruneEmptyCards();
                }, 220);
            }
        });
        affectedCards.forEach(syncCardActionableState);
    }

    function pruneEmptyCards() {
        document.querySelectorAll('[data-review-grid-card]').forEach(function (card) {
            const remaining = card.querySelectorAll('.review-grid__tile:not(.review-grid__tile--context)').length;
            if (remaining === 0) {
                removeCard(card, null);
            }
        });
    }

    /* 2026-05-23 evening — Bulk-footer + cross-event bulk handlers
       removed in Slice 8. Card-header Smart-Mode covers every per-card
       bulk path; cross-event bulk is no longer reachable from this
       page. removeTilesByDetectionIds stays because it is shared by
       the Smart-Mode handlers and Apple-Photos tile-click flow. */

    function updateCardSelectionCounter(card) {
        const tiles = card.querySelectorAll('.review-grid__tile:not([data-review-grid-removing]) .review-grid__tile-checkbox');
        const checked = card.querySelectorAll('.review-grid__tile:not([data-review-grid-removing]) .review-grid__tile-checkbox:checked').length;
        const total = tiles.length;
        const counterEl = card.querySelector('[data-review-grid-card-selected]');
        if (counterEl) {
            if (checked > 0) {
                counterEl.textContent = checked + ' of ' + total + ' selected';
                counterEl.hidden = false;
            } else {
                counterEl.textContent = '0 selected';
                counterEl.hidden = true;
            }
        }
        card.classList.toggle('has-selection', checked > 0);
    }

    document.addEventListener('change', function (event) {
        const checkbox = event.target;
        if (!(checkbox && checkbox.classList && checkbox.classList.contains('review-grid__tile-checkbox'))) {
            return;
        }
        // Hotfix 3c: visual highlight on the tile + per-card counter.
        const tile = checkbox.closest('.review-grid__tile');
        if (tile) tile.classList.toggle('is-selected', checkbox.checked);
        const card = checkbox.closest('[data-review-grid-card]');
        if (card) updateCardSelectionCounter(card);
    });

    /* Hotfix 7 — bbox-overlay toggle.
     *
     * Each tile carries data-bbox-{x,y,w,h} (frame-fraction coords).
     * A sibling canvas under the image is empty until the operator
     * turns on the global Boxes toggle. Drawing is done in CSS-px
     * coords against the rendered image rect.
     *
     * Boxes only make sense in Full mode (the crop already centres
     * on the bird) — we keep the canvas hidden in Crop mode even if
     * the toggle says "on", and redraw when the operator switches.
     */
    const BBOX_STORAGE_KEY = 'wmb_review_bbox_overlay';

    function readBboxPref() {
        return localStorage.getItem(BBOX_STORAGE_KEY) === 'on';
    }

    function writeBboxPref(on) {
        try { localStorage.setItem(BBOX_STORAGE_KEY, on ? 'on' : 'off'); } catch (e) { /* private mode */ }
    }

    function currentViewMode() {
        return localStorage.getItem('wmb_thumb_view') || 'crop';
    }

    function speciesColourFromCss(slot) {
        if (slot === '' || slot === null || slot === undefined) return '#0072B2';
        const v = getComputedStyle(document.documentElement)
            .getPropertyValue('--species-colour-' + slot)
            .trim();
        return v || '#0072B2';
    }

    function drawBboxForTile(img) {
        const canvas = img.parentElement
            && img.parentElement.querySelector('.review-grid__tile-bbox-canvas');
        if (!canvas) return;
        const x = parseFloat(img.dataset.bboxX);
        const y = parseFloat(img.dataset.bboxY);
        const w = parseFloat(img.dataset.bboxW);
        const h = parseFloat(img.dataset.bboxH);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !w || !h) {
            canvas.style.display = 'none';
            return;
        }
        const geometry = typeof window.getWmRenderedImageGeometry === 'function'
            ? window.getWmRenderedImageGeometry(img)
            : {
                elementW: img.clientWidth,
                elementH: img.clientHeight,
                contentX: 0,
                contentY: 0,
                contentW: img.clientWidth,
                contentH: img.clientHeight
            };
        const cssW = geometry.elementW;
        const cssH = geometry.elementH;
        if (cssW <= 0 || cssH <= 0) {
            canvas.style.display = 'none';
            return;
        }
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
        canvas.style.width = cssW + 'px';
        canvas.style.height = cssH + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, cssW, cssH);

        const colour = speciesColourFromCss(img.dataset.bboxSpeciesColour);
        const px = geometry.contentX + x * geometry.contentW;
        const py = geometry.contentY + y * geometry.contentH;
        const pw = w * geometry.contentW;
        const ph = h * geometry.contentH;
        // Two-pass stroke for contrast on any background: a wide
        // white halo first, then the species-colour stroke on top.
        // 3px on a green bbox over a green bird-feeder background
        // is effectively invisible — the halo trick is what every
        // photo editor uses to keep selection rects visible
        // regardless of subject colour.
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 6;
        ctx.strokeRect(px, py, pw, ph);
        ctx.strokeStyle = colour;
        ctx.lineWidth = 3;
        ctx.strokeRect(px, py, pw, ph);
        // CSS sets display:none on .review-grid__tile-bbox-canvas
        // by default. Setting style.display='' would let the CSS
        // default win — we need an explicit 'block' to override.
        canvas.style.display = 'block';
    }

    function hideBboxForTile(img) {
        const canvas = img.parentElement
            && img.parentElement.querySelector('.review-grid__tile-bbox-canvas');
        if (canvas) canvas.style.display = 'none';
    }

    function applyBboxOverlayToAllTiles() {
        // Bbox coordinates are frame-relative (0..1 of the original
        // frame). In Full mode they must be mapped to the rendered
        // bitmap inside the tile image, including object-fit cover or
        // contain offsets. In Crop mode the tile already is the bbox
        // region, so drawing the bbox on top of itself is logically
        // meaningless. The toggle is therefore gated to Full mode.
        const inFullMode = currentViewMode() === 'full';
        const on = readBboxPref() && inFullMode;
        document.querySelectorAll('.review-grid__tile-image').forEach(function (img) {
            if (!on) {
                hideBboxForTile(img);
                return;
            }
            if (img.complete && img.naturalWidth > 0) {
                drawBboxForTile(img);
            } else {
                img.addEventListener('load', function () { drawBboxForTile(img); }, { once: true });
            }
        });
        const btn = document.getElementById('reviewGridBboxToggle');
        if (btn) {
            const pref = readBboxPref();
            btn.classList.toggle('active', pref && inFullMode);
            btn.setAttribute('aria-pressed', pref ? 'true' : 'false');
            // The button stays clickable in every mode — clicking
            // it in Crop mode auto-switches to Full + enables boxes.
            // We only adjust the tooltip so the user knows what to
            // expect on the next click.
            btn.disabled = false;
            if (!inFullMode) {
                btn.title = 'Show bounding boxes (will switch to Full view since bboxes live on the full frame)';
                btn.textContent = '⬚ Boxes';
            } else {
                btn.title = 'Show or hide bounding boxes on tiles';
                btn.textContent = pref ? '⬚ Boxes ✓' : '⬚ Boxes';
            }
        }
    }

    function initBboxToggle() {
        const btn = document.getElementById('reviewGridBboxToggle');
        if (!btn) return;
        btn.addEventListener('click', function () {
            const next = !readBboxPref();
            writeBboxPref(next);
            // If the user is in Crop mode and asks for boxes,
            // auto-switch to Full mode — boxes only make sense
            // there. Reuse the existing thumb_view_toggle button
            // so all the side effects fire (img.src swap, slider
            // mode-sync, persisted pref).
            if (next && currentViewMode() !== 'full') {
                const fullToggle = document.getElementById('thumbViewToggle');
                if (fullToggle) fullToggle.click();
            }
            // applyBboxOverlayToAllTiles runs from the
            // thumb_view_toggle's own listener (we registered an
            // rAF handler in initBboxToggle below). But also
            // call it directly so disabled→enabled state and
            // the canvas update on the same click in Full mode.
            requestAnimationFrame(applyBboxOverlayToAllTiles);
        });

        // React to Crop/Full swaps: the existing thumb_view_toggle
        // swaps every <img>.src in-place. We re-run our pass on the
        // next animation frame so the canvas re-aligns with the
        // newly-loaded src (Crop -> Full may change clientWidth via
        // a different intrinsic aspect-ratio).
        const fullToggle = document.getElementById('thumbViewToggle');
        if (fullToggle) {
            fullToggle.addEventListener('click', function () {
                requestAnimationFrame(applyBboxOverlayToAllTiles);
            });
        }

        // Redraw on window resize and on slider input (tile geometry
        // changes whenever the operator drags the size slider).
        window.addEventListener('resize', function () {
            requestAnimationFrame(applyBboxOverlayToAllTiles);
        });
        const slider = document.getElementById('review-grid-size-slider');
        if (slider) {
            slider.addEventListener('input', function () {
                requestAnimationFrame(applyBboxOverlayToAllTiles);
            });
        }
    }

    function initSizeSlider() {
        const slider = document.getElementById('review-grid-size-slider');
        if (!slider) return;
        // Initial sync sets min/max/value to match the current mode.
        syncSliderToMode();
        slider.addEventListener('input', function () {
            const mode = currentViewMode();
            const px = clampSize(parseInt(slider.value, 10), modeSettings(mode));
            applyGlobalSize(px);
            writeModeSize(mode, px);
        });

        // The thumb_view_toggle button swaps localStorage.wmb_thumb_view
        // synchronously, then re-applies all data-thumb-src/full-src.
        // We catch the click on the next frame so currentViewMode()
        // returns the *new* mode, and re-sync the slider's range +
        // value to that mode's remembered position.
        const fullToggle = document.getElementById('thumbViewToggle');
        if (fullToggle) {
            fullToggle.addEventListener('click', function () {
                requestAnimationFrame(syncSliderToMode);
            });
        }
    }

    function bootstrap() {
        applySessionDefaultToAllCards();
        initSizeSlider();
        initBboxToggle();
        applyBboxOverlayToAllTiles();
    }

    document.addEventListener('DOMContentLoaded', bootstrap);
    if (document.readyState !== 'loading') {
        bootstrap();
    }
})();
