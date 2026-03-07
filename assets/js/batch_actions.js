/**
 * Batch Action Engine (Phase D)
 *
 * Shared client module for batch selection + action execution.
 * Used by both edit.html and inline_edit.html.
 *
 * Provides:
 * - getExplicitSelection(container): returns { mode, ids }
 * - buildFilterContext(surface, params): returns { mode, filter_context }
 * - previewAndConfirm(actionLabel, selectionPayload): resolves selection, shows confirm
 * - executeBatchAction(endpoint, payload): executes the batch action via moderation API
 * - runBatchRelabel(selectionPayload, options): species picker → preview → relabel → reload
 *
 * Dependencies: WmSpeciesPicker (for runBatchRelabel only)
 */

window.WmBatchActions = (function () {
    'use strict';

    /**
     * Collect explicitly checked detection IDs from a container.
     * @param {string} checkboxSelector - CSS selector for checkboxes
     * @returns {{ mode: 'explicit', ids: number[] }}
     */
    function getExplicitSelection(checkboxSelector) {
        var checked = document.querySelectorAll(checkboxSelector + ':checked');
        var ids = Array.from(checked).map(function (cb) {
            // Prefer data-detection-id; fall back to value only if numeric
            var raw = cb.dataset.detectionId || cb.value;
            return parseInt(raw, 10);
        }).filter(function (id) { return !isNaN(id); });

        return { mode: 'explicit', ids: ids };
    }

    /**
     * Build a filter_context payload for all_filtered mode.
     * @param {string} surface - 'edit' | 'gallery' | 'species_overview'
     * @param {Object} params - Surface-specific filter parameters
     * @returns {{ mode: 'all_filtered', filter_context: Object }}
     */
    function buildFilterContext(surface, params) {
        var ctx = { surface: surface };

        // Copy known filter keys
        var keys = [
            'date', 'species_key', 'status_filter',
            'min_conf', 'sort', 'min_score'
        ];
        keys.forEach(function (k) {
            if (params[k] !== undefined && params[k] !== null) {
                ctx[k] = params[k];
            }
        });

        return { mode: 'all_filtered', filter_context: ctx };
    }

    /**
     * Call /api/moderation/resolve-selection to get a preview count.
     * @param {Object} selectionPayload - { mode, ids?, filenames?, filter_context? }
     * @returns {Promise<{ detection_ids: number[], image_filenames: string[], total_count: number }>}
     */
    async function resolveSelection(selectionPayload) {
        var resp = await fetch('/api/moderation/resolve-selection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(selectionPayload)
        });

        if (typeof isAuthRedirect === 'function' && isAuthRedirect(resp)) {
            if (typeof redirectToLogin === 'function') redirectToLogin();
            throw new Error('Auth redirect');
        }

        if (!resp.ok) {
            var errData = await resp.json().catch(function () { return {}; });
            throw new Error(errData.message || 'Resolve failed (' + resp.status + ')');
        }

        return await resp.json();
    }

    /**
     * Preview the selection and ask for confirmation.
     * @param {string} actionLabel - e.g. 'Move to Trash', 'Relabel to Blaumeise'
     * @param {Object} selectionPayload - from getExplicitSelection or buildFilterContext
     * @returns {Promise<{ confirmed: boolean, resolved: Object }>}
     */
    async function previewAndConfirm(actionLabel, selectionPayload) {
        // For explicit mode with known count, skip server round-trip
        if (selectionPayload.mode === 'explicit') {
            var count = (selectionPayload.ids || []).length +
                (selectionPayload.filenames || []).length;
            if (count === 0) {
                alert('No items selected.');
                return { confirmed: false, resolved: null };
            }
            var confirmed = confirm(
                actionLabel + '\n\n' +
                'This will affect ' + count + ' item(s).\n\n' +
                'Continue?'
            );
            return {
                confirmed: confirmed,
                resolved: {
                    detection_ids: selectionPayload.ids || [],
                    image_filenames: selectionPayload.filenames || [],
                    total_count: count
                }
            };
        }

        // For all_filtered: must resolve server-side first
        try {
            var resolved = await resolveSelection(selectionPayload);
            if (resolved.total_count === 0) {
                alert('No items match the current filters.');
                return { confirmed: false, resolved: resolved };
            }

            var msg = actionLabel + '\n\n' +
                'This will affect ' + resolved.total_count + ' item(s) ' +
                'matching the current filters.\n\n' +
                'Continue?';
            return {
                confirmed: confirm(msg),
                resolved: resolved
            };
        } catch (err) {
            console.error('[batch-actions] Resolve error:', err);
            alert('Could not resolve selection: ' + err.message);
            return { confirmed: false, resolved: null };
        }
    }

    /**
     * Execute a batch action against the moderation API.
     * @param {string} endpoint - e.g. '/api/moderation/bulk/reject'
     * @param {Object} payload - { detection_ids, species?, ... }
     * @returns {Promise<Object>} - API response data
     */
    async function executeBatchAction(endpoint, payload) {
        var resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (typeof isAuthRedirect === 'function' && isAuthRedirect(resp)) {
            if (typeof redirectToLogin === 'function') redirectToLogin();
            throw new Error('Auth redirect');
        }

        var data = await resp.json().catch(function () { return {}; });
        if (!resp.ok) {
            throw new Error(data.message || data.error || 'Action failed (' + resp.status + ')');
        }

        return data;
    }

    /**
     * Convenience: preview + confirm + execute in one call.
     * @param {string} actionLabel - Human-readable action name
     * @param {Object} selectionPayload - from getExplicitSelection or buildFilterContext
     * @param {string} endpoint - Moderation API endpoint
     * @param {Object} [extraPayload] - Additional payload fields (e.g. { new_species: '...' })
     * @returns {Promise<{ success: boolean, data?: Object, cancelled?: boolean }>}
     */
    async function runBatchAction(actionLabel, selectionPayload, endpoint, extraPayload) {
        var result = await previewAndConfirm(actionLabel, selectionPayload);
        if (!result.confirmed) {
            return { success: false, cancelled: true };
        }

        var payload = Object.assign(
            { detection_ids: result.resolved.detection_ids },
            extraPayload || {}
        );

        try {
            var data = await executeBatchAction(endpoint, payload);
            return { success: true, data: data };
        } catch (err) {
            console.error('[batch-actions] Execute error:', err);
            alert('Action failed: ' + err.message);
            return { success: false, cancelled: false };
        }
    }

    /**
     * Batch relabel flow: species picker → preview/confirm → execute → toast + reload.
     *
     * @param {Object} selectionPayload - from getExplicitSelection or buildFilterContext
     * @param {Object} [options]
     * @param {string} [options.pickerTitle] - Title for the species picker
     * @param {Element} [options.mountEl] - Mount element for the picker overlay
     * @returns {Promise<{ success: boolean, cancelled?: boolean, data?: Object }>}
     */
    async function runBatchRelabel(selectionPayload, options) {
        options = options || {};

        // Guard: need WmSpeciesPicker
        if (typeof WmSpeciesPicker === 'undefined') {
            alert('Species picker not available.');
            return { success: false, cancelled: true };
        }

        // Step 1: Open species picker
        var choice = await WmSpeciesPicker.pickSpecies({
            title: options.pickerTitle || '🏷️ Batch Relabel',
            mountEl: options.mountEl || document.body
        });

        if (!choice) {
            return { success: false, cancelled: true };
        }

        var speciesDisplay = choice.scientific.replace(/_/g, ' ');

        // Step 2: Preview/confirm with species name in the label
        var actionLabel = 'Relabel to ' + speciesDisplay;
        var preview = await previewAndConfirm(actionLabel, selectionPayload);
        if (!preview.confirmed) {
            return { success: false, cancelled: true };
        }

        // Step 3: Execute via moderation API
        var payload = {
            detection_ids: preview.resolved.detection_ids,
            species: choice.scientific
        };

        try {
            var data = await executeBatchAction('/api/moderation/bulk/relabel', payload);

            // Step 4: Toast + Reload
            if (typeof wmToast === 'function') {
                wmToast(
                    (data.relabeled || 0) + ' items relabeled to ' + speciesDisplay,
                    'success', 3000
                );
            }
            setTimeout(function () { location.reload(); }, 1500);

            return { success: true, data: data };
        } catch (err) {
            console.error('[batch-actions] Relabel error:', err);
            alert('Relabel failed: ' + err.message);
            return { success: false, cancelled: false };
        }
    }

    // Public API
    return {
        getExplicitSelection: getExplicitSelection,
        buildFilterContext: buildFilterContext,
        resolveSelection: resolveSelection,
        previewAndConfirm: previewAndConfirm,
        executeBatchAction: executeBatchAction,
        runBatchAction: runBatchAction,
        runBatchRelabel: runBatchRelabel
    };
})();
