/**
 * WmSpeciesPicker — Reusable species selection overlay.
 *
 * Extracts the search+select UI that was previously inlined in relabelDetection().
 * This module is purely a selection UI: it does NOT submit anything.
 *
 * Public API:
 *   WmSpeciesPicker.pickSpecies({ currentSpecies, mountEl, title })
 *     -> Promise<{ scientific: string, common: string } | null>
 *
 * Dependencies: None (standalone module). Loads species list from /api/species-list.
 */

window.WmSpeciesPicker = (function () {
    'use strict';

    var _speciesCache = null;

    /**
     * Load the species list from the backend (cached after first call).
     * @returns {Promise<Array<{ scientific: string, common: string }>>}
     */
    async function loadSpeciesList() {
        if (_speciesCache) return _speciesCache;
        try {
            var resp = await fetch('/api/species-list');
            if (typeof isAuthRedirect === 'function' && isAuthRedirect(resp)) {
                if (typeof redirectToLogin === 'function') redirectToLogin();
                return [];
            }
            var data = await resp.json();
            if (data.status === 'success') {
                _speciesCache = data.species; // [{ scientific, common }, ...]
                return _speciesCache;
            }
        } catch (e) {
            console.error('[species-picker] Failed to load species list:', e);
        }
        return [];
    }

    /**
     * Open a species picker overlay and return the user's choice.
     *
     * @param {Object} options
     * @param {string} [options.currentSpecies] - Highlight this species as currently selected
     * @param {Element} [options.mountEl] - DOM element to mount the overlay on (default: document.body)
     * @param {string} [options.title] - Overlay title (default: '🏷️ Select Species')
     * @returns {Promise<{ scientific: string, common: string } | null>} - null if cancelled
     */
    function pickSpecies(options) {
        options = options || {};
        var currentSpecies = options.currentSpecies || '';
        var mountEl = options.mountEl || document.body;
        var title = options.title || '🏷️ Select Species';

        return new Promise(function (resolve) {
            // Load species list first
            loadSpeciesList().then(function (species) {
                if (!species.length) {
                    alert('Could not load species list.');
                    resolve(null);
                    return;
                }

                // Remove any existing picker overlay
                var existing = mountEl.querySelector('.wm-species-picker-overlay');
                if (existing) existing.remove();

                // Create overlay
                var overlay = document.createElement('div');
                overlay.className = 'wm-species-picker-overlay';
                overlay.style.cssText =
                    'position:fixed;top:0;left:0;right:0;bottom:0;' +
                    'background:rgba(0,0,0,0.7);z-index:9999;' +
                    'display:flex;align-items:center;justify-content:center;';

                var panel = document.createElement('div');
                panel.style.cssText =
                    'background:#1a1d23;border:1px solid rgba(167,139,250,0.3);' +
                    'border-radius:12px;padding:16px;width:320px;max-height:70vh;' +
                    'display:flex;flex-direction:column;gap:10px;' +
                    'font-family:system-ui,-apple-system,sans-serif;' +
                    'box-shadow:0 8px 32px rgba(0,0,0,0.5);';

                // Header
                var header = document.createElement('div');
                header.style.cssText = 'color:#a78bfa;font-weight:600;font-size:0.9rem;';
                header.textContent = title;
                panel.appendChild(header);

                // Search input
                var searchInput = document.createElement('input');
                searchInput.type = 'text';
                searchInput.placeholder = 'Search species...';
                searchInput.style.cssText =
                    'width:100%;padding:8px 12px;border-radius:6px;' +
                    'border:1px solid rgba(255,255,255,0.15);background:rgba(255,255,255,0.08);' +
                    'color:#fff;font-size:0.85rem;outline:none;box-sizing:border-box;';
                panel.appendChild(searchInput);

                // Species list container
                var listEl = document.createElement('div');
                listEl.style.cssText =
                    'overflow-y:auto;max-height:45vh;display:flex;flex-direction:column;gap:2px;';

                function renderList(filter) {
                    listEl.innerHTML = '';
                    var filtered = filter
                        ? species.filter(function (s) {
                            return s.common.toLowerCase().includes(filter) ||
                                s.scientific.toLowerCase().includes(filter);
                        })
                        : species;

                    for (var i = 0; i < filtered.length; i++) {
                        (function (sp) {
                            var item = document.createElement('button');
                            item.type = 'button';
                            var isCurrent = sp.scientific === currentSpecies;
                            item.style.cssText =
                                'display:flex;justify-content:space-between;align-items:center;' +
                                'width:100%;padding:8px 10px;border:none;border-radius:6px;' +
                                'cursor:pointer;text-align:left;font-size:0.8rem;' +
                                'background:' + (isCurrent ? 'rgba(167,139,250,0.15)' : 'transparent') + ';' +
                                'color:' + (isCurrent ? '#a78bfa' : 'rgba(255,255,255,0.85)') + ';' +
                                'transition:background 0.1s;';
                            item.onmouseenter = function () {
                                if (!isCurrent) item.style.background = 'rgba(255,255,255,0.08)';
                            };
                            item.onmouseleave = function () {
                                if (!isCurrent) item.style.background = 'transparent';
                            };
                            item.innerHTML =
                                '<span><strong style="font-style:italic;">' +
                                sp.scientific.replace(/_/g, ' ') +
                                '</strong></span>' +
                                '<span style="font-size:0.7rem;color:rgba(255,255,255,0.4);">' +
                                sp.common + '</span>';
                            item.onclick = function () {
                                cleanup();
                                resolve({ scientific: sp.scientific, common: sp.common });
                            };
                            listEl.appendChild(item);
                        })(filtered[i]);
                    }
                }

                renderList('');
                panel.appendChild(listEl);

                // Cancel button
                var cancelBtn = document.createElement('button');
                cancelBtn.type = 'button';
                cancelBtn.textContent = 'Cancel';
                cancelBtn.style.cssText =
                    'padding:6px 16px;border-radius:6px;border:1px solid rgba(255,255,255,0.15);' +
                    'background:transparent;color:rgba(255,255,255,0.6);cursor:pointer;' +
                    'font-size:0.8rem;align-self:flex-end;';
                cancelBtn.onclick = function () {
                    cleanup();
                    resolve(null);
                };
                panel.appendChild(cancelBtn);

                overlay.appendChild(panel);

                // Click outside to cancel
                overlay.onclick = function (e) {
                    if (e.target === overlay) {
                        cleanup();
                        resolve(null);
                    }
                };

                // Escape key to cancel
                function onEscape(e) {
                    if (e.key === 'Escape') {
                        cleanup();
                        resolve(null);
                    }
                }
                document.addEventListener('keydown', onEscape);

                function cleanup() {
                    overlay.remove();
                    document.removeEventListener('keydown', onEscape);
                }

                mountEl.appendChild(overlay);

                // Focus search
                setTimeout(function () { searchInput.focus(); }, 50);
                searchInput.oninput = function () {
                    renderList(searchInput.value.toLowerCase());
                };
            });
        });
    }

    // Public API
    return {
        pickSpecies: pickSpecies,
        loadSpeciesList: loadSpeciesList
    };
})();
