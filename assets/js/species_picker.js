/**
 * WmSpeciesPicker — Reusable species selection overlay.
 *
 * Public API:
 *   WmSpeciesPicker.pickSpecies({ currentSpecies, detectionId, mountEl, title })
 *     -> Promise<{ scientific: string, common: string } | null>
 */

window.WmSpeciesPicker = (function () {
    'use strict';

    var _speciesCache = Object.create(null);
    var EXTENDED_IDLE_LIMIT = 80;
    var FILTERED_RESULT_LIMIT = 200;

    function getCacheKey(detectionId) {
        return detectionId ? ('det:' + detectionId) : 'base';
    }

    async function loadSpeciesList(detectionId) {
        var cacheKey = getCacheKey(detectionId);
        if (_speciesCache[cacheKey]) return _speciesCache[cacheKey];

        var url = '/api/species-list';
        if (detectionId) {
            url += '?detection_id=' + encodeURIComponent(detectionId);
        }

        try {
            var resp = await fetch(url);
            if (typeof isAuthRedirect === 'function' && isAuthRedirect(resp)) {
                if (typeof redirectToLogin === 'function') redirectToLogin();
                return [];
            }
            var data = await resp.json();
            if (data.status === 'success') {
                _speciesCache[cacheKey] = data.species || [];
                return _speciesCache[cacheKey];
            }
        } catch (e) {
            console.error('[species-picker] Failed to load species list:', e);
        }

        return [];
    }

    function createSectionHeader(label) {
        var header = document.createElement('div');
        header.style.cssText =
            'padding:10px 4px 6px 4px;font-size:0.72rem;font-weight:700;' +
            'letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.48);';
        header.textContent = label;
        return header;
    }

    function createHint(text) {
        var hint = document.createElement('div');
        hint.style.cssText =
            'padding:8px 10px;border-radius:8px;background:rgba(255,255,255,0.04);' +
            'color:rgba(255,255,255,0.58);font-size:0.75rem;line-height:1.35;';
        hint.textContent = text;
        return hint;
    }

    function createSpeciesItem(sp, currentSpecies, onPick) {
        var item = document.createElement('button');
        item.type = 'button';

        var isCurrent = sp.scientific === currentSpecies;
        item.style.cssText =
            'display:flex;justify-content:space-between;align-items:center;gap:10px;' +
            'width:100%;padding:10px 12px;border:none;border-radius:8px;' +
            'cursor:pointer;text-align:left;font-size:0.82rem;' +
            'background:' + (isCurrent ? 'rgba(167,139,250,0.16)' : 'transparent') + ';' +
            'color:' + (isCurrent ? '#f5f3ff' : 'rgba(255,255,255,0.92)') + ';' +
            'transition:background 0.12s ease;';

        item.onmouseenter = function () {
            if (!isCurrent) item.style.background = 'rgba(255,255,255,0.08)';
        };
        item.onmouseleave = function () {
            if (!isCurrent) item.style.background = 'transparent';
        };

        var left = document.createElement('div');
        left.style.cssText = 'display:flex;flex-direction:column;gap:2px;min-width:0;';

        var common = document.createElement('div');
        common.style.cssText = 'font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
        common.textContent = sp.common;

        var scientific = document.createElement('div');
        scientific.style.cssText = 'font-size:0.72rem;color:rgba(255,255,255,0.52);font-style:italic;';
        scientific.textContent = sp.scientific.replace(/_/g, ' ');

        left.appendChild(common);
        left.appendChild(scientific);

        var right = document.createElement('div');
        right.style.cssText = 'display:flex;align-items:center;gap:8px;flex-shrink:0;';

        if (sp.source === 'prediction' && sp.score !== null && sp.score !== undefined) {
            var badge = document.createElement('span');
            badge.style.cssText =
                'padding:3px 8px;border-radius:999px;background:rgba(167,139,250,0.18);' +
                'color:#c4b5fd;font-size:0.72rem;font-weight:700;';
            badge.textContent = Math.round((sp.score || 0) * 100) + '%';
            right.appendChild(badge);
        } else {
            var sourceLabel = document.createElement('span');
            sourceLabel.style.cssText =
                'font-size:0.68rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.06em;';
            sourceLabel.textContent = sp.source === 'extended' ? 'Manual' : 'Model';
            right.appendChild(sourceLabel);
        }

        item.appendChild(left);
        item.appendChild(right);
        item.onclick = function () { onPick(sp); };
        return item;
    }

    function matchesFilter(sp, filter) {
        if (!filter) return true;
        var scientific = (sp.scientific || '').toLowerCase().replace(/_/g, ' ');
        var common = (sp.common || '').toLowerCase();
        return scientific.includes(filter) || common.includes(filter);
    }

    function splitSpecies(species, filter) {
        var predictions = [];
        var model = [];
        var extended = [];

        for (var i = 0; i < species.length; i++) {
            var sp = species[i];
            if (sp.source === 'prediction') {
                predictions.push(sp);
                continue;
            }
            if (!matchesFilter(sp, filter)) continue;
            if (sp.source === 'extended') {
                extended.push(sp);
            } else {
                model.push(sp);
            }
        }

        return {
            predictions: predictions,
            model: model,
            extended: extended
        };
    }

    function pickSpecies(options) {
        options = options || {};
        var currentSpecies = options.currentSpecies || '';
        var detectionId = options.detectionId || null;
        var mountEl = options.mountEl || document.body;
        var title = options.title || '🏷️ Select Species';

        return new Promise(function (resolve) {
            loadSpeciesList(detectionId).then(function (species) {
                if (!species.length) {
                    alert('Could not load species list.');
                    resolve(null);
                    return;
                }

                var existing = mountEl.querySelector('.wm-species-picker-overlay');
                if (existing) existing.remove();

                var overlay = document.createElement('div');
                overlay.className = 'wm-species-picker-overlay';
                overlay.style.cssText =
                    'position:fixed;top:0;left:0;right:0;bottom:0;' +
                    'background:rgba(0,0,0,0.74);z-index:9999;' +
                    'display:flex;align-items:center;justify-content:center;padding:14px;';

                var panel = document.createElement('div');
                panel.style.cssText =
                    'background:#171b21;border:1px solid rgba(167,139,250,0.26);' +
                    'border-radius:14px;padding:16px;width:min(480px, 100%);max-height:78vh;' +
                    'display:flex;flex-direction:column;gap:10px;' +
                    'font-family:system-ui,-apple-system,sans-serif;' +
                    'box-shadow:0 16px 48px rgba(0,0,0,0.5);';

                var header = document.createElement('div');
                header.style.cssText = 'color:#c4b5fd;font-weight:700;font-size:0.95rem;';
                header.textContent = title;
                panel.appendChild(header);

                var explainer = document.createElement('div');
                explainer.style.cssText =
                    'font-size:0.77rem;line-height:1.35;color:rgba(255,255,255,0.58);';
                explainer.textContent = 'Model predictions are shown first. Extended entries are manual labels only.';
                panel.appendChild(explainer);

                var searchInput = document.createElement('input');
                searchInput.type = 'text';
                searchInput.placeholder = 'Search species...';
                searchInput.style.cssText =
                    'width:100%;padding:10px 12px;border-radius:8px;' +
                    'border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.07);' +
                    'color:#fff;font-size:0.86rem;outline:none;box-sizing:border-box;';
                panel.appendChild(searchInput);

                var listEl = document.createElement('div');
                listEl.style.cssText =
                    'overflow-y:auto;max-height:50vh;display:flex;flex-direction:column;gap:2px;padding-right:4px;';

                function cleanup() {
                    overlay.remove();
                    document.removeEventListener('keydown', onEscape);
                }

                function onPick(sp) {
                    cleanup();
                    resolve({ scientific: sp.scientific, common: sp.common });
                }

                function renderList(filter) {
                    listEl.innerHTML = '';

                    var groups = splitSpecies(species, filter);
                    var renderedAny = false;

                    if (groups.predictions.length) {
                        listEl.appendChild(createSectionHeader('Model Predictions'));
                        for (var i = 0; i < groups.predictions.length; i++) {
                            listEl.appendChild(
                                createSpeciesItem(groups.predictions[i], currentSpecies, onPick)
                            );
                        }
                        renderedAny = true;
                    }

                    if (groups.model.length) {
                        listEl.appendChild(createSectionHeader('Model Species'));
                        for (var j = 0; j < groups.model.length; j++) {
                            listEl.appendChild(
                                createSpeciesItem(groups.model[j], currentSpecies, onPick)
                            );
                        }
                        renderedAny = true;
                    }

                    listEl.appendChild(createSectionHeader('All Bird Species'));
                    var extendedItems = groups.extended;
                    if (!filter) {
                        extendedItems = groups.extended.slice(0, EXTENDED_IDLE_LIMIT);
                        if (groups.extended.length > EXTENDED_IDLE_LIMIT) {
                            listEl.appendChild(
                                createHint('Type to search the full global catalog. Showing the first ' + EXTENDED_IDLE_LIMIT + ' extended entries.')
                            );
                        }
                    } else if (groups.extended.length > FILTERED_RESULT_LIMIT) {
                        extendedItems = groups.extended.slice(0, FILTERED_RESULT_LIMIT);
                        listEl.appendChild(
                            createHint('Too many matches. Showing the first ' + FILTERED_RESULT_LIMIT + ' results.')
                        );
                    }

                    if (extendedItems.length) {
                        for (var k = 0; k < extendedItems.length; k++) {
                            listEl.appendChild(
                                createSpeciesItem(extendedItems[k], currentSpecies, onPick)
                            );
                        }
                        renderedAny = true;
                    } else {
                        listEl.appendChild(
                            createHint(
                                filter
                                    ? 'No extended species match your search.'
                                    : 'Start typing to search the global catalog.'
                            )
                        );
                    }

                    if (!renderedAny && filter) {
                        listEl.appendChild(createHint('No species match your search.'));
                    }
                }

                renderList('');
                panel.appendChild(listEl);

                var cancelBtn = document.createElement('button');
                cancelBtn.type = 'button';
                cancelBtn.textContent = 'Cancel';
                cancelBtn.style.cssText =
                    'padding:7px 16px;border-radius:8px;border:1px solid rgba(255,255,255,0.14);' +
                    'background:transparent;color:rgba(255,255,255,0.68);cursor:pointer;' +
                    'font-size:0.8rem;align-self:flex-end;';
                cancelBtn.onclick = function () {
                    cleanup();
                    resolve(null);
                };
                panel.appendChild(cancelBtn);

                overlay.appendChild(panel);

                overlay.onclick = function (e) {
                    if (e.target === overlay) {
                        cleanup();
                        resolve(null);
                    }
                };

                function onEscape(e) {
                    if (e.key === 'Escape') {
                        cleanup();
                        resolve(null);
                    }
                }
                document.addEventListener('keydown', onEscape);

                mountEl.appendChild(overlay);

                setTimeout(function () { searchInput.focus(); }, 50);
                searchInput.oninput = function () {
                    renderList(searchInput.value.toLowerCase().trim());
                };
            });
        });
    }

    return {
        pickSpecies: pickSpecies,
        loadSpeciesList: loadSpeciesList
    };
})();
