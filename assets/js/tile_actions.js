/**
 * Tile Action Engine
 *
 * Central event-delegation engine for all wm-toolbox actions.
 * Uses data-action attributes to route clicks to the correct handler.
 *
 * This replaces per-surface inline JS for tile-level actions (favorite,
 * relabel, delete etc.) with a single document-level listener.
 *
 * Dependencies: gallery_utils.js (for toggleFavorite, relabelDetection,
 * deleteDetection, isAuthRedirect, redirectToLogin)
 */

(function () {
    'use strict';

    /* =========================================
       Toggle Menu (⋮ More Actions)
       ========================================= */

    function closeAllMenus(except) {
        document.querySelectorAll('.wm-toolbox__dropdown--open').forEach(function (dd) {
            if (dd !== except) dd.classList.remove('wm-toolbox__dropdown--open');
        });
        // Update aria-expanded on toggle buttons
        document.querySelectorAll('.wm-toolbox__more[aria-expanded="true"]').forEach(function (btn) {
            const dropdown = btn.nextElementSibling;
            if (dropdown !== except) btn.setAttribute('aria-expanded', 'false');
        });
    }

    function toggleMenu(btn) {
        const dropdown = btn.nextElementSibling;
        if (!dropdown) return;

        const isOpen = dropdown.classList.contains('wm-toolbox__dropdown--open');
        closeAllMenus(isOpen ? null : dropdown);

        if (isOpen) {
            dropdown.classList.remove('wm-toolbox__dropdown--open');
            btn.setAttribute('aria-expanded', 'false');
        } else {
            dropdown.classList.add('wm-toolbox__dropdown--open');
            btn.setAttribute('aria-expanded', 'true');
            // Focus first menu item for keyboard users
            const firstItem = dropdown.querySelector('[role="menuitem"]');
            if (firstItem) firstItem.focus();
        }
    }

    function setTrainingExportState(btn) {
        if (!btn || !btn.classList) return;
        btn.classList.add('wm-toolbox__training--active');
        btn.classList.add('wm-toolbox__btn--toggled');
        btn.setAttribute('aria-pressed', 'true');
        btn.setAttribute('title', 'Already in training export pool');
    }

    function safeSameOriginPath(rawUrl) {
        try {
            const parsed = new URL(rawUrl, window.location.origin);
            if (parsed.origin !== window.location.origin) return '';
            if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return '';
            return parsed.pathname + parsed.search + parsed.hash;
        } catch (e) {
            return '';
        }
    }

    async function addDetectionToTrainingExport(detectionId, btn) {
        const currentSpecies = btn ? (btn.getAttribute('data-current-species') || '').trim() : '';
        try {
            const response = await fetch('/api/training-export/add', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    detection_ids: [Number(detectionId)],
                    confirm_current_species: Boolean(currentSpecies),
                    current_species: currentSpecies
                })
            });
            if (typeof isAuthRedirect === 'function' && isAuthRedirect(response)) {
                if (typeof redirectToLogin === 'function') redirectToLogin();
                return;
            }

            const data = await response.json().catch(function () { return {}; });
            if (!response.ok) {
                throw new Error(data.message || ('HTTP ' + response.status));
            }

            const queued = Number(data.added || 0) > 0 || Number(data.already_in_pool || 0) > 0 || Number(data.eligible || 0) > 0;
            if (queued) {
                document.querySelectorAll(`.wm-toolbox__training[data-detection-id="${detectionId}"]`).forEach(function (toolboxBtn) {
                    setTrainingExportState(toolboxBtn);
                });
            }

            if (window.wmToast) {
                if (Number(data.added || 0) > 0) {
                    window.wmToast('Added to training export pool', 'success', 2600);
                } else if (Number(data.already_in_pool || 0) > 0) {
                    window.wmToast('Already in training export pool', 'info', 2600);
                } else {
                    window.wmToast('Not eligible yet: confirm species and all boxes first', 'info', 4200);
                }
            }
        } catch (error) {
            console.error('Training export add error:', error);
            if (window.wmToast) {
                window.wmToast('Training export failed: ' + (error.message || String(error)), 'error', 5000);
            }
        }
    }

    /* =========================================
       Action Dispatcher
       ========================================= */

    function handleAction(actionEl) {
        var action = actionEl.getAttribute('data-action');
        if (!action) return;

        var detectionId = actionEl.getAttribute('data-detection-id');
        var filename = actionEl.getAttribute('data-filename');
        var loginRequired = actionEl.getAttribute('data-login-required') === 'true';

        if (loginRequired) {
            closeAllMenus(null);
            if (window.wmToast) {
                window.wmToast('Please log in to use this action.', 'info', 2200);
            }
            if (typeof redirectToLogin === 'function') {
                redirectToLogin();
            }
            return;
        }

        // Close dropdown after action
        closeAllMenus(null);

        switch (action) {
            case 'toggle-menu':
                toggleMenu(actionEl);
                return; // Don't close menu, we just opened it

            case 'details':
            case 'view-details':
                var modalTarget = actionEl.getAttribute('data-modal-target');
                var detailsHref = actionEl.getAttribute('data-details-href');
                if (modalTarget && typeof bootstrap !== 'undefined') {
                    var modalEl = document.querySelector(modalTarget);
                    if (modalEl) {
                        var bsModal = new bootstrap.Modal(modalEl);
                        bsModal.show();
                        return;
                    }
                }
                if (detailsHref) {
                    const safeDetailsPath = safeSameOriginPath(detailsHref);
                    if (safeDetailsPath) window.location.assign(safeDetailsPath);
                }
                break;

            case 'favorite':
                if (typeof toggleFavorite === 'function' && detectionId) {
                    toggleFavorite(null, detectionId, actionEl);
                }
                break;

            case 'training-export':
                if (detectionId) {
                    addDetectionToTrainingExport(detectionId, actionEl);
                }
                break;

            case 'relabel':
            case 'change-species':
                if (typeof relabelDetection === 'function' && detectionId) {
                    var currentSpecies = actionEl.getAttribute('data-current-species') || '';
                    relabelDetection(null, parseInt(detectionId, 10), currentSpecies);
                }
                break;

            case 'delete':
            case 'move-trash':
                if (typeof deleteDetection === 'function' && detectionId) {
                    deleteDetection(null, parseInt(detectionId, 10));
                }
                break;

            case 'deep-scan':
                if (typeof analyzeAction === 'function' && filename) {
                    analyzeAction(null, filename);
                } else if (filename) {
                    // Fallback: direct API call
                    fetch('/api/review/analyze/' + encodeURIComponent(filename), { method: 'POST' })
                        .then(function (r) { return r.json(); })
                        .then(function (data) {
                            if (data.status === 'success') {
                                if (window.wmToast) window.wmToast('Deep Scan queued', 'success', 2000);
                            } else {
                                if (window.wmToast) window.wmToast('Error: ' + data.message, 'error', 4000);
                            }
                        })
                        .catch(function (e) {
                            console.error('Deep scan error:', e);
                        });
                }
                break;

            case 'review-confirm':
                if (typeof singleAction === 'function' && filename) {
                    singleAction(filename, 'confirm');
                }
                break;

            case 'review-no-bird':
                if (typeof singleAction === 'function' && filename) {
                    singleAction(filename, 'no_bird');
                }
                break;

            case 'restore':
                if (detectionId || filename) {
                    var trashId = detectionId || filename;
                    var trashType = detectionId ? 'detection' : 'image';
                    // Create fake checkbox for performAction compatibility
                    if (typeof performAction === 'function') {
                        var fakeCheckbox = {
                            value: trashId,
                            getAttribute: function (attr) { return attr === 'data-type' ? trashType : trashId; }
                        };
                        performAction('restore', [fakeCheckbox]);
                    }
                }
                break;

            default:
                console.warn('[tile-actions] Unknown action:', action);
        }
    }

    /* =========================================
       Event Delegation (single listener)
       ========================================= */

    document.addEventListener('click', function (event) {
        var actionEl = event.target.closest('[data-action]');
        var actionSurface = actionEl && (
            actionEl.closest('.wm-toolbox') ||
            actionEl.closest('.modal-action-bar')
        );
        if (actionEl && actionSurface) {
            event.preventDefault();
            event.stopPropagation();
            handleAction(actionEl);
            return;
        }

        // Close open menus when clicking outside
        if (!event.target.closest('.wm-toolbox__menu')) {
            closeAllMenus(null);
        }
    }, true); // Capture phase to intercept before bootstrap modals

    /* =========================================
       Keyboard Support
       ========================================= */

    document.addEventListener('keydown', function (event) {
        // Escape closes open menus
        if (event.key === 'Escape') {
            closeAllMenus(null);
        }

        // Arrow keys navigate within dropdown
        var openDropdown = document.querySelector('.wm-toolbox__dropdown--open');
        if (!openDropdown) return;

        var items = Array.from(openDropdown.querySelectorAll('[role="menuitem"]'));
        var focusedIndex = items.indexOf(document.activeElement);

        if (event.key === 'ArrowDown') {
            event.preventDefault();
            var next = focusedIndex < items.length - 1 ? focusedIndex + 1 : 0;
            items[next].focus();
        } else if (event.key === 'ArrowUp') {
            event.preventDefault();
            var prev = focusedIndex > 0 ? focusedIndex - 1 : items.length - 1;
            items[prev].focus();
        }
    });

})();
