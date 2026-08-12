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

    // safeSameOriginPath is provided globally by gallery_utils.js (loaded first
    // on every page that loads this file).

    /* =========================================
       No-Bird Frame — surface-agnostic fallback
       Used on Gallery / Subgallery / Species / Detection-Modal surfaces
       where review_workspace.js (and its singleAction) is not loaded.
       ========================================= */

    /**
     * Show a confirmation modal that displays the FULL frame (not the
     * cropped tile) so the operator can see every bird the action will
     * sweep away. The cropped tile view is the UX trap: the operator
     * sees a tight bird crop and forgets that the underlying frame may
     * carry several more detections.
     *
     * Returns a Promise that resolves with `true` (confirmed) or
     * `false` (cancelled). The confirmation is deliberately never
     * remembered: every whole-image assertion must show its full frame.
     */
    function confirmFrameNoBird(filename, triggerEl, fullSrcOverride, progressLabel) {

        // Resolve the full-image URL from the nearest tile's <img>.
        // data-full-src is set by every wm-tile in subgallery / species /
        // species_overview templates and points at the optimized full
        // frame (not the crop).
        const localRoot = triggerEl && triggerEl.closest(
            '.wm-tile, .review-grid__tile, .wm-modal, .review-stage-panel, .wm-toolbox-host'
        );
        const localImg = localRoot && localRoot.querySelector(
            'img[data-full-src], img[data-deferred-src], .wm-image-viewer__img, .wm-tile__image, .review-grid__tile-image'
        );
        let fullSrc = fullSrcOverride
            || (localImg && localImg.getAttribute('data-full-src'))
            || (localImg && localImg.getAttribute('data-deferred-src'))
            || (localImg && localImg.getAttribute('src'))
            || '';

        if (!fullSrc && filename) {
            const stem = filename.replace(/\.(jpg|jpeg|png|webp)$/i, '');
            const images = document.querySelectorAll('img[data-full-src], img[data-deferred-src]');
            for (let index = 0; index < images.length; index += 1) {
                const candidate = images[index].getAttribute('data-full-src')
                    || images[index].getAttribute('data-deferred-src')
                    || '';
                if (candidate.indexOf(stem) !== -1) {
                    fullSrc = candidate;
                    break;
                }
            }
        }
        const safeFullSrc = fullSrc ? safeSameOriginPath(fullSrc) : '';
        if (!safeFullSrc) {
            if (window.wmToast) {
                window.wmToast('Full image preview is unavailable. Nothing was changed.', 'error', 4000);
            }
            return Promise.resolve(false);
        }

        // Count siblings on the same frame so the warning can be honest:
        // "you are about to flag N detections, not 1". We approximate
        // via tiles whose data-full-src ends with the same filename stem.
        let detectionCount = 1;
        if (filename) {
            const stem = filename.replace(/\.(jpg|jpeg|png|webp)$/i, '');
            const allTiles = document.querySelectorAll('.wm-tile__image[data-full-src]');
            let n = 0;
            for (let i = 0; i < allTiles.length; i++) {
                const src = allTiles[i].getAttribute('data-full-src') || '';
                if (src.indexOf(stem) !== -1) n++;
            }
            if (n > 0) detectionCount = n;
        }

        return new Promise(function (resolve) {
            const dlg = document.createElement('dialog');
            dlg.className = 'wm-no-bird-confirm';
            dlg.style.cssText = [
                'padding: 0',
                'border: 1px solid var(--color-border, #444)',
                'border-radius: 8px',
                'max-width: min(720px, 95vw)',
                'max-height: 90vh',
                'background: var(--color-surface, #1a1a1a)',
                'color: var(--color-text, #eee)',
            ].join('; ');

            const body = document.createElement('div');
            body.style.cssText = 'display: flex; flex-direction: column; gap: 12px; padding: 16px;';

            const heading = document.createElement('h2');
            heading.textContent = 'No birds in this full image?'
                + (progressLabel ? ' · ' + progressLabel : '');
            heading.style.cssText = 'margin: 0; font-size: 18px; font-weight: 600;';
            body.appendChild(heading);

            const subline = document.createElement('p');
            subline.style.cssText = 'margin: 0; color: var(--color-text-muted, #aaa); font-size: 14px;';
            subline.textContent = detectionCount > 1
                ? ('This frame carries ' + detectionCount
                    + ' detections — ALL will be flagged as false-positives.')
                : 'All detections on this frame will be flagged as false-positives.';
            body.appendChild(subline);

            // Full-frame preview — the whole point of this dialog. If the
            // image is not yet cached, the operator sees a placeholder
            // background until the optimized version downloads.
            // Sanitise fullSrc through the same same-origin path validator
            // used for navigation. Co-located at the sink so the guard is
            // visible next to the .src assignment.
            const preview = document.createElement('img');
            preview.src = safeFullSrc;
            preview.alt = 'Full image: ' + filename;
            preview.style.cssText = [
                'width: 100%',
                'height: auto',
                'max-height: 60vh',
                'object-fit: contain',
                'background: #000',
                'border-radius: 4px',
                'display: block',
            ].join('; ');
            body.appendChild(preview);

            const hint = document.createElement('p');
            hint.style.cssText = 'margin: 0; font-size: 12px; color: var(--color-text-muted, #888);';
            hint.textContent = 'This records that the entire image contains no bird. '
                + 'Review the complete frame above, not only the crop.';
            body.appendChild(hint);

            const actions = document.createElement('div');
            actions.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end; margin-top: 4px;';

            const cancelBtn = document.createElement('button');
            cancelBtn.type = 'button';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.className = 'btn btn--secondary';
            cancelBtn.style.cssText = 'padding: 8px 16px; cursor: pointer;';
            cancelBtn.addEventListener('click', function () {
                dlg.close('cancel');
            });
            actions.appendChild(cancelBtn);

            const confirmBtn = document.createElement('button');
            confirmBtn.type = 'button';
            confirmBtn.textContent = 'Confirm no birds in full image';
            confirmBtn.className = 'btn btn--danger';
            confirmBtn.style.cssText = 'padding: 8px 16px; cursor: pointer; '
                + 'background: var(--color-danger, #dc2626); color: white; border: none; border-radius: 4px;';
            confirmBtn.addEventListener('click', function () {
                dlg.close('confirm');
            });
            actions.appendChild(confirmBtn);

            body.appendChild(actions);
            dlg.appendChild(body);
            document.body.appendChild(dlg);

            dlg.addEventListener('close', function () {
                const ok = dlg.returnValue === 'confirm';
                dlg.remove();
                resolve(ok);
            });

            // Backdrop click cancels — the operator clicking outside the
            // dialog means "I changed my mind".
            dlg.addEventListener('click', function (event) {
                if (event.target === dlg) dlg.close('cancel');
            });

            if (typeof dlg.showModal === 'function') {
                dlg.showModal();
            } else {
                // A text-only fallback would recreate the crop-context trap.
                // Fail closed when the browser cannot show the full image.
                dlg.remove();
                resolve(false);
            }
        });
    }

    async function confirmFullImageNoBird(frames, triggerEl) {
        const requested = Array.isArray(frames) ? frames : [frames];
        const targets = [];
        const seen = new Set();
        requested.forEach(function (frame) {
            const item = typeof frame === 'string' ? { filename: frame } : (frame || {});
            const filename = String(item.filename || '').trim();
            if (!filename || seen.has(filename)) return;
            seen.add(filename);
            targets.push(item);
        });

        for (let index = 0; index < targets.length; index += 1) {
            const target = targets[index];
            const progressLabel = targets.length > 1
                ? (String(index + 1) + ' of ' + String(targets.length))
                : '';
            const confirmed = await confirmFrameNoBird(
                target.filename,
                target.triggerEl || triggerEl,
                target.fullSrc || '',
                progressLabel
            );
            if (!confirmed) return false;
        }
        return targets.length > 0;
    }

    window.wmConfirmFullImageNoBird = confirmFullImageNoBird;

    async function noBirdFrame(filename, triggerEl) {
        const confirmed = await confirmFullImageNoBird([{ filename: filename }], triggerEl);
        if (!confirmed) return;

        try {
            const response = await fetch('/api/review/decision', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: [filename], action: 'no_bird' })
            });
            const data = await response.json().catch(function () { return {}; });
            if (!response.ok || data.status !== 'success') {
                throw new Error(data.message || ('HTTP ' + response.status));
            }

            // Fade out the nearest tile in-place
            const tile = triggerEl && triggerEl.closest('.wm-tile');
            if (tile) {
                tile.style.transition = 'opacity 0.3s, transform 0.3s';
                tile.style.opacity = '0';
                tile.style.transform = 'scale(0.95)';
                setTimeout(function () { tile.remove(); }, 320);
            }

            if (window.wmToast) {
                window.wmToast('Full image recorded as containing no birds', 'success', 2800);
            }
        } catch (error) {
            console.error('[tile-actions] no-bird error:', error);
            if (window.wmToast) {
                window.wmToast('Failed to mark frame: ' + (error.message || String(error)), 'error', 5000);
            } else {
                alert('Failed to mark frame: ' + (error.message || String(error)));
            }
        }
    }

    function correctionFactValue(fact) {
        if (fact.fact_type === 'bbox_correction') {
            return [fact.bbox_x, fact.bbox_y, fact.bbox_w, fact.bbox_h]
                .map(function (value) { return Number(value).toFixed(4); })
                .join(', ');
        }
        if (fact.fact_type === 'species_identity' && fact.species_key) {
            return String(fact.answer_value) + ' · ' + String(fact.species_key);
        }
        return String(fact.answer_value || 'unknown');
    }

    function readinessCopy(readiness) {
        const labels = {
            object_bird_presence_unknown: 'bird presence unanswered',
            object_bird_absent: 'object marked as not a bird',
            bbox_quality_unknown: 'box quality unanswered',
            bbox_unsuitable: 'box marked unsuitable',
            species_identity_unknown: 'species unanswered',
            species_unknown: 'species marked unknown',
            species_wrong: 'species marked wrong',
            species_identity_unsupported: 'species answer unsupported',
            species_key_missing: 'species name missing'
        };
        if (!readiness || readiness.ready) return 'Ready';
        return (readiness.reasons || []).map(function (reason) {
            return labels[reason] || reason;
        }).join(' · ');
    }

    async function showCorrectionDetails(filename, detectionId) {
        const params = new URLSearchParams({
            filename: filename,
            detection_id: String(detectionId)
        });
        const response = await fetch('/api/labels/state?' + params.toString(), {
            credentials: 'same-origin'
        });
        const data = await response.json().catch(function () { return {}; });
        if (!response.ok || data.status !== 'success') {
            throw new Error(data.message || ('HTTP ' + response.status));
        }

        const dialog = document.createElement('dialog');
        dialog.className = 'wm-correction-details';
        const panel = document.createElement('div');
        panel.className = 'wm-correction-details__panel';

        const heading = document.createElement('h2');
        heading.textContent = 'Correction details — this box only';
        panel.appendChild(heading);

        const progress = data.object_progress || {};
        const scope = document.createElement('p');
        scope.className = 'wm-correction-details__scope';
        scope.textContent = Number(progress.total || 0) > 1
            ? (String(progress.answered || 0) + ' of ' + String(progress.total)
                + ' boxes on this image have recorded answers. Other boxes are unchanged.')
            : 'Answers here apply only to the offered box. Full-image answers are listed separately.';
        panel.appendChild(scope);

        const readiness = document.createElement('div');
        readiness.className = 'wm-correction-details__readiness';
        ['od', 'cls'].forEach(function (kind) {
            const item = document.createElement('div');
            const status = data.readiness && data.readiness[kind];
            item.className = 'wm-correction-details__readiness-item'
                + (status && status.ready ? ' is-ready' : '');
            const title = document.createElement('strong');
            title.textContent = kind.toUpperCase() + ': ';
            item.appendChild(title);
            item.appendChild(document.createTextNode(readinessCopy(status)));
            readiness.appendChild(item);
        });
        panel.appendChild(readiness);

        const facts = Array.isArray(data.facts) ? data.facts : [];
        const list = document.createElement('dl');
        list.className = 'wm-correction-details__facts';
        facts.forEach(function (fact) {
            const term = document.createElement('dt');
            term.textContent = String(fact.scope) + ' · ' + String(fact.fact_type);
            const value = document.createElement('dd');
            value.textContent = correctionFactValue(fact);
            list.appendChild(term);
            list.appendChild(value);
        });
        if (facts.length === 0) {
            const empty = document.createElement('p');
            empty.className = 'wm-correction-details__empty';
            empty.textContent = 'No explicit answers recorded yet.';
            panel.appendChild(empty);
        } else {
            panel.appendChild(list);
        }

        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'btn btn--secondary wm-correction-details__close';
        close.textContent = 'Close';
        close.addEventListener('click', function () { dialog.close(); });
        panel.appendChild(close);
        dialog.appendChild(panel);
        dialog.addEventListener('close', function () { dialog.remove(); });
        dialog.addEventListener('click', function (event) {
            if (event.target === dialog) dialog.close();
        });
        document.body.appendChild(dialog);
        dialog.showModal();
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
                    // Re-assert the same-origin / safe-path shape at the
                    // sink — the helper above already enforces it, but
                    // keeping the check local makes the invariant obvious.
                    if (safeDetailsPath
                        && safeDetailsPath.charAt(0) === '/'
                        && /^[A-Za-z0-9_\-./?&=#]+$/.test(safeDetailsPath)) {
                        window.location.assign(safeDetailsPath);
                    }
                }
                break;

            case 'correction-details':
                showCorrectionDetails(filename, Number(detectionId)).catch(function (error) {
                    console.error('[tile-actions] correction details error:', error);
                    if (window.wmToast) {
                        window.wmToast('Could not load correction details.', 'error', 3500);
                    }
                });
                break;

            case 'favorite':
                if (typeof toggleFavorite === 'function' && detectionId) {
                    toggleFavorite(null, detectionId, actionEl);
                }
                break;

            case 'relabel':
            case 'change-species':
                if (typeof relabelDetection === 'function' && detectionId) {
                    var currentSpecies = actionEl.getAttribute('data-current-species') || '';
                    relabelDetection(null, parseInt(detectionId, 10), currentSpecies);
                }
                break;

            case 'correct-bbox':
                if (typeof window.startWmBboxEditor === 'function') {
                    window.startWmBboxEditor(actionEl);
                } else if (window.wmToast) {
                    window.wmToast('Box editor is unavailable.', 'error', 3200);
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
                if (filename) {
                    // Every surface uses the same full-image safety gate.
                    // No legacy review callback may bypass the preview.
                    noBirdFrame(filename, actionEl);
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
