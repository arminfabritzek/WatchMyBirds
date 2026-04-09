/**
 * Shared Gallery Navigation Logic for WatchMyBirds
 * Handles modal navigation, keyboard shortcuts, deletion, relabeling, and favorites.
 */

/* =========================================
   Auth Redirect Detection (Session Expiry)
   =========================================
   When a session expires, fetch() follows the 302 to /login transparently
   and returns the login HTML with status 200. We detect this by checking
   resp.redirected + URL containing '/login', or content-type is text/html.
*/
function isAuthRedirect(resp) {
    if (resp.redirected && resp.url && resp.url.includes('/login')) return true;
    const ct = resp.headers.get('content-type');
    if (resp.ok && ct && ct.includes('text/html')) return true;
    return false;
}

function getViewerScope(el) {
    if (!el || !el.closest) return null;
    return el.closest('.wm-viewer-scope') || el.closest('.modal');
}

function getViewerHost(el) {
    if (!el || !el.closest) return null;
    return el.closest('.wm-toolbox-host');
}

/**
 * Read viewer preference keys from scope element's data attributes.
 * Review scopes set data-bbox-pref-key / data-zoom-pref-key explicitly;
 * gallery modals fall back to the default modal keys.
 */
function getViewerPrefKeys(scope) {
    var isReview = Boolean(scope && scope.classList && scope.classList.contains('wm-viewer-scope'));
    return {
        bbox: (scope && scope.dataset && scope.dataset.bboxPrefKey) || (isReview ? 'wmb_review_bbox_pref' : 'wmb_modal_bbox_pref'),
        bboxDefault: isReview ? 'on' : 'off',
        zoom: (scope && scope.dataset && scope.dataset.zoomPrefKey) || (isReview ? 'wmb_review_zoom_pref' : 'wmb_modal_zoom_pref'),
    };
}

function redirectToLogin() {
    if (window.wmToast) window.wmToast('Session expired. Please log in.', 'error');
    window.location.href = '/login?next=' + encodeURIComponent(
        window.location.pathname + window.location.search + window.location.hash
    );
}

/* =========================================
   Favorite Toggle (Modal Action Bar & Badges)
   ========================================= */

// Intercept all mouse events on the favorite badge early in the capture phase.
// This guarantees that Bootstrap modals or parent <a> tags are never triggered.
['mousedown', 'mouseup', 'click'].forEach(eventType => {
    document.addEventListener(eventType, function (event) {
        const favBtn = event.target.closest('.wm-tile__fav-badge');
        if (favBtn) {
            // Completely swallow the event before it bubbles or captures further down
            event.preventDefault();
            event.stopPropagation();

            if (eventType === 'click') {
                const tile = favBtn.closest('[data-detection-id]');
                if (tile) {
                    const detId = tile.getAttribute('data-detection-id');
                    if (detId) {
                        toggleFavorite(null, detId, favBtn);
                    }
                }
            }
        }
    }, true); // `true` ensures it runs in the capture phase, before any targets
});

function setToolboxFavoriteState(btn, isFav) {
    if (!btn || !btn.classList) return;
    btn.classList.toggle('wm-toolbox__fav--active', isFav);
    btn.textContent = isFav ? '⭐' : '☆';
    btn.setAttribute('aria-pressed', isFav ? 'true' : 'false');
    btn.setAttribute('aria-label', isFav ? 'Remove from favorites' : 'Add to favorites');
    btn.setAttribute('title', isFav ? 'Unfavorite' : 'Favorite');
}

function setLegacyTileBadgeState(btn, isFav) {
    if (!btn || !btn.classList) return;
    btn.classList.toggle('wm-tile__fav-badge--active', isFav);
    btn.textContent = isFav ? '⭐' : '☆';
}

function setModalFavoriteState(btn, isFav) {
    if (!btn || !btn.classList) return;
    btn.classList.toggle('fav-btn--active', isFav);
    btn.textContent = isFav ? '⭐' : '☆';
}

async function toggleFavorite(event, detectionId, btn) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    let step = "start";
    try {
        step = "fetch";
        const resp = await fetch('/api/detections/favorite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ detection_id: Number(detectionId) })
        });
        if (isAuthRedirect(resp)) {
            redirectToLogin();
            return;
        }
        if (resp.ok) {

            step = "json";
            const data = await resp.json();
            const isFav = Boolean(data.is_favorite);

            step = "btn-toggle";
            // Update the button in the modal or the hovered badge
            if (btn && btn.classList) {
                setToolboxFavoriteState(btn, isFav);
                setLegacyTileBadgeState(btn, isFav);
                setModalFavoriteState(btn, isFav);
            }

            step = "dom-queries";
            try {
                // Keep every rendered instance of the same detection in sync.
                document.querySelectorAll(`.wm-toolbox__fav[data-detection-id="${detectionId}"]`).forEach(function (toolboxBtn) {
                    if (toolboxBtn !== btn) setToolboxFavoriteState(toolboxBtn, isFav);
                });

                document.querySelectorAll(`.wm-tile[data-detection-id="${detectionId}"] .wm-tile__fav-badge`).forEach(function (tileBadge) {
                    if (tileBadge !== btn) setLegacyTileBadgeState(tileBadge, isFav);
                });

                document.querySelectorAll(`.gallery-modal[data-detection-id="${detectionId}"] .fav-btn`).forEach(function (modalBtn) {
                    if (modalBtn !== btn) setModalFavoriteState(modalBtn, isFav);
                });
            } catch (domErr) {
                console.warn('DOM update error ignored:', domErr);
            }

            step = "toast";
            // Toast feedback
            if (window.wmToast) {
                window.wmToast(isFav ? '⭐ Favorite added' : '☆ Favorite removed', isFav ? 'success' : 'info', 2000);
            }
        } else {
            step = "error-text";
            const errText = await resp.text().catch(function () { return ''; });
            console.error('Favorite API error:', resp.status, errText);
            if (window.wmToast) {
                window.wmToast('Favorite error: ' + resp.status + ' ' + errText.slice(0, 80), 'error', 5000);
            }
        }
    } catch (err) {
        console.error('Favorite toggle error:', err);
        if (window.wmToast) {
            window.wmToast('Favorite failed: ' + (err.message || String(err)), 'error', 5000);
        }
    }
}

/* =========================================
   Modal Navigation (Simple, Fast)
   ========================================= */
let modalNavigationInFlight = false;

function showModalTransition(currentModalEl, nextModalEl) {
    if (!currentModalEl || !nextModalEl || currentModalEl === nextModalEl || modalNavigationInFlight) return;

    modalNavigationInFlight = true;

    const unlockNavigation = function () {
        modalNavigationInFlight = false;
    };

    const showNextModal = function () {
        const nextInstance = bootstrap.Modal.getOrCreateInstance
            ? bootstrap.Modal.getOrCreateInstance(nextModalEl)
            : new bootstrap.Modal(nextModalEl);

        nextModalEl.addEventListener('shown.bs.modal', unlockNavigation, { once: true });
        nextInstance.show();
    };

    const currentInstance = bootstrap.Modal.getInstance(currentModalEl);
    if (currentInstance && currentModalEl.classList.contains('show')) {
        currentModalEl.addEventListener('hidden.bs.modal', showNextModal, { once: true });
        currentInstance.hide();
        return;
    }

    showNextModal();
}

function navigateModal(btn, direction) {
    const currentModalEl = btn.closest('.modal');
    if (!currentModalEl) return;

    const navScope = currentModalEl.getAttribute('data-nav-scope');
    if (navScope) {
        const scopedModals = Array.from(
            document.querySelectorAll(`.gallery-modal[data-nav-scope="${navScope}"]`)
        ).sort(function (a, b) {
            return Number(a.getAttribute('data-nav-index')) - Number(b.getAttribute('data-nav-index'));
        });

        const currentIndex = scopedModals.indexOf(currentModalEl);
        if (currentIndex === -1 || scopedModals.length <= 1) return;

        const step = direction === 'next' ? 1 : -1;
        let nextIndex = currentIndex + step;

        if (nextIndex >= scopedModals.length) nextIndex = 0;
        if (nextIndex < 0) nextIndex = scopedModals.length - 1;

        showModalTransition(currentModalEl, scopedModals[nextIndex]);
        return;
    }

    const group = currentModalEl.getAttribute('data-modal-group');
    if (!group) return;

    // Get current image path to skip siblings (multiple detections on same image)
    // For observation groups (obs*), do NOT skip — all detections must be reachable
    const isObservationGroup = group.startsWith('obs');
    const currentImagePath = isObservationGroup ? null : currentModalEl.getAttribute('data-image-path');

    // Find all modals in this group
    const allModals = Array.from(document.querySelectorAll(`.gallery-modal[data-modal-group="${group}"]`));
    const currentIndex = allModals.indexOf(currentModalEl);

    if (currentIndex === -1) return;

    // Find next modal with a DIFFERENT image path (skip siblings)
    let nextIndex = currentIndex;
    const step = direction === 'next' ? 1 : -1;
    const totalModals = allModals.length;

    // Loop through modals until we find one with a different image
    for (let i = 0; i < totalModals; i++) {
        nextIndex = nextIndex + step;
        // Wrap around
        if (nextIndex >= totalModals) nextIndex = 0;
        if (nextIndex < 0) nextIndex = totalModals - 1;

        const candidateModal = allModals[nextIndex];
        const candidateImagePath = candidateModal.getAttribute('data-image-path');

        // If different image (or no image path set), use this modal
        if (candidateImagePath !== currentImagePath || !currentImagePath) {
            break;
        }

        // Safety: if we've checked all modals without finding different image, stop
        if (nextIndex === currentIndex) return;
    }

    const nextModalEl = allModals[nextIndex];
    showModalTransition(currentModalEl, nextModalEl);
}


/* =========================================
   Deletion Logic
   ========================================= */
async function deleteDetection(event, id) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    if (!confirm('Move this detection to trash?')) return;

    try {
        const response = await fetch('/api/detections/reject', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ids: [id] })
        });

        if (isAuthRedirect(response)) {
            redirectToLogin();
            return;
        }
        if (response.ok) {
            // Check if this is a sibling card delete (inside an open modal)
            const openModal = document.querySelector('.gallery-modal.show');
            if (openModal) {
                const siblingCard = openModal.querySelector(`.sibling-card[data-detection-id="${id}"]`);
                const allSiblingCards = openModal.querySelectorAll('.sibling-card');

                if (siblingCard && allSiblingCards.length > 1) {
                    // Remove just the sibling card with a fade-out
                    siblingCard.style.transition = 'opacity 0.25s, transform 0.25s';
                    siblingCard.style.opacity = '0';
                    siblingCard.style.transform = 'scale(0.8)';
                    setTimeout(() => siblingCard.remove(), 260);

                    // Also hide the bbox overlay if it was showing this detection
                    const canvas = openModal.querySelector('.bbox-overlay');
                    if (canvas) {
                        canvas.style.display = 'none';
                    }
                    return; // Don't reload — modal stays open
                }
            }

            // Fallback: main detection deleted or last one — reload page
            location.reload();
        } else {
            const data = await response.json();
            alert('Failed to delete: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while deleting.');
    }
}

/* =========================================
   Relabel Logic — uses WmSpeciesPicker for UI
   ========================================= */

async function relabelDetection(event, detectionId, currentSpecies) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    if (typeof WmSpeciesPicker === 'undefined') {
        alert('Species picker not available.');
        return;
    }

    // Determine mount element (inside open modal or body)
    const openModal = document.querySelector('.gallery-modal.show');
    const mountEl = openModal || document.body;

    // Open the shared species picker
    const choice = await WmSpeciesPicker.pickSpecies({
        currentSpecies: currentSpecies,
        detectionId: detectionId,
        mountEl: mountEl,
        title: '🏷️ Relabel Species'
    });

    // User cancelled
    if (!choice) return;

    // Perform the single-item relabel POST
    try {
        const resp = await fetch('/api/detections/relabel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ detection_id: detectionId, species: choice.scientific })
        });
        if (isAuthRedirect(resp)) {
            redirectToLogin();
            return;
        }
        if (resp.ok) {
            // Species changed — reload to get correct groupings and status
            location.reload();
            return;
        } else {
            const data = await resp.json();
            alert('Relabel failed: ' + (data.error || data.message || 'Unknown error'));
        }
    } catch (err) {
        console.error('Relabel error:', err);
        alert('Error relabeling detection.');
    }
}

/* =========================================
   Keyboard Shortcuts
   ========================================= */
document.addEventListener('keydown', function (event) {
    // Check if any modal is open
    const openModal = document.querySelector('.gallery-modal.show');
    if (!openModal) return;

    if (event.key === 'ArrowLeft') {
        const prevBtn = openModal.querySelector('.prev-btn');
        if (prevBtn) navigateModal(prevBtn, 'prev');
    } else if (event.key === 'ArrowRight') {
        const nextBtn = openModal.querySelector('.next-btn');
        if (nextBtn) navigateModal(nextBtn, 'next');
    }
});

/* =========================================
   Bounding Box Visualization
   ========================================= */

// Wong (2011) colour-blind-friendly palette mirrored from
// :root --species-colour-N tokens in design-system.css. The palette is
// read live from CSS custom properties so canvas bbox strokes follow
// the active theme without duplicating color logic in JavaScript.
const SPECIES_COLOURS_FALLBACK = [
    '#0072B2', // 0 blue
    '#E69F00', // 1 orange
    '#009E73', // 2 green
    '#CC79A7', // 3 pink
    '#56B4E9', // 4 sky
    '#D55E00', // 5 vermilion
    '#F0E442', // 6 yellow (non-text only — light theme constraint)
    '#000000', // 7 black fallback (light theme)
];

let _speciesColoursCache = null;

function getSpeciesColours() {
    if (_speciesColoursCache) return _speciesColoursCache;
    // SSR / detached-node safety net — if document.documentElement is
    // not available (unlikely in a browser) or the computed style has
    // no custom property set, fall back to the light Wong palette.
    try {
        const rootStyle = getComputedStyle(document.documentElement);
        const resolved = SPECIES_COLOURS_FALLBACK.map(function (fallback, idx) {
            const value = rootStyle.getPropertyValue('--species-colour-' + idx).trim();
            return value || fallback;
        });
        _speciesColoursCache = resolved;
        return resolved;
    } catch (err) {
        return SPECIES_COLOURS_FALLBACK;
    }
}

// Backwards-compatibility: the old static SPECIES_COLOURS constant is
// still referenced by tests and potentially by other scripts. Expose it
// as a read-only getter-like binding that always returns the currently
// resolved palette.
const SPECIES_COLOURS = new Proxy([], {
    get: function (_target, prop) {
        const palette = getSpeciesColours();
        if (prop === 'length') return palette.length;
        if (typeof prop === 'string' && /^\d+$/.test(prop)) {
            return palette[Number(prop)];
        }
        return palette[prop];
    },
});

// Invalidate the cache when the OS theme preference changes so the
// next canvas draw picks up the new Wong palette variant. Guarded so
// the test-jsdom environment (which may not implement matchMedia)
// stays happy.
try {
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
        const darkMql = window.matchMedia('(prefers-color-scheme: dark)');
        const invalidate = function () { _speciesColoursCache = null; };
        if (typeof darkMql.addEventListener === 'function') {
            darkMql.addEventListener('change', invalidate);
        } else if (typeof darkMql.addListener === 'function') {
            // Legacy Safari fallback.
            darkMql.addListener(invalidate);
        }
    }
} catch (err) {
    /* no-op */
}

/**
 * Drop label text gracefully when the bbox is too narrow.
 *
 * Priority order: full common name → first 4 chars + ".'" → first
 * letter → empty string. The container border + species colour still
 * carry the species identity even when the text is dropped.
 */
function truncateBboxLabel(text, maxPx, ctx) {
    if (!text) return '';
    if (ctx.measureText(text).width <= maxPx) return text;
    const abbr = text.slice(0, 4) + '.';
    if (ctx.measureText(abbr).width <= maxPx) return abbr;
    const initial = text.charAt(0);
    if (ctx.measureText(initial).width <= maxPx) return initial;
    return '';
}

/**
 * Resolve a species colour for a bbox draw call. Reads
 * ``box.speciesColour`` (numeric slot 0..7) when present and falls back
 * to the legacy ``BBOX_COLORS`` rotation otherwise.
 */
function resolveBboxColour(box, idx) {
    const slot = Number(box && box.speciesColour);
    if (Number.isFinite(slot) && slot >= 0 && slot < SPECIES_COLOURS.length) {
        return SPECIES_COLOURS[slot];
    }
    return box && box.isCurrent
        ? BBOX_COLORS[0]
        : BBOX_COLORS[(idx % (BBOX_COLORS.length - 1)) + 1];
}

// Color palette for bounding boxes (distinct colors for multiple detections)
const BBOX_COLORS = [
    '#FF6B6B', // coral red
    '#4ECDC4', // teal
    '#FFE66D', // yellow
    '#95E1D3', // mint
    '#F38181', // salmon
    '#AA96DA', // purple
    '#FCBAD3', // pink
    '#A8D8EA', // light blue
];

/**
 * Initialize bbox overlay canvas when image loads
 */
function initBboxOverlay(img) {
    const container = img.closest('.modal-image-viewer');
    if (!container) return;
    const scope = getViewerScope(img);
    const host = getViewerHost(img);

    const canvas = container.querySelector('.bbox-overlay');
    if (!canvas) return;

    const apply = function () {
        // Canvas backing store sizing is now owned entirely by
        // drawBoundingBoxes (hi-DPI + smart-zoom-aware). initBboxOverlay
        // no longer sets canvas.width / canvas.height to avoid a brief
        // window where the canvas is dimensioned at raw CSS pixels
        // before the first draw upgrades it. We still cache the natural
        // dimensions for coordinate calculations that happen before the
        // first draw.
        canvas.dataset.naturalWidth = img.naturalWidth;
        canvas.dataset.naturalHeight = img.naturalHeight;

        const prefs = getViewerPrefKeys(scope);
        if (localStorage.getItem(prefs.bbox) !== 'off' && localStorage.getItem(prefs.bbox) !== 'on') {
            localStorage.setItem(prefs.bbox, prefs.bboxDefault);
        }

        if (localStorage.getItem(prefs.bbox) === 'on') {
            const btn = host?.querySelector('.bbox-toggle');
            if (btn) {
                if (!btn.classList.contains('active')) toggleBboxOverlay(btn);
                else redrawBboxOverlay(btn);
            } else if (host?.dataset.contextOnly === '1') {
                const x = parseFloat(container.dataset.bboxX);
                const y = parseFloat(container.dataset.bboxY);
                const w = parseFloat(container.dataset.bboxW);
                const h = parseFloat(container.dataset.bboxH);
                if (!isNaN(x) && !isNaN(y) && !isNaN(w) && !isNaN(h) && w > 0 && h > 0) {
                    const speciesColour = host.dataset.speciesColour !== undefined
                        ? Number(host.dataset.speciesColour)
                        : undefined;
                    const detectionId = Number(img.dataset.detectionId || 0);
                    canvas.style.display = 'block';
                    drawBoundingBoxes(canvas, img, [{
                        x: x,
                        y: y,
                        w: w,
                        h: h,
                        name: img.alt || 'Detection',
                        id: detectionId,
                        speciesColour: speciesColour,
                        isCurrent: true
                    }], detectionId || null);
                }
            }
        }
    };

    // Defer until image has layout dimensions (may be 0 during modal transition)
    if (img.clientWidth > 0) {
        apply();
    } else {
        requestAnimationFrame(function () { requestAnimationFrame(apply); });
    }
}

/**
 * Toggle bounding box overlay visibility
 */
function toggleBboxOverlay(btn) {
    const scope = getViewerScope(btn);
    const host = getViewerHost(btn);
    if (!scope) return;
    const prefs = getViewerPrefKeys(scope);

    const container = host?.querySelector('.modal-image-viewer');
    const canvas = container?.querySelector('.bbox-overlay');
    const img = container?.querySelector('.bbox-base-image');

    if (!canvas || !img) return;

    const isVisible = btn.classList.contains('active');

    if (isVisible) {
        // Hide overlay
        canvas.style.display = 'none';
        btn.textContent = 'Boxes';
        btn.classList.remove('active', 'btn-secondary', 'btn--secondary');
        btn.classList.add('btn-outline-secondary', 'btn--outline-secondary');
        localStorage.setItem(prefs.bbox, 'off');
    } else {
        // Show and draw overlay
        canvas.style.display = 'block';
        btn.textContent = 'Boxes ✓';
        btn.classList.add('active', 'btn-secondary', 'btn--secondary');
        btn.classList.remove('btn-outline-secondary', 'btn--outline-secondary');
        localStorage.setItem(prefs.bbox, 'on');

        // Collect all bounding boxes
        const currentBbox = JSON.parse(btn.dataset.currentBbox || '{}');
        let siblings = [];
        try {
            siblings = JSON.parse(btn.dataset.siblings || '[]');
        } catch (e) {
            console.error('Failed to parse siblings:', e);
        }

        // Build box list: use siblings if available (includes current), else just current
        let boxes = [];
        if (siblings && siblings.length > 0) {
            boxes = siblings.map((sib, idx) => ({
                x: sib.bbox_x,
                y: sib.bbox_y,
                w: sib.bbox_w,
                h: sib.bbox_h,
                name: sib.common_name,
                id: sib.detection_id,
                isCurrent: sib.detection_id === currentBbox.id
            }));
        } else if (currentBbox.x !== undefined) {
            boxes = [{
                x: currentBbox.x,
                y: currentBbox.y,
                w: currentBbox.w,
                h: currentBbox.h,
                name: currentBbox.name,
                id: currentBbox.id,
                isCurrent: true
            }];
        }

        drawBoundingBoxes(canvas, img, boxes, currentBbox.id);
    }
}

/**
 * Redraw bounding boxes on an already-active overlay (e.g. after image load or navigation).
 */
function redrawBboxOverlay(btn) {
    const scope = getViewerScope(btn);
    const host = getViewerHost(btn);
    if (!scope) return;

    const container = host?.querySelector('.modal-image-viewer');
    const canvas = container?.querySelector('.bbox-overlay');
    const img = container?.querySelector('.bbox-base-image');
    if (!canvas || !img) return;

    // Do NOT set canvas.width / canvas.height here. The hi-DPI +
    // zoom-aware setup lives entirely in drawBoundingBoxes so there is
    // one source of truth for the backing-store size. Resetting the
    // size to clientWidth/clientHeight would stomp on that and
    // reintroduce the blurry CSS-pixel backing.
    canvas.dataset.naturalWidth = img.naturalWidth;
    canvas.dataset.naturalHeight = img.naturalHeight;

    const currentBbox = JSON.parse(btn.dataset.currentBbox || '{}');
    let siblings = [];
    try { siblings = JSON.parse(btn.dataset.siblings || '[]'); } catch (e) { /* ignore */ }

    // Propagate the workspace-scoped species slot from the host element
    // through to each draw call so resolveBboxColour() prefers the
    // Wong token over the legacy fallback rotation.
    const hostEl = btn.closest('[data-species-colour]');
    const hostSlot = hostEl ? Number(hostEl.dataset.speciesColour) : NaN;
    const fallbackSlot = Number.isFinite(hostSlot) ? hostSlot : undefined;
    const currentSlot = currentBbox && currentBbox.speciesColour !== undefined
        ? Number(currentBbox.speciesColour)
        : fallbackSlot;

    let boxes = [];
    if (siblings && siblings.length > 0) {
        boxes = siblings.map(function (sib) {
            const sibSlot = sib && sib.species_colour !== undefined
                ? Number(sib.species_colour)
                : fallbackSlot;
            return { x: sib.bbox_x, y: sib.bbox_y, w: sib.bbox_w, h: sib.bbox_h,
                     name: sib.common_name, id: sib.detection_id,
                     speciesColour: sibSlot,
                     isCurrent: sib.detection_id === currentBbox.id };
        });
    } else if (currentBbox.x !== undefined) {
        boxes = [{ x: currentBbox.x, y: currentBbox.y, w: currentBbox.w, h: currentBbox.h,
                   name: currentBbox.name, id: currentBbox.id,
                   speciesColour: currentSlot,
                   isCurrent: true }];
    }

    drawBoundingBoxes(canvas, img, boxes, currentBbox.id);
}

/**
 * Draw bounding boxes on canvas
 */
function drawBoundingBoxes(canvas, img, boxes, currentDetectionId) {
    const ctx = canvas.getContext('2d');

    // ──────────────────────────────────────────────────────────────
    // Hi-DPI + zoom-aware canvas setup (2026-04-08)
    //
    // Before this fix the canvas was set up with the raw CSS pixel
    // dimensions of the image, which caused two compounding quality
    // issues:
    //
    //  1. On Retina displays (DPR ≥ 2) every drawn pixel was
    //     upsampled by the browser to 4-9 physical screen pixels,
    //     which blurred strokes and text.
    //  2. Smart Zoom scales the canvas with CSS `transform: scale()`
    //     — that multiplies every already-rasterised pixel by the
    //     zoom factor. Combined with DPR on Retina the effective
    //     upsample hit ~8x at 4x zoom, which is exactly the
    //     "pixelated text at crop zoom" symptom.
    //
    // Fix:
    //  • Read devicePixelRatio.
    //  • Detect whether the viewer is currently smart-zoomed and
    //    extract the scale factor from the image's inline transform.
    //  • Allocate the canvas backing store at
    //    `clientSize * DPR * zoomScale`, capped at 4× total density
    //    so large zooms on Retina don't explode RAM.
    //  • Keep the canvas CSS size at `clientSize` so the layout and
    //    the CSS transform keep working unchanged.
    //  • Pre-scale the 2d context so the draw code below can stay in
    //    CSS-pixel coordinates (12px font stays 12 CSS pixels, etc.)
    //    but render into the higher-density backing store.
    //  • Finally, divide every visual dimension (font size, stroke
    //    width, padding) by the zoomScale at draw time, because the
    //    CSS transform applied by Smart Zoom enlarges the already-
    //    rendered output. Without that division a 12px label would
    //    balloon to 12 * zoomScale on screen and overwhelm the cell.
    // ──────────────────────────────────────────────────────────────
    const cssW = img.clientWidth;
    const cssH = img.clientHeight;
    if (cssW <= 0 || cssH <= 0) return;

    const dpr = Math.max(1, window.devicePixelRatio || 1);

    // Detect current Smart Zoom factor by inspecting the image's
    // inline transform. The transform is set by applySmartZoom as
    // `scale(${scale.toFixed(3)}) translate(...)`. When no zoom is
    // active the transform is empty and zoomScale stays at 1.
    let zoomScale = 1;
    const viewer = img.closest('.wm-image-viewer');
    if (viewer && viewer.classList.contains('wm-image-viewer--zoomed')) {
        const tf = img.style.transform || '';
        const m = tf.match(/scale\(([0-9.]+)\)/);
        if (m) {
            const parsed = parseFloat(m[1]);
            if (Number.isFinite(parsed) && parsed > 0) zoomScale = parsed;
        }
    }

    // Cap effective backing density at 4× so a 5x zoom on a Retina
    // display doesn't allocate a 100-megapixel canvas. 4× is enough
    // oversampling to keep strokes and text crisp at any practical
    // viewport size.
    const MAX_DENSITY = 4;
    const density = Math.min(dpr * zoomScale, MAX_DENSITY);

    canvas.width = Math.round(cssW * density);
    canvas.height = Math.round(cssH * density);
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';

    // Reset any prior transform, then scale so (density) backing px
    // = 1 CSS px. Draw code below can now pretend it's drawing at
    // CSS pixel resolution.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(density, density);
    ctx.clearRect(0, 0, cssW, cssH);

    // Quality hints — not all browsers honour these on every pass,
    // but they're cheap and nudge stroke / text rasterisation toward
    // the sharper end when honoured.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Scale factors (bbox coords are normalized 0-1). Now in CSS px.
    const scaleX = cssW;
    const scaleY = cssH;

    // Visual dimensions compensate for Smart Zoom's CSS transform so
    // the label and stroke look the same size on screen regardless
    // of zoom level. On zoomScale=1 (no zoom) these collapse to the
    // legacy values.
    const inv = 1 / zoomScale;
    const fontPx = 12 * inv;
    const labelHeight = 18 * inv;
    const labelPadX = 4 * inv;
    const labelPadY = 2 * inv;
    const strokeCurrent = 3 * inv;
    const strokeOther = 2 * inv;

    boxes.forEach((box, idx) => {
        if (!box.x && !box.y && !box.w && !box.h) return; // Skip empty boxes

        // Calculate pixel coordinates
        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        // Prefer the species-colour slot when present so every frame of
        // the same species shares one bbox stroke colour.
        const color = resolveBboxColour(box, idx);

        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = box.isCurrent ? strokeCurrent : strokeOther;
        ctx.strokeRect(x, y, w, h);

        // Responsive label: shrink the text by step rather than the
        // font. On narrow viewports start at the abbreviated level so
        // labels never dominate touch.
        const fullLabel = box.name || 'Detection';
        ctx.font = 'bold ' + fontPx + 'px system-ui, -apple-system, sans-serif';
        const labelMaxWidth = Math.max(0, w - labelPadX * 2);
        const isNarrowViewport = window.innerWidth < 640;
        const startingLabel = isNarrowViewport && fullLabel.length > 4
            ? fullLabel.slice(0, 4) + '.'
            : fullLabel;
        const label = truncateBboxLabel(startingLabel, labelMaxWidth, ctx);

        if (label) {
            const textMetrics = ctx.measureText(label);
            const labelWidth = textMetrics.width + labelPadX * 2;

            // Position label above box, or below if too close to top
            let labelY = y - labelHeight - labelPadY;
            if (labelY < 0) labelY = y + h + labelPadY;

            ctx.fillStyle = color;
            ctx.fillRect(x, labelY, labelWidth, labelHeight);

            // Yellow (slot 6) has insufficient contrast against the
            // black default text fill. Switch to white-on-black for
            // that one slot. Every other slot keeps the legacy
            // black-on-fill treatment.
            ctx.fillStyle = '#000';
            // Baseline offset tracks the label height so the text
            // still sits vertically centred inside the scaled label
            // box. 13/18 ≈ 0.72.
            ctx.fillText(label, x + labelPadX, labelY + labelHeight * (13 / 18));
        }
    });
}

// Re-draw boxes on window resize
window.addEventListener('resize', function () {
    const openModal = document.querySelector('.gallery-modal.show');
    if (!openModal) return;

    const canvas = openModal.querySelector('.bbox-overlay');
    if (!canvas || canvas.style.display === 'none') return;

    const btn = openModal.querySelector('.bbox-toggle.active');
    if (btn) {
        // Re-trigger drawing with current state
        canvas.style.display = 'none';
        toggleBboxOverlay(btn);
    }
});

/* =========================================
   Hover Bounding Box Preview
   ========================================= */

/**
 * Show bounding box overlay when hovering over a detection card
 */
function showHoverBbox(cardEl) {
    const modal = cardEl.closest('.modal');
    if (!modal) return;

    // Add visual highlight to card
    cardEl.style.background = 'rgba(13, 110, 253, 0.1)';
    cardEl.style.borderColor = '#0d6efd';

    const container = modal.querySelector('.modal-image-viewer');
    const canvas = container?.querySelector('.bbox-overlay');
    const img = container?.querySelector('.bbox-base-image');

    if (!canvas || !img) return;

    // Get bbox data from card
    const x = parseFloat(cardEl.dataset.bboxX) || 0;
    const y = parseFloat(cardEl.dataset.bboxY) || 0;
    const w = parseFloat(cardEl.dataset.bboxW) || 0;
    const h = parseFloat(cardEl.dataset.bboxH) || 0;
    const name = cardEl.dataset.bboxName || 'Detection';

    if (!w && !h) return; // No valid bbox

    // Show canvas and draw single bbox
    canvas.style.display = 'block';

    const box = { x, y, w, h, name, isCurrent: true };
    drawBoundingBoxes(canvas, img, [box], null);
}

/**
 * Hide bounding box overlay when mouse leaves detection card
 */
function hideHoverBbox(cardEl) {
    const modal = cardEl.closest('.modal');
    if (!modal) return;

    // Remove visual highlight from card
    cardEl.style.background = '';
    cardEl.style.borderColor = '';

    const container = modal.querySelector('.modal-image-viewer');
    const canvas = container?.querySelector('.bbox-overlay');

    if (!canvas) return;

    // Check if the "Boxes" toggle is active - if so, don't hide
    const btn = modal.querySelector('.bbox-toggle.active');
    if (btn) {
        // Redraw all boxes instead of hiding
        toggleBboxOverlay(btn);
        canvas.style.display = 'block';
        return;
    }

    // Hide canvas
    canvas.style.display = 'none';
}

/* =========================================
   Smart Zoom - Auto-zoom to Bird BBox
   ========================================= */

/**
 * Initialize smart zoom on image load.
 * Reads bbox data from the parent .wm-image-viewer container.
 * If bbox exists and is valid, auto-zooms to that region.
 */
function resetSmartZoomViewer(viewer, img) {
    if (!viewer || !img) return;
    viewer.classList.remove('wm-image-viewer--zoomed');
    img.style.transform = '';
    img.style.transformOrigin = '';

    const canvas = viewer.querySelector('.bbox-overlay');
    if (canvas) {
        canvas.style.transform = '';
        canvas.style.transformOrigin = '';
    }
}

function getSmartZoomViewerState(viewer) {
    if (!viewer) {
        return { hasBbox: false, bx: NaN, by: NaN, bw: NaN, bh: NaN };
    }
    const bx = parseFloat(viewer.dataset.bboxX);
    const by = parseFloat(viewer.dataset.bboxY);
    const bw = parseFloat(viewer.dataset.bboxW);
    const bh = parseFloat(viewer.dataset.bboxH);
    const hasBbox = !isNaN(bx) && !isNaN(by) && !isNaN(bw) && !isNaN(bh) && bw > 0 && bh > 0;
    return { hasBbox, bx, by, bw, bh };
}

function applySmartZoomPreferenceToScope(scope) {
    if (!scope || !scope.querySelectorAll) return;
    const prefs = getViewerPrefKeys(scope);
    const storedPref = localStorage.getItem(prefs.zoom);
    const viewerHosts = Array.from(scope.querySelectorAll('.wm-toolbox-host')).filter(function (host) {
        return host.querySelector('.wm-image-viewer') && host.querySelector('.smart-zoom-toggle');
    });

    viewerHosts.forEach(function (host) {
        const viewer = host.querySelector('.wm-image-viewer');
        const img = host.querySelector('.wm-image-viewer__img');
        const zoomBtn = host.querySelector('.smart-zoom-toggle');
        if (!viewer || !img || !zoomBtn) return;

        const state = getSmartZoomViewerState(viewer);
        if (!state.hasBbox) {
            resetSmartZoomViewer(viewer, img);
            zoomBtn.style.display = 'none';
            return;
        }

        zoomBtn.style.display = '';
        if (storedPref === 'full') {
            resetSmartZoomViewer(viewer, img);
            zoomBtn.classList.remove('active');
            zoomBtn.textContent = '🔍 Zoom';
            applySmartZoomToggleState(zoomBtn, false, true);
        } else {
            applySmartZoom(viewer, img, state.bx, state.by, state.bw, state.bh);
            zoomBtn.classList.add('active');
            zoomBtn.textContent = '🖼 Full';
            applySmartZoomToggleState(zoomBtn, true, true);
        }
    });
}

function initSmartZoom(img) {
    const viewer = img.closest('.wm-image-viewer');
    if (!viewer) return;
    const scope = getViewerScope(viewer);
    const host = getViewerHost(viewer);
    const prefs = getViewerPrefKeys(scope);
    const zoomBtn = host ? host.querySelector('.smart-zoom-toggle') : null;
    const state = getSmartZoomViewerState(viewer);

    // Only zoom if we have valid bbox data
    if (!state.hasBbox) {
        // No bbox → hide zoom button
        if (zoomBtn) zoomBtn.style.display = 'none';
        return;
    }
    if (zoomBtn) zoomBtn.style.display = '';

    // Respect stored zoom preference: if user chose 'full', skip auto-zoom
    if (localStorage.getItem(prefs.zoom) !== 'full' && localStorage.getItem(prefs.zoom) !== 'zoom') {
        localStorage.setItem(prefs.zoom, 'zoom');
    }
    const storedPref = localStorage.getItem(prefs.zoom);
    if (storedPref === 'full') {
        // Ensure full-image state
        resetSmartZoomViewer(viewer, img);
        if (zoomBtn) {
            zoomBtn.classList.remove('active');
            zoomBtn.textContent = '🔍 Zoom';
            applySmartZoomToggleState(zoomBtn, false, true);
        }
        return;
    }

    // Default behavior (or storedPref === 'zoom'): auto-zoom to bbox
    applySmartZoom(viewer, img, state.bx, state.by, state.bw, state.bh);
    if (zoomBtn) applySmartZoomToggleState(zoomBtn, true, true);
}

/**
 * Apply CSS transform to zoom into the bbox region.
 * Replicates server-side CropService.create_thumbnail_crop() logic:
 * - Square side = max(bbox_w, bbox_h) * (1 + expansion)
 * - Centered on bbox center
 * - Edge-shift clamping (shift instead of clip at edges)
 *
 * Uses transform-origin: 0 0 with scale() + translate() to correctly
 * pan and zoom so the crop region fills the visible container.
 *
 * bbox values are fractional (0-1) relative to image dimensions.
 * Expansion is 80% (larger than server 50%) to show more context.
 */
function applySmartZoom(viewer, img, bx, by, bw, bh) {
    // Match CropService logic: square side = max(w,h) * (1 + expansion)
    const EXPANSION = 0.80; // 80% expansion for comfortable zoom level
    const side = Math.max(bw, bh) * (1 + EXPANSION);

    if (side >= 0.80) {
        // Bird already fills most of the frame, no zoom needed
        viewer.classList.remove('wm-image-viewer--zoomed');
        img.style.transform = '';
        img.style.transformOrigin = '';
        return;
    }

    // Center of bbox (fractional)
    let cx = bx + bw / 2;
    let cy = by + bh / 2;

    // Compute square crop region (fractional 0-1)
    let sqX1 = cx - side / 2;
    let sqY1 = cy - side / 2;
    let sqX2 = sqX1 + side;
    let sqY2 = sqY1 + side;

    // Edge-shift clamping (same as CropService)
    if (sqX1 < 0) { sqX2 -= sqX1; sqX1 = 0; }
    if (sqY1 < 0) { sqY2 -= sqY1; sqY1 = 0; }
    if (sqX2 > 1) { sqX1 -= (sqX2 - 1); sqX2 = 1; }
    if (sqY2 > 1) { sqY1 -= (sqY2 - 1); sqY2 = 1; }
    sqX1 = Math.max(0, sqX1);
    sqY1 = Math.max(0, sqY1);

    // Scale factor: how much to zoom in
    const scale = 1 / side;

    // Use transform-origin: 0 0 with scale + translate.
    // CSS transforms apply right-to-left:
    //   1. translate: moves image so crop top-left (sqX1, sqY1) is at (0, 0)
    //   2. scale: zooms from (0, 0), making the crop fill the entire container
    const tx = -(sqX1 * 100);  // percentage of element width
    const ty = -(sqY1 * 100);  // percentage of element height

    const transformCSS = `scale(${scale.toFixed(3)}) translate(${tx.toFixed(2)}%, ${ty.toFixed(2)}%)`;
    img.style.transformOrigin = '0 0';
    img.style.transform = transformCSS;
    viewer.classList.add('wm-image-viewer--zoomed');

    // Sync bbox overlay canvas with the same transform so boxes stay aligned
    const canvas = viewer.querySelector('.bbox-overlay');
    if (canvas) {
        canvas.style.transformOrigin = '0 0';
        canvas.style.transform = transformCSS;
    }

    // Update button state
    const host = getViewerHost(viewer);
    if (host) {
        const zoomBtn = host.querySelector('.smart-zoom-toggle');
        if (zoomBtn) {
            zoomBtn.classList.add('active');
            zoomBtn.textContent = '🖼 Full';
        }
    }
}

/**
 * Reflect the current zoom state on a `.smart-zoom-toggle` button so
 * pressed-state is visible without rewriting the zoom math.
 *
 * Contract:
 *   - aria-pressed mirrors the viewer's zoomed state
 *   - wm-toolbox__btn--toggled carries the pressed-state CSS hook
 *   - title is state-aware:
 *       "Zoom into bird"              — when the next click will zoom in
 *       "Show full image"             — when the next click will zoom out
 *       "No bounding box — zoom unavailable for this frame" — when the
 *         persisted intent is 'zoom' but the current viewer has no bbox
 *
 * `hasBbox` distinguishes "truly zoomable" (real bbox data on the
 * viewer) from "no bbox visible, button stays pressed as intent". The
 * emoji label swap on `textContent` stays for cross-surface
 * consistency with Gallery/Stream viewers.
 */
function applySmartZoomToggleState(btn, isZoomed, hasBbox) {
    if (!btn) return;
    btn.classList.toggle('wm-toolbox__btn--toggled', isZoomed);
    btn.setAttribute('aria-pressed', String(isZoomed));
    if (isZoomed && !hasBbox) {
        btn.title = 'No bounding box — zoom unavailable for this frame';
    } else if (isZoomed) {
        btn.title = 'Show full image';
    } else {
        btn.title = 'Zoom into bird';
    }
}

/**
 * Toggle between zoomed (bbox close-up) and full image view.
 */
function toggleSmartZoom(btn) {
    const scope = getViewerScope(btn);
    if (!scope) return;

    const prefs = getViewerPrefKeys(scope);
    const nextPref = localStorage.getItem(prefs.zoom) === 'zoom' ? 'full' : 'zoom';
    localStorage.setItem(prefs.zoom, nextPref);
    applySmartZoomPreferenceToScope(scope);

    // Redraw every active bbox overlay in the shared workbench after the
    // zoom preference changed so canvases stay aligned with their cells.
    scope.querySelectorAll('.bbox-toggle.active').forEach(function (bboxBtn) {
        requestAnimationFrame(function () { redrawBboxOverlay(bboxBtn); });
    });
}

// Reset zoom state when navigating between modals
document.addEventListener('shown.bs.modal', function (event) {
    const modal = event.target;
    if (!modal.classList.contains('gallery-modal')) return;

    // The img onload handler will take care of applying zoom
    // But if image is already cached, we may need to trigger manually
    const img = modal.querySelector('.wm-image-viewer__img');
    if (img && img.complete && img.naturalWidth > 0) {
        if (typeof initSmartZoom === 'function') initSmartZoom(img);
        if (typeof initBboxOverlay === 'function') initBboxOverlay(img);
    }
});

/* =========================================
   Image Viewer Init (replaces inline onload)
   ========================================= */

// Delegated load handler for modal image viewers
document.addEventListener('load', function (event) {
    const img = event.target;
    if (!img.classList || !img.classList.contains('wm-image-viewer__img')) return;
    if (typeof initBboxOverlay === 'function') initBboxOverlay(img);
    if (typeof initSmartZoom === 'function') initSmartZoom(img);
}, true);

/* =========================================
   Delegated Sibling Card Handlers
   ========================================= */

// Delegated click handler for sibling-card data-action buttons
document.addEventListener('click', function (event) {
    const btn = event.target.closest('[data-action]');
    if (!btn) return;
    const card = btn.closest('.sibling-card');
    if (!card) return;

    const action = btn.dataset.action;
    const detectionId = parseInt(btn.dataset.detectionId, 10);

    if (action === 'change-species') {
        const currentSpecies = btn.dataset.currentSpecies || '';
        relabelDetection(event, detectionId, currentSpecies);
    } else if (action === 'move-trash') {
        deleteDetection(event, detectionId);
    }
});

// Delegated hover handlers for sibling-card bbox preview
document.addEventListener('mouseenter', function (event) {
    if (!(event.target instanceof Element)) return;
    const card = event.target.closest('.sibling-card');
    if (card && typeof showHoverBbox === 'function') showHoverBbox(card);
}, true);

document.addEventListener('mouseleave', function (event) {
    if (!(event.target instanceof Element)) return;
    const card = event.target.closest('.sibling-card');
    if (card && typeof hideHoverBbox === 'function') hideHoverBbox(card);
}, true);
