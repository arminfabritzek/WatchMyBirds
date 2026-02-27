/**
 * Shared Gallery Navigation Logic for WatchMyBirds
 * Handles modal navigation, keyboard shortcuts, deletion, relabeling, and favorites.
 */

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
        if (resp.ok) {
            // Check if we hit an auth redirect (fetch follows 302s to /login transparently, returning HTML with 200 OK)
            const contentType = resp.headers.get("content-type");
            if (contentType && contentType.includes("text/html")) {
                if (window.wmToast) window.wmToast('Session expired. Please log in.', 'error');
                window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
                return;
            }

            step = "json";
            const data = await resp.json();
            const isFav = Boolean(data.is_favorite);

            step = "btn-toggle";
            // Update the button in the modal or the hovered badge
            if (btn && btn.classList) {
                btn.classList.toggle('fav-btn--active', isFav);
                btn.classList.toggle('wm-tile__fav-badge--active', isFav);
                btn.textContent = isFav ? '⭐' : '☆';
            }

            step = "dom-queries";
            try {
                // Also update the counterpart (if clicking on tile, update modal; if modal, update tile)
                const tileBadge = document.querySelector(`.wm-tile[data-detection-id="${detectionId}"] .wm-tile__fav-badge`);
                if (tileBadge && tileBadge !== btn) {
                    tileBadge.classList.toggle('wm-tile__fav-badge--active', isFav);
                    tileBadge.textContent = isFav ? '⭐' : '☆';
                }

                const modalBtn = document.querySelector(`.gallery-modal[data-detection-id="${detectionId}"] .fav-btn`);
                if (modalBtn && modalBtn !== btn) {
                    modalBtn.classList.toggle('fav-btn--active', isFav);
                    modalBtn.textContent = isFav ? '⭐' : '☆';
                }
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
function navigateModal(btn, direction) {
    const currentModalEl = btn.closest('.modal');
    if (!currentModalEl) return;

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

    // Hide current modal
    const currentInstance = bootstrap.Modal.getInstance(currentModalEl);
    if (currentInstance) {
        currentInstance.hide();
    }

    // Show next modal
    const nextInstance = new bootstrap.Modal(nextModalEl);
    nextInstance.show();
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
   Relabel Logic
   ========================================= */
let _speciesCache = null;

async function loadSpeciesList() {
    if (_speciesCache) return _speciesCache;
    try {
        const resp = await fetch('/api/species-list');
        const data = await resp.json();
        if (data.status === 'success') {
            _speciesCache = data.species; // [{scientific, common}, ...]
            return _speciesCache;
        }
    } catch (e) {
        console.error('Failed to load species list:', e);
    }
    return [];
}

async function relabelDetection(event, detectionId, currentSpecies) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    const species = await loadSpeciesList();
    if (!species.length) {
        alert('Artenliste konnte nicht geladen werden.');
        return;
    }

    // Create overlay dropdown
    const openModal = document.querySelector('.gallery-modal.show');
    const container = openModal || document.body;

    // Remove any existing relabel overlay
    const existing = container.querySelector('.relabel-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.className = 'relabel-overlay';
    overlay.style.cssText = `
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.7); z-index: 9999;
        display: flex; align-items: center; justify-content: center;
    `;

    const panel = document.createElement('div');
    panel.style.cssText = `
        background: #1a1d23; border: 1px solid rgba(167,139,250,0.3);
        border-radius: 12px; padding: 16px; width: 320px; max-height: 70vh;
        display: flex; flex-direction: column; gap: 10px;
        font-family: system-ui, -apple-system, sans-serif;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    `;

    // Header
    const header = document.createElement('div');
    header.style.cssText = 'color:#a78bfa;font-weight:600;font-size:0.9rem;';
    header.textContent = '🏷️ Art korrigieren';
    panel.appendChild(header);

    // Search input
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Art suchen...';
    searchInput.style.cssText = `
        width: 100%; padding: 8px 12px; border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.08);
        color: #fff; font-size: 0.85rem; outline: none; box-sizing: border-box;
    `;
    panel.appendChild(searchInput);

    // Species list
    const listEl = document.createElement('div');
    listEl.style.cssText = 'overflow-y: auto; max-height: 45vh; display: flex; flex-direction: column; gap: 2px;';

    function renderList(filter) {
        listEl.innerHTML = '';
        const filtered = filter
            ? species.filter(s => s.common.toLowerCase().includes(filter) || s.scientific.toLowerCase().includes(filter))
            : species;
        for (const sp of filtered) {
            const item = document.createElement('button');
            item.type = 'button';
            const isCurrent = sp.scientific === currentSpecies;
            item.style.cssText = `
                display: flex; justify-content: space-between; align-items: center;
                width: 100%; padding: 8px 10px; border: none; border-radius: 6px;
                cursor: pointer; text-align: left; font-size: 0.8rem;
                background: ${isCurrent ? 'rgba(167,139,250,0.15)' : 'transparent'};
                color: ${isCurrent ? '#a78bfa' : 'rgba(255,255,255,0.85)'};
                transition: background 0.1s;
            `;
            item.onmouseenter = () => { if (!isCurrent) item.style.background = 'rgba(255,255,255,0.08)'; };
            item.onmouseleave = () => { if (!isCurrent) item.style.background = 'transparent'; };
            item.innerHTML = `<span><strong style="font-style:italic;">${sp.scientific.replace(/_/g, ' ')}</strong></span><span style="font-size:0.7rem;color:rgba(255,255,255,0.4);">${sp.common}</span>`;
            item.onclick = () => doRelabel(sp.scientific, sp.common);
            listEl.appendChild(item);
        }
    }

    renderList('');
    panel.appendChild(listEl);

    // Cancel button
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Abbrechen';
    cancelBtn.style.cssText = `
        padding: 6px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15);
        background: transparent; color: rgba(255,255,255,0.6); cursor: pointer;
        font-size: 0.8rem; align-self: flex-end;
    `;
    cancelBtn.onclick = () => overlay.remove();
    panel.appendChild(cancelBtn);

    overlay.appendChild(panel);
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    container.appendChild(overlay);

    // Focus search
    setTimeout(() => searchInput.focus(), 50);
    searchInput.oninput = () => renderList(searchInput.value.toLowerCase());

    // Perform relabel
    async function doRelabel(scientific, common) {
        try {
            const resp = await fetch('/api/detections/relabel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detection_id: detectionId, species: scientific })
            });
            if (resp.ok) {
                overlay.remove();
                // Update the sibling card text in-place
                const card = openModal?.querySelector(`.sibling-card[data-detection-id="${detectionId}"]`);
                if (card) {
                    const nameEl = card.querySelector('div[style*="font-weight: 600"]');
                    if (nameEl) nameEl.textContent = common;
                    card.dataset.bboxName = common;
                    // Flash green briefly
                    card.style.transition = 'background 0.3s';
                    card.style.background = 'rgba(52,211,153,0.2)';
                    setTimeout(() => { card.style.background = ''; }, 1200);
                }
                // Also update modal title if this is the main detection
                const titleEl = openModal?.querySelector('.wm-modal__title-text');
                if (titleEl && titleEl.textContent.trim().includes(currentSpecies?.replace(/_/g, ' '))) {
                    titleEl.innerHTML = `<em>${common}</em>`;
                }
            } else {
                const data = await resp.json();
                alert('Relabel fehlgeschlagen: ' + (data.error || data.message || 'Unbekannter Fehler'));
            }
        } catch (err) {
            console.error('Relabel error:', err);
            alert('Fehler beim Umbenennen.');
        }
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

    const canvas = container.querySelector('.bbox-overlay');
    if (!canvas) return;

    // Match canvas size to displayed image size
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;

    // Store natural dimensions for coordinate calculation
    canvas.dataset.naturalWidth = img.naturalWidth;
    canvas.dataset.naturalHeight = img.naturalHeight;

    // Apply stored preference
    if (localStorage.getItem('wmb_modal_bbox_pref') === 'on') {
        const modal = img.closest('.modal');
        const btn = modal?.querySelector('.bbox-toggle');
        if (btn && !btn.classList.contains('active')) {
            toggleBboxOverlay(btn);
        }
    }
}

/**
 * Toggle bounding box overlay visibility
 */
function toggleBboxOverlay(btn) {
    const modal = btn.closest('.modal');
    if (!modal) return;

    const container = modal.querySelector('.modal-image-viewer');
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
        localStorage.setItem('wmb_modal_bbox_pref', 'off');
    } else {
        // Show and draw overlay
        canvas.style.display = 'block';
        btn.textContent = 'Boxes ✓';
        btn.classList.add('active', 'btn-secondary', 'btn--secondary');
        btn.classList.remove('btn-outline-secondary', 'btn--outline-secondary');
        localStorage.setItem('wmb_modal_bbox_pref', 'on');

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
 * Draw bounding boxes on canvas
 */
function drawBoundingBoxes(canvas, img, boxes, currentDetectionId) {
    const ctx = canvas.getContext('2d');

    // Ensure canvas matches image size
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Scale factors (bbox coords are normalized 0-1)
    const scaleX = canvas.width;
    const scaleY = canvas.height;

    boxes.forEach((box, idx) => {
        if (!box.x && !box.y && !box.w && !box.h) return; // Skip empty boxes

        // Calculate pixel coordinates
        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        // Choose color (current detection gets first color, others cycle)
        const color = box.isCurrent ? BBOX_COLORS[0] : BBOX_COLORS[(idx % (BBOX_COLORS.length - 1)) + 1];

        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = box.isCurrent ? 3 : 2;
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        const label = box.name || 'Detection';
        ctx.font = 'bold 12px system-ui, -apple-system, sans-serif';
        const textMetrics = ctx.measureText(label);
        const labelHeight = 18;
        const labelWidth = textMetrics.width + 8;

        // Position label above box, or below if too close to top
        let labelY = y - labelHeight - 2;
        if (labelY < 0) labelY = y + h + 2;

        ctx.fillStyle = color;
        ctx.fillRect(x, labelY, labelWidth, labelHeight);

        // Draw label text
        ctx.fillStyle = '#000';
        ctx.fillText(label, x + 4, labelY + 13);
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
function initSmartZoom(img) {
    const viewer = img.closest('.wm-image-viewer');
    if (!viewer) return;

    const bx = parseFloat(viewer.dataset.bboxX);
    const by = parseFloat(viewer.dataset.bboxY);
    const bw = parseFloat(viewer.dataset.bboxW);
    const bh = parseFloat(viewer.dataset.bboxH);

    // Only zoom if we have valid bbox data
    if (isNaN(bx) || isNaN(by) || isNaN(bw) || isNaN(bh) || bw <= 0 || bh <= 0) {
        // No bbox → hide zoom button
        const modal = viewer.closest('.modal');
        if (modal) {
            const zoomBtn = modal.querySelector('.smart-zoom-toggle');
            if (zoomBtn) zoomBtn.style.display = 'none';
        }
        return;
    }

    // Respect stored zoom preference: if user chose 'full', skip auto-zoom
    const storedPref = localStorage.getItem('wmb_modal_zoom_pref');
    if (storedPref === 'full') {
        // Ensure full-image state
        viewer.classList.remove('wm-image-viewer--zoomed');
        img.style.transform = '';
        img.style.transformOrigin = '';
        const modal = viewer.closest('.modal');
        if (modal) {
            const zoomBtn = modal.querySelector('.smart-zoom-toggle');
            if (zoomBtn) {
                zoomBtn.classList.remove('active');
                zoomBtn.textContent = '🔍 Zoom';
            }
        }
        return;
    }

    // Default behavior (or storedPref === 'zoom'): auto-zoom to bbox
    applySmartZoom(viewer, img, bx, by, bw, bh);
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
    const modal = viewer.closest('.modal');
    if (modal) {
        const zoomBtn = modal.querySelector('.smart-zoom-toggle');
        if (zoomBtn) {
            zoomBtn.classList.add('active');
            zoomBtn.textContent = '🖼 Full';
        }
    }
}

/**
 * Toggle between zoomed (bbox close-up) and full image view.
 */
function toggleSmartZoom(btn) {
    const modal = btn.closest('.modal');
    if (!modal) return;

    const viewer = modal.querySelector('.wm-image-viewer');
    const img = modal.querySelector('.wm-image-viewer__img');
    if (!viewer || !img) return;

    const isZoomed = viewer.classList.contains('wm-image-viewer--zoomed');

    if (isZoomed) {
        // Zoom out → show full image
        viewer.classList.remove('wm-image-viewer--zoomed');
        img.style.transform = '';
        img.style.transformOrigin = '';
        // Reset bbox canvas transform too
        const canvas = viewer.querySelector('.bbox-overlay');
        if (canvas) { canvas.style.transform = ''; canvas.style.transformOrigin = ''; }
        btn.classList.remove('active');
        btn.textContent = '🔍 Zoom';
        // Persist preference: user wants full view
        localStorage.setItem('wmb_modal_zoom_pref', 'full');
    } else {
        // Zoom in → read bbox from viewer data attributes
        const bx = parseFloat(viewer.dataset.bboxX);
        const by = parseFloat(viewer.dataset.bboxY);
        const bw = parseFloat(viewer.dataset.bboxW);
        const bh = parseFloat(viewer.dataset.bboxH);

        if (!isNaN(bx) && !isNaN(by) && bw > 0 && bh > 0) {
            applySmartZoom(viewer, img, bx, by, bw, bh);
            // Persist preference: user wants zoom
            localStorage.setItem('wmb_modal_zoom_pref', 'zoom');
        }
    }
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
