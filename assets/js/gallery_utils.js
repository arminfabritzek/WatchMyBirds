/**
 * Shared Gallery Navigation Logic for WatchMyBirds
 * Handles modal navigation, keyboard shortcuts, and deletion.
 */

/* =========================================
   Modal Navigation (Simple, Fast)
   ========================================= */
function navigateModal(btn, direction) {
    const currentModalEl = btn.closest('.modal');
    if (!currentModalEl) return;

    const group = currentModalEl.getAttribute('data-modal-group');
    if (!group) return;

    // Find all modals in this group
    const allModals = Array.from(document.querySelectorAll(`.gallery-modal[data-modal-group="${group}"]`));
    const currentIndex = allModals.indexOf(currentModalEl);

    if (currentIndex === -1) return;

    // Calculate next index
    let nextIndex;
    if (direction === 'next') {
        nextIndex = currentIndex + 1;
        if (nextIndex >= allModals.length) nextIndex = 0;
    } else {
        nextIndex = currentIndex - 1;
        if (nextIndex < 0) nextIndex = allModals.length - 1;
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

    const isVisible = canvas.style.display !== 'none';

    if (isVisible) {
        // Hide overlay
        canvas.style.display = 'none';
        btn.textContent = 'Boxes';
        btn.classList.remove('active', 'btn-secondary');
        btn.classList.add('btn-outline-secondary');
    } else {
        // Show and draw overlay
        canvas.style.display = 'block';
        btn.textContent = 'Boxes ✓';
        btn.classList.add('active', 'btn-secondary');
        btn.classList.remove('btn-outline-secondary');

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
