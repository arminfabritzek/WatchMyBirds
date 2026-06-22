/* "Move Review Queue to Trash" — preview-first, reversible bulk cleanup.
 *
 * Fetches the dry-run count, shows the disclosure (favorites / export-relevant
 * are included but counted), then on confirm POSTs the reversible run. Deletes
 * no files; everything lands in Trash and restores from there.
 */
(function () {
    "use strict";

    const openBtn = document.getElementById("reviewClearQueueBtn");
    const modal = document.getElementById("reviewCleanupModal");
    if (!openBtn || !modal) return;

    const body = document.getElementById("reviewCleanupBody");
    const confirmBtn = document.getElementById("reviewCleanupConfirmBtn");

    function show() {
        modal.hidden = false;
    }
    function hide() {
        modal.hidden = true;
        confirmBtn.disabled = true;
    }

    function renderPreview(p) {
        const disclosure =
            p.favorites || p.export_relevant
                ? `<p class="review-cleanup-modal__note">Includes <strong>${p.favorites}</strong> favorite${
                      p.favorites === 1 ? "" : "s"
                  } and <strong>${p.export_relevant}</strong> export-relevant item${
                      p.export_relevant === 1 ? "" : "s"
                  }.</p>`
                : "";
        body.innerHTML =
            `<p>This moves the whole Review queue to Trash:</p>` +
            `<ul class="review-cleanup-modal__counts">` +
            `<li><strong>${p.events}</strong> event${p.events === 1 ? "" : "s"}</li>` +
            `<li><strong>${p.images}</strong> image${p.images === 1 ? "" : "s"}</li>` +
            `<li><strong>${p.detections}</strong> detection${p.detections === 1 ? "" : "s"}</li>` +
            `</ul>` +
            disclosure +
            `<p class="review-cleanup-modal__reassure">No files will be deleted. Items can be restored from Trash.</p>`;
        const nothing = !p.events && !p.images && !p.detections;
        confirmBtn.disabled = nothing;
        if (nothing) {
            body.innerHTML = `<p>The Review queue is already empty.</p>`;
        }
    }

    async function loadPreview() {
        body.innerHTML = `<p class="review-cleanup-modal__loading">Counting review items…</p>`;
        confirmBtn.disabled = true;
        try {
            const r = await fetch("/api/review/cleanup/preview", {
                credentials: "same-origin",
            });
            if (!r.ok) throw new Error("HTTP " + r.status);
            renderPreview(await r.json());
        } catch (e) {
            body.textContent = "Could not load preview. Please try again.";
        }
    }

    async function runCleanup() {
        confirmBtn.disabled = true;
        confirmBtn.textContent = "Moving…";
        try {
            const r = await fetch("/api/review/cleanup/run", {
                method: "POST",
                credentials: "same-origin",
            });
            if (!r.ok) throw new Error("HTTP " + r.status);
            window.location.reload();
        } catch (e) {
            body.textContent = "Cleanup failed. Please try again.";
            confirmBtn.textContent = "Move to Trash";
            confirmBtn.disabled = false;
        }
    }

    openBtn.addEventListener("click", function () {
        if (openBtn.disabled) return;
        show();
        loadPreview();
    });
    confirmBtn.addEventListener("click", runCleanup);
    modal.querySelectorAll("[data-cleanup-dismiss]").forEach(function (el) {
        el.addEventListener("click", hide);
    });
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && !modal.hidden) hide();
    });
})();
