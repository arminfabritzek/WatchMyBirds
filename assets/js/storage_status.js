(() => {
    const notice = document.getElementById('photo-saving-notice');
    if (!notice) return;
    async function update() {
        if (document.hidden) return;
        try {
            const response = await fetch('/api/status', { credentials: 'same-origin' });
            if (!response.ok) return;
            const state = (await response.json()).photo_saving || {};
            const freed = state.last_cleanup?.freed_bytes || 0;
            const cleanup = freed ? `Card nearly full: early deletion recovered ${(freed / 1024 ** 2).toFixed(1)} MiB from unprotected photos. Favorites and training photos remain protected.` : '';
            notice.textContent = [state.message, cleanup].filter(Boolean).join(' ');
            notice.hidden = !notice.textContent;
        } catch (_) {
            // Preserve the last known status during a temporary connection loss.
        }
    }
    update();
    window.setInterval(update, 5000);
})();
