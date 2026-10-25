(() => {
    const url = document.currentScript?.dataset.exportDownload;
    if (!url) return;
    const pageUrl = new URL(window.location.href);
    pageUrl.searchParams.delete('download');
    window.history.replaceState(null, '', pageUrl);
    window.location.assign(url);
})();
