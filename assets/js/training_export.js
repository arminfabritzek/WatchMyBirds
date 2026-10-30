(() => {
    const script = document.currentScript;
    const url = script?.dataset.exportDownload;
    const pageUrl = new URL(window.location.href);
    pageUrl.searchParams.delete('download');
    window.history.replaceState(null, '', pageUrl);
    if (url) window.location.assign(url);
    let active = true;
    let timer = null;
    window.addEventListener('pagehide', () => { active = false; clearTimeout(timer); });
    const jobs = [...document.querySelectorAll('[data-export-job]')];
    async function poll() {
        if (!active) return;
        let waiting = false;
        for (const node of jobs) {
            if (!['queued', 'running', 'cancelling'].includes(node.dataset.jobState)) continue;
            waiting = true;
            try {
                const response = await fetch(`/api/canonical-dataset/jobs/${node.dataset.exportJob}`, {cache: 'no-store'});
                if (!response.ok || response.redirected) throw new Error('Status unavailable');
                const job = await response.json();
                if (!active) return;
                const labels = {snapshot: 'Copying metadata', selecting: 'Finding waiting photos', archives: 'Preparing ZIP parts', rebuild: 'Preparing download again'};
                const mib = bytes => (bytes / (1024 * 1024)).toFixed(1);
                const bytes = job.bytes_total ? ` ${mib(job.bytes_done)} / about ${mib(job.bytes_total)} MiB in the current step.` : '';
                const photos = job.phase === 'snapshot' ? '' : ` ${job.processed} / ${job.total} photos;`;
                node.querySelector('[data-job-progress]').textContent = `${labels[job.phase] || job.status}:${photos} ${job.part_count} ZIP parts prepared.${bytes}`;
                if (!['queued', 'running', 'cancelling'].includes(job.status)) {
                    const target = new URL(window.location.href);
                    target.searchParams.delete('job');
                    const first = job.parts.find(part => part.available && (!part.received_at || job.is_rebuild));
                    if (script.dataset.autoJob === job.job_id && ['complete', 'paused'].includes(job.status) && first) target.searchParams.set('download', first.export_id);
                    window.location.replace(target);
                    return;
                }
            } catch (_) {
                node.querySelector('[data-job-progress]').textContent = 'Could not read export status. Retrying…';
            }
        }
        if (waiting && active) timer = setTimeout(poll, 1000);
    }
    const alreadyReady = jobs.find(node => node.dataset.exportJob === script?.dataset.autoJob && ['complete', 'paused'].includes(node.dataset.jobState));
    if (alreadyReady) alreadyReady.dataset.jobState = 'running';
    poll();
})();
