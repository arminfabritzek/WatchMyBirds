<script>
    import { onMount } from "svelte";

    let loading = true;
    let error = null;
    let series = [];

    onMount(async () => {
        try {
            const res = await fetch("/api/analytics/species-activity");
            if (!res.ok) throw new Error("Failed to load data");
            series = await res.json();
            loading = false;
        } catch (err) {
            error = err.message;
            loading = false;
        }
    });

    function formatTime(decimalHour) {
        if (decimalHour === null || decimalHour === undefined) return "-";
        const h = Math.floor(decimalHour);
        const m = Math.round((decimalHour - h) * 60);
        return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
    }

    function generateSparkline(points) {
        if (!points || points.length < 2) return "";
        const width = 200;
        const height = 30;
        const minX = 0;
        const maxX = 24;

        return points
            .map((p, i) => {
                const x = ((p.x - minX) / (maxX - minX)) * width;
                const y = height - p.y * height;
                return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
            })
            .join(" ");
    }
</script>

<div class="alt-view-container">
    <h4 class="alt-header">Activity Patterns by Species</h4>

    {#if loading}
        <div class="message">Loading...</div>
    {:else if error}
        <div class="message error">{error}</div>
    {:else if series.length === 0}
        <div class="message">No data available</div>
    {:else}
        <div class="table-container">
            <div class="table-header">
                <div class="col-name">Species</div>
                <div class="col-graph">Activity Pattern (24h)</div>
                <div class="col-peak">Peak</div>
            </div>

            <div class="table-body">
                {#each series as item}
                    <div class="table-row">
                        <div class="col-name" title={item.species}>
                            {item.species.replace(/_/g, " ")}
                        </div>
                        <div class="col-graph">
                            <svg
                                class="sparkline"
                                viewBox="0 0 200 30"
                                preserveAspectRatio="none"
                                width="100%"
                                height="30"
                            >
                                <path
                                    d={generateSparkline(item.points)}
                                    fill="none"
                                    stroke="#2ecc71"
                                    stroke-width="1.5"
                                    vector-effect="non-scaling-stroke"
                                />
                                <path
                                    d={`${generateSparkline(item.points)} L 200 30 L 0 30 Z`}
                                    fill="#2ecc71"
                                    fill-opacity="0.1"
                                    stroke="none"
                                />
                            </svg>
                        </div>
                        <div class="col-peak">
                            {formatTime(item.peak_hour)}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
    {/if}
</div>

<style>
    .alt-view-container {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px dashed #e2e8f0;
    }
    .alt-header {
        font-size: 0.85rem;
        font-weight: 700;
        font-weight: 700;
        color: #2ecc71;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
    }
    .message {
        color: #a0aec0;
        font-style: italic;
        padding: 1rem;
        text-align: center;
    }
    .error {
        color: #e53e3e;
    }

    .table-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }

    .table-header {
        display: flex;
        flex-direction: row;
        align-items: center;
        padding: 0.5rem 0.75rem;
        font-weight: 600;
        color: #718096;
        font-size: 0.75rem;
        text-transform: uppercase;
        text-transform: uppercase;
        border-bottom: 2px solid #2ecc71;
        margin-bottom: 0.5rem;
        background-color: #f8fafc; /* visual check */
    }

    .table-row {
        display: flex; /* Mandatory Flexbox */
        flex-direction: row;
        align-items: center;
        justify-content: flex-start;
        padding: 6px 12px;
        margin-bottom: 2px;
        border-radius: 4px;
        background-color: #fcfcfc;
        height: 42px;
    }
    .table-row:hover {
        background-color: #edf2f7;
    }

    .col-name {
        width: 180px;
        flex-shrink: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 500;
        color: #4a5568;
        font-size: 0.9rem;
        font-style: italic;
    }

    .col-graph {
        flex: 1;
        height: 30px;
        margin: 0 1rem;
        position: relative;
        display: flex;
        align-items: center;
    }

    .col-peak {
        width: 60px;
        flex-shrink: 0;
        text-align: right;
        font-family: monospace;
        color: #718096;
        font-size: 0.85rem;
    }

    .sparkline {
        width: 100%;
        height: 30px;
        display: block;
        overflow: visible;
    }
</style>
