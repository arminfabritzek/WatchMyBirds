<script>
    import { onMount, onDestroy, tick } from "svelte";
    import { Chart, registerables } from "chart.js";

    Chart.register(...registerables);

    let canvas;
    let chart;
    let loading = true;
    let error = null;
    let peakHour = null;

    onMount(async () => {
        try {
            const res = await fetch("/api/analytics/time-of-day");
            if (!res.ok) throw new Error("Failed to load data");
            const data = await res.json();

            // Expected data format:
            // { points: [{x: 0.1, y: 0.05}, ...], peak_hour: 14.25, histogram: [{x, y}, ...] }

            const points = data.points;
            const histogram = data.histogram || [];
            peakHour = convertDecimalHour(data.peak_hour);

            loading = false;
            await tick();

            if (canvas) {
                const ctx = canvas.getContext("2d");

                // Beautiful gradient for the area
                const gradient = ctx.createLinearGradient(0, 0, 0, 400);
                gradient.addColorStop(0, "rgba(46, 204, 113, 0.4)");
                gradient.addColorStop(1, "rgba(46, 204, 113, 0.05)");

                chart = new Chart(canvas, {
                    type: "scatter", // Base type, datasets override
                    data: {
                        datasets: [
                            {
                                type: "line",
                                label: "Activity Density",
                                data: points,
                                borderColor: "#27ae60",
                                borderWidth: 3,
                                backgroundColor: gradient,
                                fill: true,
                                pointRadius: 0, // No dots on line
                                tension: 0.4, // Smooth curve
                                xAxisID: "x",
                            },
                            {
                                type: "bar", // Underlying histogram
                                label: "Raw Distribution",
                                data: histogram.map((h) => ({
                                    x: h.x,
                                    y: h.y,
                                })),
                                backgroundColor: "rgba(52, 152, 219, 0.2)",
                                barPercentage: 1.0,
                                categoryPercentage: 1.0,
                                xAxisID: "x",
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        resizeDelay: 0,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                mode: "index",
                                intersect: false,
                                callbacks: {
                                    title: (items) => {
                                        const val = items[0].parsed.x;
                                        return convertDecimalHour(val);
                                    },
                                },
                            },
                        },
                        layout: {
                            padding: { top: 12, right: 16, bottom: 10, left: 10 },
                        },
                        scales: {
                            x: {
                                type: "linear",
                                min: 0,
                                max: 24,
                                grid: { display: false },
                                ticks: {
                                    stepSize: 4, // 0, 4, 8, 12, 16, 20, 24
                                    callback: (val) =>
                                        val === 24 ? "00:00" : val + ":00",
                                    font: { size: 10 },
                                    color: "#718096",
                                    padding: 6,
                                },
                            },
                            y: {
                                display: false, // Hide Y axis as requested
                                beginAtZero: true,
                                grace: "12%",
                            },
                        },
                    },
                });
            }
        } catch (err) {
            error = err.message;
            loading = false;
        }
    });

    function convertDecimalHour(val) {
        if (val === null || val === undefined) return "-";
        const h = Math.floor(val);
        const m = Math.round((val - h) * 60);
        return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
    }

    onDestroy(() => {
        if (chart) chart.destroy();
    });
</script>

<div class="chart-wrapper">
    {#if loading}
        <div class="loading"><div class="spinner-small"></div></div>
    {:else if error}
        <div class="error">{error}</div>
    {:else}
        {#if peakHour}
            <div class="info-badge">Peak Activity: {peakHour}</div>
        {/if}
        <div class="canvas-container">
            <canvas bind:this={canvas}></canvas>
        </div>
    {/if}
</div>

<style>
    .chart-wrapper {
        height: 340px;
        position: relative;
    }

    .info-badge {
        position: absolute;
        top: 0;
        right: 0;
        background: rgba(46, 204, 113, 0.1);
        color: #27ae60;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        z-index: 1;
    }

    .loading,
    .error {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #718096;
    }

    .spinner-small {
        width: 24px;
        height: 24px;
        border: 2px solid #e0e0e0;
        border-top-color: #2ecc71;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    .error {
        color: #e53e3e;
    }

    .canvas-container {
        position: relative;
        height: 100%;
        width: 100%;
    }
</style>
