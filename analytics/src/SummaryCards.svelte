<script>
    export let summary;

    function formatNumber(n) {
        if (n >= 1000) return (n / 1000).toFixed(1) + "k";
        return n?.toString() || "0";
    }
</script>

<div
    style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;"
>
    <!-- Total Detections -->
    <div
        style="background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #4caf50;"
    >
        <div style="font-size: 2rem; font-weight: 700; color: #1a202c;">
            {formatNumber(summary?.total_detections)}
        </div>
        <div style="font-size: 0.85rem; color: #718096; margin-top: 0.25rem;">
            Total Detections
        </div>
    </div>

    <!-- Species -->
    <div
        style="background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #0288d1;"
    >
        <div style="font-size: 2rem; font-weight: 700; color: #1a202c;">
            {summary?.total_species || 0}
        </div>
        <div style="font-size: 0.85rem; color: #718096; margin-top: 0.25rem;">
            Species Detected
        </div>
    </div>

    <!-- First Detection -->
    <div
        style="background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #00897b;"
    >
        <div style="font-size: 1.1rem; font-weight: 700; color: #1a202c;">
            {summary?.date_range?.first || "—"}
        </div>
        <div style="font-size: 0.85rem; color: #718096; margin-top: 0.25rem;">
            First Detection
        </div>
    </div>

    <!-- Last Detection -->
    <div
        style="background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #ff9800;"
    >
        <div style="font-size: 1.1rem; font-weight: 700; color: #1a202c;">
            {summary?.date_range?.last || "—"}
        </div>
        <div style="font-size: 0.85rem; color: #718096; margin-top: 0.25rem;">
            Last Detection
        </div>
    </div>
</div>

<style>
    @media (max-width: 992px) {
        div[style*="grid-template-columns: repeat(4"] {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }

    @media (max-width: 480px) {
        div[style*="grid-template-columns: repeat(4"] {
            grid-template-columns: 1fr !important;
        }
    }
</style>
