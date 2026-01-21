<script>
  import { onMount } from "svelte";
  import SummaryCards from "./SummaryCards.svelte";

  import TimeOfDay from "./TimeOfDay.svelte";
  import SpeciesActivityTable from "./SpeciesActivityTable.svelte";

  let summary = null;
  let loading = true;
  let error = null;

  onMount(async () => {
    try {
      const res = await fetch("/api/analytics/summary");
      if (!res.ok) throw new Error("Failed to load summary");
      summary = await res.json();
    } catch (err) {
      error = err.message;
    } finally {
      loading = false;
    }
  });
</script>

<div class="analytics-dashboard">
  <header class="dashboard-header">
    <h1>Analytics</h1>
    <p class="subtitle">All-Time Detection Statistics</p>
  </header>

  {#if loading}
    <div class="loading-container">
      <div class="spinner"></div>
      <p>Loading analytics...</p>
    </div>
  {:else if error}
    <div class="error-container">
      <p>{error}</p>
    </div>
  {:else}
    <SummaryCards {summary} />

    <div class="charts-section">
      <div class="chart-card chart-card--full">
        <h3 class="chart-title">Activity by Time of Day</h3>
        <TimeOfDay />
      </div>

      <div class="chart-card chart-card--full">
        <SpeciesActivityTable />
      </div>
    </div>
  {/if}
</div>

<style>
  .analytics-dashboard {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      sans-serif;
    background: linear-gradient(135deg, #f8faf9 0%, #eef5f0 100%);
    min-height: 100vh;
  }

  .dashboard-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    border-radius: 16px;
    color: white;
    box-shadow: 0 4px 20px rgba(46, 204, 113, 0.3);
  }

  .dashboard-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
  }

  .subtitle {
    margin: 0.5rem 0 0;
    opacity: 0.9;
    font-size: 1rem;
    font-weight: 400;
  }

  .loading-container,
  .error-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    background: white;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #e0e0e0;
    border-top-color: #2ecc71;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-container {
    color: #e53e3e;
  }

  .charts-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-top: 1.5rem;
  }

  .chart-card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
    padding: 1.5rem;
    transition:
      box-shadow 0.2s ease,
      transform 0.2s ease;
  }

  .chart-card:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
  }

  .chart-card--full {
    grid-column: 1 / -1;
  }

  .chart-title {
    margin: 0 0 1rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #2d3748;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #f0f0f0;
  }

  @media (max-width: 768px) {
    .analytics-dashboard {
      padding: 1rem;
    }

    .dashboard-header {
      padding: 1.25rem;
      border-radius: 12px;
    }

    .dashboard-header h1 {
      font-size: 1.5rem;
    }

    .chart-card {
      padding: 1rem;
      border-radius: 12px;
    }

    .chart-title {
      font-size: 0.9rem;
    }
  }
</style>
