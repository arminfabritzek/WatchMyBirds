import AnalyticsDashboard from './AnalyticsDashboard.svelte';

const target = document.querySelector('[data-analytics-dashboard]');

if (target) {
  new AnalyticsDashboard({
    target,
  });
}
