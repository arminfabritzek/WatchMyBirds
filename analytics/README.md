# Analytics UI (Isolated)

This folder contains a standalone Svelte component for the hourly detections chart.
It is isolated from the core UI and can be built later for use under `/analytics/*`.

## Build (optional, later)

```bash
cd analytics
npm install
npm run build
```

Build output is configured to land in `assets/analytics/` as an ES module.

## Embed (optional, later)

```html
<div data-hourly-analytics data-date="2025-01-19"></div>
<script type="module" src="/assets/analytics/hourly-analytics.js"></script>
```

If the build emits a CSS file (e.g. `hourly-analytics.css`), include it as well:

```html
<link rel="stylesheet" href="/assets/analytics/hourly-analytics.css">
```
