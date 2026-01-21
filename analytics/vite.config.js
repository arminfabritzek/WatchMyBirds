import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: {
    lib: {
      entry: 'src/main.js',
      name: 'AnalyticsDashboard',
      formats: ['es'],
      fileName: () => 'analytics-dashboard.js'
    },
    outDir: '../assets/analytics',
    emptyOutDir: true,
    cssCodeSplit: false
  }
});
