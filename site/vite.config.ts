// Vite config for the atlas site.
// - base './' makes every asset URL relative, so the same build works at
//   https://burnycoder.github.io/circuits-and-manifolds-from-mechanistic-interpretability-in-llms-and-other-models-atlas/ and in `vite preview` (docs:
//   https://vite.dev/guide/static-deploy.html#github-pages). Routing uses a HashRouter, so no server rewrites needed.
// - `test` configures Vitest (https://vitest.dev/config/), which validates entry content (schema, citations, quotes).
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    // Plotly is large; it is lazy-loaded per entry, so raise the warning threshold instead of splitting it further.
    chunkSizeWarningLimit: 5000,
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
})
