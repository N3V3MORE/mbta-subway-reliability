import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Relative asset and data paths, so the build can be hosted under any folder.
  base: './',
  build: {
    chunkSizeWarningLimit: 1_100,
  },
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
  plugins: [react()],
})
