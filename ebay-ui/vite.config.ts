import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: true, // Allow all hosts for tunnel access
    proxy: {
      '/seller': {
        target: 'http://localhost:8050',
        changeOrigin: true,
      },
      '/agent': {
        target: 'http://localhost:8050',
        changeOrigin: true,
      },
      '/search': {
        target: 'http://localhost:8050',
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://localhost:8050',
        changeOrigin: true,
      },
    },
  },
})