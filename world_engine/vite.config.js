import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/moverse/',
  plugins: [
    vue(),
    tailwindcss(),
  ],
  server: {
    // Proxy API and WebSocket requests to backend in dev mode
    proxy: {
      '/api': {
        target: 'http://localhost:8889',
        changeOrigin: true,
      },
      '/thumbnails': {
        target: 'http://localhost:8889',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8889',
        ws: true,
      },
    },
  },
})
