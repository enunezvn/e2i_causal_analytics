import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')

  // Development proxy target - forwards /api/* to the backend server
  // Default localhost:8000 for local development; override with VITE_API_URL if needed
  const apiTarget = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        // Polyfill Node.js util module for CopilotKit's power-assert dependency
        util: 'util/',
      },
    },
    define: {
      // Provide global for Node.js modules
      global: 'globalThis',
    },
    optimizeDeps: {
      // Force re-optimization when dependencies change
      force: true,
    },
    server: {
      port: 5174,
      proxy: {
        // Proxy API requests to FastAPI backend
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          secure: false,
          // Auto-rewrite Location headers in redirect responses to use proxy URL
          autoRewrite: true,
          // Configure proxy to handle redirects properly
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.log('[vite proxy] error:', err);
            });
            proxy.on('proxyReq', (_proxyReq, req, _res) => {
              console.log('[vite proxy] sending:', req.method, req.url, '→', apiTarget);
            });
            // Rewrite redirect Location headers to avoid CORS issues
            proxy.on('proxyRes', (proxyRes, _req, _res) => {
              const location = proxyRes.headers['location'];
              if (location && location.startsWith(apiTarget)) {
                proxyRes.headers['location'] = location.replace(apiTarget, '');
                console.log('[vite proxy] rewrote redirect:', location, '→', proxyRes.headers['location']);
              }
            });
          },
        },
      },
    },
  }
})
