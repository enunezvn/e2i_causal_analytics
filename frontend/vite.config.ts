import { defineConfig, loadEnv, type Plugin, type ResolvedConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'node:fs'

/**
 * Drop the MSW service worker from production builds.
 *
 * public/mockServiceWorker.js is a dev-only artifact (required for `npm run
 * dev` API mocking) but Vite copies public/ verbatim into dist/, so without
 * this plugin every production build ships the mock service worker (the
 * dev-flavored container bundle audited 2026-06 shipped AND invoked it).
 *
 * Gated on config.isProduction (effective NODE_ENV at build time) so an
 * intentional dev-flavored build (NODE_ENV=development vite build), whose
 * bundle actually invokes MSW, keeps its worker.
 */
function dropMswWorkerInProd(): Plugin {
  let config: ResolvedConfig
  return {
    name: 'e2i:drop-msw-worker-in-prod',
    apply: 'build',
    configResolved(resolved) {
      config = resolved
    },
    closeBundle() {
      if (!config.isProduction) return
      const workerPath = path.resolve(config.root, config.build.outDir, 'mockServiceWorker.js')
      if (fs.existsSync(workerPath)) {
        fs.rmSync(workerPath)
        config.logger.info('[e2i] removed dev-only mockServiceWorker.js from production bundle')
      }
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')

  // Development proxy target - forwards /api/* to the backend server
  // Default localhost:8000 for local development; override with VITE_API_URL if needed
  const apiTarget = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react(), dropMswWorkerInProd()],
    build: {
      rollupOptions: {
        output: {
          /**
           * Split the heavy shared vendors out of the single ~3MB `index`
           * chunk. Routes are already React.lazy()-loaded (src/router/routes.tsx),
           * but vendors imported by 2+ lazy routes (recharts in 17 pages,
           * @copilotkit in the chat shell, etc.) get hoisted into one common
           * `index` chunk. manualChunks forces each vendor family into its own
           * cacheable chunk regardless of the route graph.
           *
           * recharts + d3 MUST stay in the SAME chunk: recharts re-exports
           * cartesian components (Bar, etc.) through its barrel while those
           * modules also depend back on it. Splitting them across chunks
           * produces the circular-dependency-between-chunks warning Rollup
           * emits and risks broken execution order, so they share `vendor-charts`.
           */
          manualChunks(id) {
            if (!id.includes('node_modules')) return undefined

            // React core: shared by every route, keep it isolated + tiny.
            if (
              /[\\/]node_modules[\\/](react|react-dom|react-router|react-router-dom|scheduler)[\\/]/.test(
                id,
              )
            ) {
              return 'vendor-react'
            }
            // Charts: recharts + d3 + victory-vendor (recharts' d3 shim) together
            // to avoid the cross-chunk circular dependency warned about above.
            if (
              /[\\/]node_modules[\\/](recharts|d3-[^\\/]+|d3|victory-vendor|internmap|delaunator|robust-predicates)[\\/]/.test(
                id,
              )
            ) {
              return 'vendor-charts'
            }
            // CopilotKit chat SDK (largest single dependency family on disk).
            if (/[\\/]node_modules[\\/]@copilotkit[\\/]/.test(id)) {
              return 'vendor-copilotkit'
            }
            // Supabase client + its gotrue/realtime/postgrest sub-packages.
            if (/[\\/]node_modules[\\/]@supabase[\\/]/.test(id)) {
              return 'vendor-supabase'
            }
            // TanStack Query.
            if (/[\\/]node_modules[\\/]@tanstack[\\/]/.test(id)) {
              return 'vendor-query'
            }
            // Animation library.
            if (/[\\/]node_modules[\\/]framer-motion[\\/]/.test(id)) {
              return 'vendor-motion'
            }
            // Radix UI primitives (many small packages -> one shared chunk).
            if (/[\\/]node_modules[\\/]@radix-ui[\\/]/.test(id)) {
              return 'vendor-radix'
            }
            // Form stack.
            if (
              /[\\/]node_modules[\\/](react-hook-form|@hookform|zod)[\\/]/.test(id)
            ) {
              return 'vendor-forms'
            }
            return undefined
          },
        },
      },
    },
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
