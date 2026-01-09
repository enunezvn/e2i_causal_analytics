/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: '@', replacement: path.resolve(__dirname, './src') },
      { find: /\.css$/, replacement: path.resolve(__dirname, './src/test/css-stub.ts') },
    ],
  },
  css: {
    // Ignore CSS files in tests
    modules: {
      scopeBehaviour: 'global',
    },
  },
  test: {
    css: true,
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    // Provide mock env vars for tests (actual calls are intercepted by MSW)
    env: {
      VITE_SUPABASE_URL: 'https://test-project.supabase.co',
      VITE_SUPABASE_ANON_KEY: 'test-anon-key-for-vitest',
      VITE_API_URL: 'http://localhost:8000',
    },
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', 'dist', 'e2e'],
    deps: {
      optimizer: {
        web: {
          include: ['katex', '@copilotkit/react-ui'],
        },
      },
    },
    server: {
      deps: {
        inline: ['katex', '@copilotkit/react-ui', '@copilotkit/react-core'],
      },
    },
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules',
        'dist',
        'src/test',
        '**/*.d.ts',
        '**/*.config.{ts,js}',
        '**/types/**',
      ],
    },
  },
})
