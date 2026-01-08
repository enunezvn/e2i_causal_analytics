import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './lib/query-client'
import { AuthProvider, CopilotKitWrapper } from './providers'
import { AppRouter } from './router'
import { env } from './config/env'
import './index.css'

/**
 * Initialize and render the application
 *
 * In development mode, MSW is initialized first to intercept API requests.
 * The app only renders after MSW is ready to ensure all requests are mocked.
 */
async function initApp() {
  // Initialize MSW in development mode
  if (import.meta.env.DEV) {
    const { initMSW } = await import('./mocks/browser')
    await initMSW()
  }

  // Render the React application
  // Provider order (outer to inner):
  // 1. QueryClientProvider - React Query for data fetching
  // 2. AuthProvider - Authentication state (needs QueryClient for cache invalidation)
  // 3. CopilotKitWrapper - AI assistant (disabled by default in dev)
  // 4. AppRouter - React Router
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <CopilotKitWrapper runtimeUrl="/api/copilotkit/" enabled={env.copilotEnabled}>
            <AppRouter />
          </CopilotKitWrapper>
        </AuthProvider>
      </QueryClientProvider>
    </StrictMode>,
  )
}

// Start the application
initApp()
