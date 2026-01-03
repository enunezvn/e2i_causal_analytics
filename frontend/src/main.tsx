import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './lib/query-client'
import { CopilotKitWrapper } from './providers'
import { AppRouter } from './router'
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
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <CopilotKitWrapper runtimeUrl="/api/copilotkit" enabled={true}>
          <AppRouter />
        </CopilotKitWrapper>
      </QueryClientProvider>
    </StrictMode>,
  )
}

// Start the application
initApp()
