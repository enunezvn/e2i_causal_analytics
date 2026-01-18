import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import { routes } from './routes';
import { Layout } from '@/components/layout/Layout';
import { E2ICopilotProvider, CopilotKitWrapper } from '@/providers';
import { env } from '@/config/env';

// Root layout component that wraps all routes
// Uses the Layout component with Header, Sidebar, and Footer
// CopilotKitWrapper MUST be inside the router tree for CopilotChat to access the context
// (RouterProvider creates a separate React context boundary)
function RootLayout() {
  return (
    <CopilotKitWrapper
      runtimeUrl={`${env.apiUrl.replace(/\/$/, '')}/copilotkit/`}
      enabled={env.copilotEnabled}
    >
      <E2ICopilotProvider>
        <Layout>
          <Outlet />
        </Layout>
      </E2ICopilotProvider>
    </CopilotKitWrapper>
  );
}

// Create the browser router with nested routes under the root layout
const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: routes,
  },
]);

// Router component to be used in main.tsx
export function AppRouter() {
  return <RouterProvider router={router} />;
}

// Export router instance for programmatic navigation if needed
export { router };
