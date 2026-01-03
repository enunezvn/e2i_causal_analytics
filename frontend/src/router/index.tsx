import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import { routes } from './routes';
import { Layout } from '@/components/layout/Layout';
import { E2ICopilotProvider } from '@/providers';

// Root layout component that wraps all routes
// Uses the Layout component with Header, Sidebar, and Footer
// Wrapped with E2ICopilotProvider for AI readables/actions (requires router context)
function RootLayout() {
  return (
    <E2ICopilotProvider>
      <Layout>
        <Outlet />
      </Layout>
    </E2ICopilotProvider>
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
