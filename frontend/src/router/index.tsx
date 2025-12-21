import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import { routes } from './routes';

// Root layout component that wraps all routes
// This will be enhanced later with Header, Sidebar, Footer in phase-2-5
function RootLayout() {
  return (
    <div className="min-h-screen bg-background">
      <Outlet />
    </div>
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
