import { Link } from 'react-router-dom';
import { getNavigationRoutes } from '@/router/routes';

function Home() {
  const navRoutes = getNavigationRoutes().filter((route) => route.path !== '/');

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-2">E2I Causal Analytics</h1>
      <p className="text-muted-foreground mb-8">
        Welcome to the E2I Causal Analytics dashboard. Select a section below to get started.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {navRoutes.map((route) => (
          <Link
            key={route.path}
            to={route.path}
            className="block p-6 bg-card border border-border rounded-lg hover:border-primary transition-colors"
          >
            <h2 className="text-xl font-semibold mb-2">{route.title}</h2>
            <p className="text-muted-foreground text-sm">{route.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

export default Home;
