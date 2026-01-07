import { lazy, Suspense } from 'react';
import type { RouteObject } from 'react-router-dom';

// Lazy load page components for code splitting
const Home = lazy(() => import('@/pages/Home'));
const KnowledgeGraph = lazy(() => import('@/pages/KnowledgeGraph'));
const CausalDiscovery = lazy(() => import('@/pages/CausalDiscovery'));
const ModelPerformance = lazy(() => import('@/pages/ModelPerformance'));
const FeatureImportance = lazy(() => import('@/pages/FeatureImportance'));
const TimeSeries = lazy(() => import('@/pages/TimeSeries'));
const InterventionImpact = lazy(() => import('@/pages/InterventionImpact'));
const PredictiveAnalytics = lazy(() => import('@/pages/PredictiveAnalytics'));
const DataQuality = lazy(() => import('@/pages/DataQuality'));
const SystemHealth = lazy(() => import('@/pages/SystemHealth'));
const Monitoring = lazy(() => import('@/pages/Monitoring'));
const NotFound = lazy(() => import('@/pages/NotFound'));
const AgentOrchestration = lazy(() => import('@/pages/AgentOrchestration'));
const KPIDictionary = lazy(() => import('@/pages/KPIDictionary'));
const MemoryArchitecture = lazy(() => import('@/pages/MemoryArchitecture'));
const DigitalTwin = lazy(() => import('@/pages/DigitalTwin'));
const AIAgentInsights = lazy(() => import('@/pages/AIAgentInsights'));

// Loading fallback component for lazy-loaded routes
function PageLoadingFallback() {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
    </div>
  );
}

// Wrapper component for lazy-loaded pages
function LazyPage({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<PageLoadingFallback />}>{children}</Suspense>;
}

// Route configuration with metadata
export interface RouteConfig {
  path: string;
  title: string;
  description: string;
  icon?: string;
  showInNav?: boolean;
}

export const routeConfigs: RouteConfig[] = [
  {
    path: '/',
    title: 'Home',
    description: 'Dashboard overview and key metrics',
    icon: 'home',
    showInNav: true,
  },
  {
    path: '/knowledge-graph',
    title: 'Knowledge Graph',
    description: 'Explore the knowledge graph visualization',
    icon: 'share-2',
    showInNav: true,
  },
  {
    path: '/causal-discovery',
    title: 'Causal Discovery',
    description: 'Causal analysis and DAG visualization',
    icon: 'git-branch',
    showInNav: true,
  },
  {
    path: '/model-performance',
    title: 'Model Performance',
    description: 'Model metrics and performance analysis',
    icon: 'bar-chart-2',
    showInNav: true,
  },
  {
    path: '/feature-importance',
    title: 'Feature Importance',
    description: 'SHAP values and feature analysis',
    icon: 'layers',
    showInNav: true,
  },
  {
    path: '/time-series',
    title: 'Time Series',
    description: 'Time series analysis and forecasting',
    icon: 'trending-up',
    showInNav: true,
  },
  {
    path: '/intervention-impact',
    title: 'Intervention Impact',
    description: 'Treatment effects and intervention analysis',
    icon: 'target',
    showInNav: true,
  },
  {
    path: '/predictive-analytics',
    title: 'Predictive Analytics',
    description: 'Risk scores and predictions',
    icon: 'zap',
    showInNav: true,
  },
  {
    path: '/data-quality',
    title: 'Data Quality',
    description: 'Data profiling and validation',
    icon: 'check-circle',
    showInNav: true,
  },
  {
    path: '/system-health',
    title: 'System Health',
    description: 'System monitoring and status',
    icon: 'activity',
    showInNav: true,
  },
  {
    path: '/monitoring',
    title: 'Monitoring',
    description: 'Logs, API usage, and error tracking',
    icon: 'monitor',
    showInNav: true,
  },
  {
    path: '/agent-orchestration',
    title: 'Agent Orchestration',
    description: '18-agent tiered orchestration system',
    icon: 'bot',
    showInNav: true,
  },
  {
    path: '/kpi-dictionary',
    title: 'KPI Dictionary',
    description: '46 KPIs across 6 workstreams',
    icon: 'book-open',
    showInNav: true,
  },
  {
    path: '/memory-architecture',
    title: 'Memory Architecture',
    description: 'Tri-memory cognitive system',
    icon: 'brain',
    showInNav: true,
  },
  {
    path: '/digital-twin',
    title: 'Digital Twin',
    description: 'Intervention simulation & pre-screening',
    icon: 'flask-conical',
    showInNav: true,
  },
  {
    path: '/ai-insights',
    title: 'AI Insights',
    description: 'GPT-powered briefs, recommendations & alerts',
    icon: 'brain',
    showInNav: true,
  },
];

// React Router route definitions
export const routes: RouteObject[] = [
  {
    path: '/',
    element: (
      <LazyPage>
        <Home />
      </LazyPage>
    ),
  },
  {
    path: '/knowledge-graph',
    element: (
      <LazyPage>
        <KnowledgeGraph />
      </LazyPage>
    ),
  },
  {
    path: '/causal-discovery',
    element: (
      <LazyPage>
        <CausalDiscovery />
      </LazyPage>
    ),
  },
  {
    path: '/model-performance',
    element: (
      <LazyPage>
        <ModelPerformance />
      </LazyPage>
    ),
  },
  {
    path: '/feature-importance',
    element: (
      <LazyPage>
        <FeatureImportance />
      </LazyPage>
    ),
  },
  {
    path: '/time-series',
    element: (
      <LazyPage>
        <TimeSeries />
      </LazyPage>
    ),
  },
  {
    path: '/intervention-impact',
    element: (
      <LazyPage>
        <InterventionImpact />
      </LazyPage>
    ),
  },
  {
    path: '/predictive-analytics',
    element: (
      <LazyPage>
        <PredictiveAnalytics />
      </LazyPage>
    ),
  },
  {
    path: '/data-quality',
    element: (
      <LazyPage>
        <DataQuality />
      </LazyPage>
    ),
  },
  {
    path: '/system-health',
    element: (
      <LazyPage>
        <SystemHealth />
      </LazyPage>
    ),
  },
  {
    path: '/monitoring',
    element: (
      <LazyPage>
        <Monitoring />
      </LazyPage>
    ),
  },
  {
    path: '/agent-orchestration',
    element: (
      <LazyPage>
        <AgentOrchestration />
      </LazyPage>
    ),
  },
  {
    path: '/kpi-dictionary',
    element: (
      <LazyPage>
        <KPIDictionary />
      </LazyPage>
    ),
  },
  {
    path: '/memory-architecture',
    element: (
      <LazyPage>
        <MemoryArchitecture />
      </LazyPage>
    ),
  },
  {
    path: '/digital-twin',
    element: (
      <LazyPage>
        <DigitalTwin />
      </LazyPage>
    ),
  },
  {
    path: '/ai-insights',
    element: (
      <LazyPage>
        <AIAgentInsights />
      </LazyPage>
    ),
  },
  {
    path: '*',
    element: (
      <LazyPage>
        <NotFound />
      </LazyPage>
    ),
  },
];

// Helper function to get route config by path
export function getRouteConfig(path: string): RouteConfig | undefined {
  return routeConfigs.find((config) => config.path === path);
}

// Helper function to get navigation routes
export function getNavigationRoutes(): RouteConfig[] {
  return routeConfigs.filter((config) => config.showInNav);
}
