import { lazy, Suspense } from 'react';
import type { RouteObject } from 'react-router-dom';
import { ProtectedRoute } from '@/components/auth';

// Lazy load page components for code splitting
const Home = lazy(() => import('@/pages/Home'));
const Login = lazy(() => import('@/pages/Login'));
const Signup = lazy(() => import('@/pages/Signup'));
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
const GapAnalysis = lazy(() => import('@/pages/GapAnalysis'));
const Experiments = lazy(() => import('@/pages/Experiments'));
const CausalAnalysis = lazy(() => import('@/pages/CausalAnalysis'));
const ResourceOptimization = lazy(() => import('@/pages/ResourceOptimization'));
const SegmentAnalysis = lazy(() => import('@/pages/SegmentAnalysis'));

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
  {
    path: '/gap-analysis',
    title: 'Gap Analysis',
    description: 'ROI opportunity detection and performance gap prioritization',
    icon: 'target',
    showInNav: true,
  },
  {
    path: '/experiments',
    title: 'Experiments',
    description: 'A/B testing, randomization, and experiment monitoring',
    icon: 'flask',
    showInNav: true,
  },
  {
    path: '/causal-analysis',
    title: 'Causal Analysis',
    description: 'Multi-library causal inference with hierarchical CATE estimation',
    icon: 'git-branch',
    showInNav: true,
  },
  {
    path: '/resource-optimization',
    title: 'Resource Optimization',
    description: 'Mathematical optimization for budget and resource allocation',
    icon: 'calculator',
    showInNav: true,
  },
  {
    path: '/segment-analysis',
    title: 'Segment Analysis',
    description: 'Heterogeneous treatment effects and targeting optimization',
    icon: 'users',
    showInNav: true,
  },
];

// React Router route definitions
export const routes: RouteObject[] = [
  // Auth routes (no layout)
  {
    path: '/login',
    element: (
      <LazyPage>
        <Login />
      </LazyPage>
    ),
  },
  {
    path: '/signup',
    element: (
      <LazyPage>
        <Signup />
      </LazyPage>
    ),
  },
  // Protected app routes
  {
    path: '/',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <Home />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/knowledge-graph',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <KnowledgeGraph />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/causal-discovery',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <CausalDiscovery />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/model-performance',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <ModelPerformance />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/feature-importance',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <FeatureImportance />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/time-series',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <TimeSeries />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/intervention-impact',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <InterventionImpact />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/predictive-analytics',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <PredictiveAnalytics />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/data-quality',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <DataQuality />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/system-health',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <SystemHealth />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/monitoring',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <Monitoring />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/agent-orchestration',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <AgentOrchestration />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/kpi-dictionary',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <KPIDictionary />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/memory-architecture',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <MemoryArchitecture />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/digital-twin',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <DigitalTwin />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/ai-insights',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <AIAgentInsights />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/gap-analysis',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <GapAnalysis />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/experiments',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <Experiments />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/causal-analysis',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <CausalAnalysis />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/resource-optimization',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <ResourceOptimization />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/segment-analysis',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <SegmentAnalysis />
        </LazyPage>
      </ProtectedRoute>
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
