import { lazy, Suspense } from 'react';
import type { RouteObject } from 'react-router-dom';
import { ProtectedRoute } from '@/components/auth';

// Lazy load page components for code splitting
const Home = lazy(() => import('@/pages/Home'));
const Login = lazy(() => import('@/pages/Login'));
const Signup = lazy(() => import('@/pages/Signup'));
const ForgotPassword = lazy(() => import('@/pages/ForgotPassword'));
const ResetPassword = lazy(() => import('@/pages/ResetPassword'));
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
const ExpertReviews = lazy(() => import('@/pages/ExpertReviews'));
const AIAgentInsights = lazy(() => import('@/pages/AIAgentInsights'));
const GapAnalysis = lazy(() => import('@/pages/GapAnalysis'));
const Experiments = lazy(() => import('@/pages/Experiments'));
const CausalAnalysis = lazy(() => import('@/pages/CausalAnalysis'));
const ResourceOptimization = lazy(() => import('@/pages/ResourceOptimization'));
const SegmentAnalysis = lazy(() => import('@/pages/SegmentAnalysis'));
const AuditChain = lazy(() => import('@/pages/AuditChain'));
const FeedbackLearning = lazy(() => import('@/pages/FeedbackLearning'));
const Analytics = lazy(() => import('@/pages/Analytics'));

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

// Navigation section keys — ordered groups rendered in the sidebar.
export type NavSection = 'main' | 'causal' | 'predictive' | 'decisions' | 'data' | 'system';

// Route configuration with metadata
export interface RouteConfig {
  path: string;
  title: string;
  description: string;
  icon?: string;
  showInNav?: boolean;
  /** Sidebar group this route belongs to. Defaults to 'main' (top, no header). */
  section?: NavSection;
}

export const routeConfigs: RouteConfig[] = [
  // ── Main ────────────────────────────────────────────────────────────────
  {
    path: '/',
    title: 'Home',
    description: 'Dashboard overview and key metrics',
    icon: 'home',
    section: 'main',
    showInNav: true,
  },

  // ── Causal Analytics — understand cause → effect ─────────────────────────
  {
    path: '/causal-discovery',
    title: 'Causal Discovery',
    description: 'Causal analysis and DAG visualization',
    icon: 'git-branch',
    section: 'causal',
    showInNav: true,
  },
  {
    path: '/knowledge-graph',
    title: 'Knowledge Graph',
    description: 'Explore the knowledge graph visualization',
    icon: 'share-2',
    section: 'causal',
    showInNav: true,
  },
  {
    path: '/causal-analysis',
    title: 'Causal Analysis',
    description: 'Multi-library causal inference with hierarchical CATE estimation',
    icon: 'git-branch',
    section: 'causal',
    showInNav: true,
  },
  {
    path: '/intervention-impact',
    title: 'Intervention Impact',
    description: 'Treatment effects and intervention analysis',
    icon: 'target',
    section: 'causal',
    showInNav: true,
  },
  {
    path: '/segment-analysis',
    title: 'Segment Analysis',
    description: 'Heterogeneous treatment effects and targeting optimization',
    icon: 'users',
    section: 'causal',
    showInNav: true,
  },
  {
    path: '/expert-reviews',
    title: 'Expert Reviews',
    description: 'Human-in-the-loop review queue for causal DAGs',
    icon: 'clipboard-check',
    section: 'causal',
    showInNav: true,
  },

  // ── Predictive Modeling — predict & explain ──────────────────────────────
  {
    path: '/predictive-analytics',
    title: 'Predictive Analytics',
    description: 'Risk scores and predictions',
    icon: 'zap',
    section: 'predictive',
    showInNav: true,
  },
  {
    path: '/model-performance',
    title: 'Model Performance',
    description: 'Model metrics and performance analysis',
    icon: 'bar-chart-2',
    section: 'predictive',
    showInNav: true,
  },
  {
    path: '/feature-importance',
    title: 'Feature Importance',
    description: 'SHAP values and feature analysis',
    icon: 'layers',
    section: 'predictive',
    showInNav: true,
  },
  {
    path: '/time-series',
    title: 'Time Series',
    description: 'Time series analysis and forecasting',
    icon: 'trending-up',
    section: 'predictive',
    showInNav: true,
  },
  {
    path: '/digital-twin',
    title: 'Digital Twin',
    description: 'Intervention simulation & pre-screening',
    icon: 'flask-conical',
    section: 'predictive',
    showInNav: true,
  },

  // ── Decisions & Optimization — act on the insight ────────────────────────
  {
    path: '/gap-analysis',
    title: 'Gap Analysis',
    description: 'ROI opportunity detection and performance gap prioritization',
    icon: 'target',
    section: 'decisions',
    showInNav: true,
  },
  {
    path: '/resource-optimization',
    title: 'Resource Optimization',
    description: 'Mathematical optimization for budget and resource allocation',
    icon: 'calculator',
    section: 'decisions',
    showInNav: true,
  },
  {
    path: '/experiments',
    title: 'Experiments',
    description: 'A/B testing, randomization, and experiment monitoring',
    icon: 'flask',
    section: 'decisions',
    showInNav: true,
  },
  {
    path: '/ai-insights',
    title: 'AI Insights',
    description: 'GPT-powered briefs, recommendations & alerts',
    icon: 'brain',
    section: 'decisions',
    showInNav: true,
  },

  // ── Data & Reference ─────────────────────────────────────────────────────
  {
    path: '/kpi-dictionary',
    title: 'KPI Dictionary',
    description: '46 KPIs across 6 workstreams',
    icon: 'book-open',
    section: 'data',
    showInNav: true,
  },
  {
    path: '/data-quality',
    title: 'Data Quality',
    description: 'Data profiling and validation',
    icon: 'check-circle',
    section: 'data',
    showInNav: true,
  },

  // ── System & Platform — internals & ops ──────────────────────────────────
  {
    path: '/system-health',
    title: 'System Health',
    description: 'System monitoring and status',
    icon: 'activity',
    section: 'system',
    showInNav: true,
  },
  {
    path: '/monitoring',
    title: 'Monitoring',
    description: 'Logs, API usage, and error tracking',
    icon: 'monitor',
    section: 'system',
    showInNav: true,
  },
  {
    path: '/analytics',
    title: 'Analytics',
    description: 'Agent performance metrics and query analytics dashboard',
    icon: 'bar-chart',
    section: 'system',
    showInNav: true,
  },
  {
    path: '/agent-orchestration',
    title: 'Agent Orchestration',
    description: '21-agent tiered orchestration system',
    icon: 'bot',
    section: 'system',
    showInNav: true,
  },
  {
    path: '/memory-architecture',
    title: 'Memory Architecture',
    description: 'Tri-memory cognitive system',
    icon: 'brain',
    section: 'system',
    showInNav: true,
  },
  {
    path: '/audit-chain',
    title: 'Audit Chain',
    description: 'Workflow audit trails with cryptographic verification',
    icon: 'shield-check',
    section: 'system',
    showInNav: true,
  },
  {
    path: '/feedback-learning',
    title: 'Feedback Learning',
    description: 'Tier 5 self-improvement with pattern detection and knowledge updates',
    icon: 'sparkles',
    section: 'system',
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
  // Password recovery (public, like /login): /forgot-password is linked from
  // the login page; /reset-password is the AuthProvider resetPassword
  // redirect target reached from the recovery email.
  {
    path: '/forgot-password',
    element: (
      <LazyPage>
        <ForgotPassword />
      </LazyPage>
    ),
  },
  {
    path: '/reset-password',
    element: (
      <LazyPage>
        <ResetPassword />
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
    path: '/expert-reviews',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <ExpertReviews />
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
    path: '/audit-chain',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <AuditChain />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/feedback-learning',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <FeedbackLearning />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
  {
    path: '/analytics',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <Analytics />
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

// Ordered sidebar sections. `label: null` renders without a header (e.g. Home).
const NAV_SECTION_ORDER: { key: NavSection; label: string | null }[] = [
  { key: 'main', label: null },
  { key: 'causal', label: 'Causal Analytics' },
  { key: 'predictive', label: 'Predictive Modeling' },
  { key: 'decisions', label: 'Decisions & Optimization' },
  { key: 'data', label: 'Data & Reference' },
  { key: 'system', label: 'System & Platform' },
];

export interface NavSectionGroup {
  key: NavSection;
  label: string | null;
  routes: RouteConfig[];
}

// Group navigation routes into ordered sidebar sections by explicit
// `route.section` (defaults to 'main'). Membership is semantic, not positional,
// so reordering a route can never silently move it into the wrong section.
export function getNavigationSections(): NavSectionGroup[] {
  const navRoutes = getNavigationRoutes();
  return NAV_SECTION_ORDER.map(({ key, label }) => ({
    key,
    label,
    routes: navRoutes.filter((route) => (route.section ?? 'main') === key),
  })).filter((group) => group.routes.length > 0);
}
