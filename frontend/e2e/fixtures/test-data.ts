/**
 * Test data constants for E2E tests.
 * Provides consistent test data across all tests.
 */

// ============================================================================
// Brand Data
// ============================================================================

export const BRANDS = {
  REMIBRUTINIB: 'Remibrutinib',
  FABHALTA: 'Fabhalta',
  KISQALI: 'Kisqali',
  ALL: 'All',
} as const

export type Brand = typeof BRANDS[keyof typeof BRANDS]

// ============================================================================
// Date Ranges
// ============================================================================

export const DATE_RANGES = {
  LAST_7_DAYS: {
    start: getDateOffset(-7),
    end: getDateOffset(0),
    label: 'Last 7 Days',
  },
  LAST_30_DAYS: {
    start: getDateOffset(-30),
    end: getDateOffset(0),
    label: 'Last 30 Days',
  },
  LAST_90_DAYS: {
    start: getDateOffset(-90),
    end: getDateOffset(0),
    label: 'Last 90 Days',
  },
  YEAR_TO_DATE: {
    start: getYearStart(),
    end: getDateOffset(0),
    label: 'Year to Date',
  },
} as const

function getDateOffset(days: number): string {
  const date = new Date()
  date.setDate(date.getDate() + days)
  return date.toISOString().split('T')[0]
}

function getYearStart(): string {
  const now = new Date()
  return `${now.getFullYear()}-01-01`
}

// ============================================================================
// KPI Test Data
// ============================================================================

export const KPI_NAMES = [
  'TRx Volume',
  'NRx Volume',
  'Market Share',
  'Conversion Rate',
  'HCP Reach',
  'Patient Starts',
  'NBRx',
  'TRx Share',
  'Refill Rate',
  'Days of Therapy',
] as const

export const KPI_CATEGORIES = {
  VOLUME: ['TRx Volume', 'NRx Volume', 'NBRx'],
  SHARE: ['Market Share', 'TRx Share'],
  ENGAGEMENT: ['HCP Reach', 'Conversion Rate'],
  PATIENT: ['Patient Starts', 'Refill Rate', 'Days of Therapy'],
} as const

// ============================================================================
// Agent Test Data
// ============================================================================

export const AGENT_TIERS = {
  TIER_0: { number: 0, name: 'Foundation', agents: ['scope_definer', 'data_preparer', 'feature_analyzer', 'model_selector', 'model_trainer', 'model_deployer', 'observability_connector'] },
  TIER_1: { number: 1, name: 'Orchestration', agents: ['orchestrator', 'tool_composer'] },
  TIER_2: { number: 2, name: 'Causal', agents: ['causal_impact', 'gap_analyzer', 'heterogeneous_optimizer'] },
  TIER_3: { number: 3, name: 'Monitoring', agents: ['drift_monitor', 'experiment_designer', 'health_score'] },
  TIER_4: { number: 4, name: 'ML', agents: ['prediction_synthesizer', 'resource_optimizer'] },
  TIER_5: { number: 5, name: 'Learning', agents: ['explainer', 'feedback_learner'] },
} as const

// ============================================================================
// Page Routes
// ============================================================================

export const ROUTES = {
  HOME: '/',
  CAUSAL_DISCOVERY: '/causal-discovery',
  KNOWLEDGE_GRAPH: '/knowledge-graph',
  MODEL_PERFORMANCE: '/model-performance',
  FEATURE_IMPORTANCE: '/feature-importance',
  TIME_SERIES: '/time-series',
  INTERVENTION_IMPACT: '/intervention-impact',
  PREDICTIVE_ANALYTICS: '/predictive-analytics',
  DATA_QUALITY: '/data-quality',
  SYSTEM_HEALTH: '/system-health',
  MONITORING: '/monitoring',
  AGENT_ORCHESTRATION: '/agent-orchestration',
  KPI_DICTIONARY: '/kpi-dictionary',
  MEMORY_ARCHITECTURE: '/memory-architecture',
  DIGITAL_TWIN: '/digital-twin',
} as const

export type Route = typeof ROUTES[keyof typeof ROUTES]

// ============================================================================
// Test Users
// ============================================================================

export const TEST_USERS = {
  DEFAULT: {
    email: 'test@e2i.com',
    name: 'Test User',
    role: 'analyst',
  },
  ADMIN: {
    email: 'admin@e2i.com',
    name: 'Admin User',
    role: 'admin',
  },
} as const

// ============================================================================
// Expected UI Elements
// ============================================================================

export const UI_ELEMENTS = {
  SIDEBAR_LINKS: [
    'Home',
    'Causal Discovery',
    'Knowledge Graph',
    'Model Performance',
    'Feature Importance',
    'Time Series',
    'Predictive Analytics',
    'Data Quality',
    'System Health',
  ],
  HEADER_ELEMENTS: [
    'Brand Selector',
    'Date Range',
    'User Menu',
  ],
  HOME_SECTIONS: [
    'KPI Overview',
    'System Health',
    'Quick Actions',
    'Recent Activity',
  ],
} as const

// ============================================================================
// Timeouts
// ============================================================================

export const TIMEOUTS = {
  SHORT: 5000,
  MEDIUM: 10000,
  LONG: 30000,
  API_RESPONSE: 15000,
  PAGE_LOAD: 20000,
} as const
