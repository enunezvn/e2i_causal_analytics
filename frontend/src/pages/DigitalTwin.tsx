/**
 * Digital Twin Page
 * =================
 *
 * E2I Digital Twin simulation interface for intervention pre-screening.
 * Allows running simulations, comparing scenarios, and viewing results.
 *
 * Features:
 * - Simulation configuration panel
 * - Results visualization with confidence intervals
 * - Recommendation display
 * - Fidelity metrics
 * - Simulation history
 *
 * @module pages/DigitalTwin
 */

import { useState } from 'react';
import {
  FlaskConical,
  Play,
  History,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  BarChart3,
  Settings2,
  Gauge,
} from 'lucide-react';
import {
  useDigitalTwinHealth,
  useSimulationHistory,
  useRunSimulation,
} from '@/hooks/api/use-digital-twin';
import {
  InterventionType,
  RecommendationType,
  ConfidenceLevel,
  type LegacySimulationResponse,
  type SimulationConfidenceInterval,
} from '@/types/digital-twin';

// =============================================================================
// TYPES
// =============================================================================

interface StatCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_SIMULATION: LegacySimulationResponse = {
  simulation_id: 'sim-001',
  created_at: '2026-01-04T10:00:00Z',
  request: {
    intervention_type: InterventionType.HCP_ENGAGEMENT,
    brand: 'Remibrutinib',
    sample_size: 1000,
    duration_days: 90,
    target_regions: ['Northeast', 'Midwest'],
    budget: 500000,
  },
  outcomes: {
    ate: { lower: 0.12, estimate: 0.18, upper: 0.24 },
    trx_lift: { lower: 45, estimate: 72, upper: 99 },
    nrx_lift: { lower: 22, estimate: 35, upper: 48 },
    market_share_change: { lower: 0.5, estimate: 0.8, upper: 1.1 },
    roi: { lower: 2.1, estimate: 3.2, upper: 4.3 },
    nnt: 14,
  },
  fidelity: {
    overall_score: 0.87,
    data_coverage: 0.92,
    calibration: 0.85,
    temporal_alignment: 0.88,
    feature_completeness: 0.83,
    confidence_level: ConfidenceLevel.HIGH,
  },
  sensitivity: [
    { parameter: 'sample_size', base_value: 1000, low_value: 500, high_value: 2000, ate_at_low: 0.15, ate_at_high: 0.20, sensitivity_score: 0.65 },
    { parameter: 'duration_days', base_value: 90, low_value: 60, high_value: 120, ate_at_low: 0.14, ate_at_high: 0.21, sensitivity_score: 0.72 },
    { parameter: 'budget', base_value: 500000, low_value: 250000, high_value: 750000, ate_at_low: 0.12, ate_at_high: 0.22, sensitivity_score: 0.85 },
  ],
  recommendation: {
    type: RecommendationType.DEPLOY,
    confidence: ConfidenceLevel.HIGH,
    rationale: 'Simulation indicates strong positive ATE with acceptable uncertainty bounds. ROI projection exceeds threshold.',
    evidence: [
      'ATE 95% CI does not include zero',
      'ROI lower bound exceeds 2.0x',
      'Model fidelity score above 0.85',
    ],
    risk_factors: [
      'Budget sensitivity is high - consider staged rollout',
      'Regional variation in effect size observed',
    ],
    expected_value: 1600000,
  },
  projections: [
    { date: '2026-02-01', with_intervention: 150, without_intervention: 140, lower_bound: 145, upper_bound: 155 },
    { date: '2026-03-01', with_intervention: 165, without_intervention: 142, lower_bound: 158, upper_bound: 172 },
    { date: '2026-04-01', with_intervention: 182, without_intervention: 145, lower_bound: 172, upper_bound: 192 },
  ],
  execution_time_ms: 2350,
};

const SAMPLE_HISTORY = [
  { simulation_id: 'sim-001', created_at: '2026-01-04T10:00:00Z', intervention_type: InterventionType.HCP_ENGAGEMENT, brand: 'Remibrutinib', ate_estimate: 0.18, recommendation_type: RecommendationType.DEPLOY },
  { simulation_id: 'sim-002', created_at: '2026-01-03T14:30:00Z', intervention_type: InterventionType.DIGITAL_MARKETING, brand: 'Fabhalta', ate_estimate: 0.09, recommendation_type: RecommendationType.REFINE },
  { simulation_id: 'sim-003', created_at: '2026-01-02T09:15:00Z', intervention_type: InterventionType.PATIENT_SUPPORT, brand: 'Kisqali', ate_estimate: 0.05, recommendation_type: RecommendationType.SKIP },
  { simulation_id: 'sim-004', created_at: '2026-01-01T11:45:00Z', intervention_type: InterventionType.REP_TRAINING, brand: 'Remibrutinib', ate_estimate: 0.22, recommendation_type: RecommendationType.DEPLOY },
];

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    healthy: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    degraded: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    unknown: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${styles[status] || styles.unknown}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function RecommendationBadge({ type }: { type: RecommendationType }) {
  const config = {
    [RecommendationType.DEPLOY]: { icon: CheckCircle, className: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' },
    [RecommendationType.SKIP]: { icon: XCircle, className: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' },
    [RecommendationType.REFINE]: { icon: Settings2, className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' },
    [RecommendationType.ANALYZE]: { icon: BarChart3, className: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400' },
  };

  const { icon: Icon, className } = config[type];

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${className}`}>
      <Icon className="h-4 w-4" />
      {type.charAt(0).toUpperCase() + type.slice(1)}
    </span>
  );
}

function ConfidenceIntervalDisplay({ ci, label, unit = '' }: { ci: SimulationConfidenceInterval; label: string; unit?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-sm text-[var(--color-text-secondary)]">{label}</span>
      <span className="text-xl font-bold text-[var(--color-text-primary)]">
        {ci.estimate.toFixed(2)}{unit}
      </span>
      <span className="text-xs text-[var(--color-text-tertiary)]">
        95% CI: [{ci.lower.toFixed(2)}, {ci.upper.toFixed(2)}]
      </span>
    </div>
  );
}

function StatCard({ title, value, subtext, icon, trend }: StatCardProps) {
  const trendColors = {
    up: 'text-green-600 dark:text-green-400',
    down: 'text-red-600 dark:text-red-400',
    neutral: 'text-gray-600 dark:text-gray-400',
  };

  return (
    <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-[var(--color-text-secondary)]">{title}</span>
        <div className="p-1.5 rounded bg-[var(--color-primary)]/10 text-[var(--color-primary)]">
          {icon}
        </div>
      </div>
      <div className={`text-2xl font-bold ${trend ? trendColors[trend] : 'text-[var(--color-text-primary)]'}`}>
        {value}
      </div>
      {subtext && <p className="text-xs text-[var(--color-text-tertiary)] mt-1">{subtext}</p>}
    </div>
  );
}

function FidelityGauge({ score, label }: { score: number; label: string }) {
  const percentage = score * 100;
  const color = percentage >= 80 ? 'bg-green-500' : percentage >= 60 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[var(--color-text-secondary)]">{label}</span>
        <span className="text-xs font-medium text-[var(--color-text-primary)]">{percentage.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-[var(--color-border)] rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

function SimulationForm({
  onSubmit,
  isLoading,
}: {
  onSubmit: (data: { interventionType: InterventionType; brand: string; sampleSize: number; durationDays: number }) => void;
  isLoading: boolean;
}) {
  const [interventionType, setInterventionType] = useState<InterventionType>(InterventionType.HCP_ENGAGEMENT);
  const [brand, setBrand] = useState('Remibrutinib');
  const [sampleSize, setSampleSize] = useState(1000);
  const [durationDays, setDurationDays] = useState(90);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ interventionType, brand, sampleSize, durationDays });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
          Intervention Type
        </label>
        <select
          value={interventionType}
          onChange={(e) => setInterventionType(e.target.value as InterventionType)}
          className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
        >
          <option value={InterventionType.HCP_ENGAGEMENT}>HCP Engagement</option>
          <option value={InterventionType.PATIENT_SUPPORT}>Patient Support</option>
          <option value={InterventionType.DIGITAL_MARKETING}>Digital Marketing</option>
          <option value={InterventionType.REP_TRAINING}>Rep Training</option>
          <option value={InterventionType.PRICING}>Pricing</option>
          <option value={InterventionType.FORMULARY_ACCESS}>Formulary Access</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
          Brand
        </label>
        <select
          value={brand}
          onChange={(e) => setBrand(e.target.value)}
          className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
        >
          <option value="Remibrutinib">Remibrutinib</option>
          <option value="Fabhalta">Fabhalta</option>
          <option value="Kisqali">Kisqali</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
            Sample Size
          </label>
          <input
            type="number"
            value={sampleSize}
            onChange={(e) => setSampleSize(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
            min={100}
            max={10000}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1">
            Duration (days)
          </label>
          <input
            type="number"
            value={durationDays}
            onChange={(e) => setDurationDays(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg text-[var(--color-text-primary)]"
            min={30}
            max={365}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-hover)] transition-colors disabled:opacity-50"
      >
        {isLoading ? (
          <RefreshCw className="h-4 w-4 animate-spin" />
        ) : (
          <Play className="h-4 w-4" />
        )}
        Run Simulation
      </button>
    </form>
  );
}

// =============================================================================
// MAIN PAGE
// =============================================================================

export default function DigitalTwin() {
  const [selectedSimulation, setSelectedSimulation] = useState<LegacySimulationResponse | null>(SAMPLE_SIMULATION);
  const [activeTab, setActiveTab] = useState<'results' | 'history'>('results');

  const { data: healthData, isLoading: _healthLoading } = useDigitalTwinHealth();
  const { data: historyData, refetch: refetchHistory } = useSimulationHistory({ limit: 10 });
  const { mutate: runSim, isPending: isRunning } = useRunSimulation({
    onSuccess: () => {
      // Refetch history to show new simulation, switch to history tab
      refetchHistory();
      setActiveTab('history');
    },
  });

  const health = healthData ?? { status: 'unknown', service: 'digital-twin', models_available: 0, simulations_pending: 0, last_simulation_at: undefined };
  const history = historyData?.simulations || SAMPLE_HISTORY;
  const simulation = selectedSimulation;

  const handleRunSimulation = (formData: { interventionType: InterventionType; brand: string; sampleSize: number; durationDays: number }) => {
    runSim({
      intervention: {
        intervention_type: formData.interventionType,
        duration_weeks: Math.ceil(formData.durationDays / 7),
      },
      brand: formData.brand,
      twin_count: formData.sampleSize,
    });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-primary)] flex items-center gap-3">
            <FlaskConical className="h-7 w-7 text-[var(--color-primary)]" />
            Digital Twin
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            Intervention pre-screening and scenario analysis
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={health.status} />
          <span className="text-xs text-[var(--color-text-tertiary)]">
            {health.models_available} model{health.models_available !== 1 ? 's' : ''} available
          </span>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Simulations Today"
          value={history.length}
          subtext="Across all brands"
          icon={<FlaskConical className="h-4 w-4" />}
        />
        <StatCard
          title="Avg. Execution Time"
          value="2.4s"
          subtext="Last 24 hours"
          icon={<Gauge className="h-4 w-4" />}
        />
        <StatCard
          title="Deploy Rate"
          value="68%"
          subtext="Recommendations"
          icon={<CheckCircle className="h-4 w-4" />}
          trend="up"
        />
        <StatCard
          title="Model Fidelity"
          value="87%"
          subtext="Overall score"
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Simulation Form */}
        <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
            <Settings2 className="h-5 w-5 text-[var(--color-primary)]" />
            Configure Simulation
          </h3>
          <SimulationForm onSubmit={handleRunSimulation} isLoading={isRunning} />
        </div>

        {/* Results / History Panel */}
        <div className="lg:col-span-2 bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
          {/* Tabs */}
          <div className="flex items-center gap-4 mb-6 border-b border-[var(--color-border)] pb-4">
            <button
              onClick={() => setActiveTab('results')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'results'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]'
              }`}
            >
              <BarChart3 className="h-4 w-4" />
              Results
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'history'
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-border)]'
              }`}
            >
              <History className="h-4 w-4" />
              History
            </button>
          </div>

          {/* Results Tab */}
          {activeTab === 'results' && simulation && (
            <div className="space-y-6">
              {/* Recommendation */}
              <div className="flex items-start justify-between p-4 bg-[var(--color-background)] rounded-lg border border-[var(--color-border)]">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <RecommendationBadge type={simulation.recommendation.type} />
                    <StatusBadge status={simulation.recommendation.confidence} />
                  </div>
                  <p className="text-sm text-[var(--color-text-primary)]">{simulation.recommendation.rationale}</p>
                  {simulation.recommendation.expected_value && (
                    <p className="text-xs text-[var(--color-text-tertiary)] mt-2">
                      Expected Value: ${simulation.recommendation.expected_value.toLocaleString()}
                    </p>
                  )}
                </div>
              </div>

              {/* Outcomes Grid */}
              <div>
                <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">Simulation Outcomes</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <ConfidenceIntervalDisplay ci={simulation.outcomes.ate} label="ATE" />
                  <ConfidenceIntervalDisplay ci={simulation.outcomes.trx_lift} label="TRx Lift" />
                  <ConfidenceIntervalDisplay ci={simulation.outcomes.nrx_lift} label="NRx Lift" />
                  <ConfidenceIntervalDisplay ci={simulation.outcomes.roi} label="ROI" unit="x" />
                </div>
              </div>

              {/* Fidelity Metrics */}
              <div>
                <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">Model Fidelity</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <FidelityGauge score={simulation.fidelity.data_coverage} label="Data Coverage" />
                  <FidelityGauge score={simulation.fidelity.calibration} label="Calibration" />
                  <FidelityGauge score={simulation.fidelity.temporal_alignment} label="Temporal Alignment" />
                  <FidelityGauge score={simulation.fidelity.feature_completeness} label="Feature Completeness" />
                </div>
              </div>

              {/* Evidence & Risks */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-2 flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Supporting Evidence
                  </h4>
                  <ul className="space-y-1">
                    {simulation.recommendation.evidence.map((item, idx) => (
                      <li key={idx} className="text-sm text-[var(--color-text-primary)] flex items-start gap-2">
                        <span className="text-green-500 mt-1">+</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-2 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    Risk Factors
                  </h4>
                  <ul className="space-y-1">
                    {simulation.recommendation.risk_factors?.map((item, idx) => (
                      <li key={idx} className="text-sm text-[var(--color-text-primary)] flex items-start gap-2">
                        <span className="text-yellow-500 mt-1">!</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Execution Info */}
              <div className="pt-4 border-t border-[var(--color-border)] flex items-center justify-between text-xs text-[var(--color-text-tertiary)]">
                <span>Simulation ID: {simulation.simulation_id}</span>
                <span>Executed in {simulation.execution_time_ms}ms</span>
              </div>
            </div>
          )}

          {activeTab === 'results' && !simulation && (
            <div className="text-center py-12">
              <FlaskConical className="h-12 w-12 text-[var(--color-text-tertiary)] mx-auto mb-4" />
              <p className="text-[var(--color-text-secondary)]">Run a simulation to see results</p>
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="space-y-3">
              {history.map((sim) => (
                <div
                  key={sim.simulation_id}
                  onClick={() => {
                    // In real app, would fetch full simulation
                    setSelectedSimulation(SAMPLE_SIMULATION);
                    setActiveTab('results');
                  }}
                  className="flex items-center justify-between p-4 bg-[var(--color-background)] rounded-lg border border-[var(--color-border)] cursor-pointer hover:border-[var(--color-primary)] transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="p-2 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)]">
                      <FlaskConical className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-[var(--color-text-primary)]">
                        {sim.intervention_type.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                      </p>
                      <p className="text-xs text-[var(--color-text-tertiary)]">
                        {sim.brand} - {new Date(sim.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <p className="text-sm font-medium text-[var(--color-text-primary)]">
                        ATE: {sim.ate_estimate.toFixed(2)}
                      </p>
                    </div>
                    <RecommendationBadge type={sim.recommendation_type} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Info Footer */}
      <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
          About the Digital Twin
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-[var(--color-text-secondary)]">
          <div>
            <h4 className="font-medium text-[var(--color-text-primary)] mb-2">Intervention Types</h4>
            <ul className="list-disc list-inside space-y-1">
              <li><strong>HCP Engagement</strong> - Field force interactions with physicians</li>
              <li><strong>Patient Support</strong> - Hub services and adherence programs</li>
              <li><strong>Digital Marketing</strong> - Online campaigns and content</li>
              <li><strong>Rep Training</strong> - Sales force education programs</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-[var(--color-text-primary)] mb-2">How It Works</h4>
            <p>
              The Digital Twin uses causal models trained on historical data to simulate the
              counterfactual outcomes of interventions. It estimates the Average Treatment Effect (ATE)
              and provides confidence intervals to quantify uncertainty.
            </p>
          </div>
        </div>
        <p className="text-xs text-[var(--color-text-tertiary)] mt-4">
          Last simulation: {health.last_simulation_at ?? 'Never'}
        </p>
      </div>
    </div>
  );
}
