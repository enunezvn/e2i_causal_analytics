/**
 * Intervention Impact Page
 * ========================
 *
 * Intervention analysis dashboard. Current real substrate:
 * - Digital Twin simulation (`POST /api/digital-twin/simulate`) — fully
 *   wired: run a simulation and see the real ATE/CI/recommendation.
 *
 * Honestly gated (no backend substrate yet — verified against the live
 * OpenAPI spec):
 * - Interventions catalog: no endpoint serves a list of real intervention
 *   programs, so there is no selector. The previous fabricated
 *   INTERVENTIONS catalog (four invented pharma programs presented as
 *   real records) was DELETED.
 * - Causal-impact time-series, before/after comparisons, treatment-effect
 *   estimates, and segment heterogeneity per intervention: each tab
 *   renders an explicit empty state (F-002). The dead chart scaffolding
 *   and unreachable fabricated narratives that used to sit behind those
 *   gates were removed.
 *
 * @module pages/InterventionImpact
 */

import { useState, useMemo } from 'react';
import {
  Activity,
  Beaker,
  GitBranch,
  ArrowRight,
  Info,
  FlaskConical,
  Download,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EmptyState } from '@/components/ui/EmptyState';
import { SimulationPanel, ScenarioResults, RecommendationCards } from '@/components/digital-twin';
import { useRunSimulation } from '@/hooks/api/use-digital-twin';
import type { SimulationRequest, SimulationResponse, SimulationRecommendation } from '@/types/digital-twin';
import { RecommendationType, ConfidenceLevel, Recommendation } from '@/types/digital-twin';

// =============================================================================
// COMPONENT
// =============================================================================

function InterventionImpact() {
  const [simulationResults, setSimulationResults] = useState<SimulationResponse | null>(null);

  // Digital Twin simulation mutation (real API)
  const { mutate: runSimulation, isPending: isSimulating } = useRunSimulation({
    onSuccess: (data) => {
      setSimulationResults(data);
    },
  });

  // Handle simulation request - converts legacy SimulationRequest to SimulateRequest
  const handleSimulate = (request: SimulationRequest) => {
    runSimulation({
      intervention: {
        intervention_type: request.intervention_type,
        duration_weeks: Math.ceil(request.duration_days / 7),
      },
      brand: request.brand,
      twin_count: request.sample_size,
    });
  };

  // Convert SimulationResponse recommendation to SimulationRecommendation interface
  const simulationRecommendation = useMemo((): SimulationRecommendation | null => {
    if (!simulationResults) return null;

    // Map Recommendation enum to RecommendationType
    const typeMap: Record<Recommendation, RecommendationType> = {
      [Recommendation.DEPLOY]: RecommendationType.DEPLOY,
      [Recommendation.SKIP]: RecommendationType.SKIP,
      [Recommendation.REFINE]: RecommendationType.REFINE,
    };

    // Derive confidence level from simulation_confidence score
    let confidence: ConfidenceLevel;
    if (simulationResults.simulation_confidence >= 0.7) {
      confidence = ConfidenceLevel.HIGH;
    } else if (simulationResults.simulation_confidence >= 0.4) {
      confidence = ConfidenceLevel.MEDIUM;
    } else {
      confidence = ConfidenceLevel.LOW;
    }

    // Build evidence array from real simulation results
    const evidence: string[] = [];
    if (simulationResults.is_significant) {
      evidence.push(`Effect is statistically significant (ATE: ${simulationResults.simulated_ate.toFixed(3)})`);
    }
    if (simulationResults.effect_size_cohens_d) {
      evidence.push(`Effect size (Cohen's d): ${simulationResults.effect_size_cohens_d.toFixed(2)}`);
    }
    if (simulationResults.statistical_power) {
      evidence.push(`Statistical power: ${(simulationResults.statistical_power * 100).toFixed(0)}%`);
    }
    evidence.push(`CI: [${simulationResults.simulated_ci_lower.toFixed(3)}, ${simulationResults.simulated_ci_upper.toFixed(3)}]`);

    return {
      type: typeMap[simulationResults.recommendation],
      confidence,
      rationale: simulationResults.recommendation_rationale,
      evidence,
      risk_factors: simulationResults.fidelity_warning ? [simulationResults.fidelity_warning_reason ?? 'Fidelity warning present'] : undefined,
    };
  }, [simulationResults]);

  // Export: only real simulation output, never fabricated analysis blobs.
  const handleExport = () => {
    if (!simulationResults) return;
    const blob = new Blob([JSON.stringify(simulationResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `digital-twin-simulation-${simulationResults.simulation_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Intervention Impact</h1>
          <p className="text-muted-foreground">
            Before/after comparisons, treatment effects, and counterfactual analysis.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="icon"
            onClick={handleExport}
            disabled={!simulationResults}
            aria-label="Export simulation results"
            title={
              simulationResults
                ? 'Export the latest simulation results as JSON'
                : 'Run a digital-twin simulation to enable export'
            }
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Interventions catalog — honestly gated. The backend exposes no
          interventions-registry endpoint, so there is nothing real to put
          in a selector. The former fabricated catalog was removed. */}
      <div className="mb-8">
        <EmptyState
          title="No intervention catalog available"
          description="The backend does not yet expose an interventions registry, so historical intervention analyses cannot be selected here. Use the Digital Twin tab to pre-screen intervention scenarios against the live twin simulator."
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="causal" className="space-y-6">
        <TabsList>
          <TabsTrigger value="causal" className="gap-2">
            <Activity className="h-4 w-4" />
            Causal Impact
          </TabsTrigger>
          <TabsTrigger value="beforeafter" className="gap-2">
            <ArrowRight className="h-4 w-4" />
            Before/After
          </TabsTrigger>
          <TabsTrigger value="effects" className="gap-2">
            <Beaker className="h-4 w-4" />
            Treatment Effects
          </TabsTrigger>
          <TabsTrigger value="segments" className="gap-2">
            <GitBranch className="h-4 w-4" />
            Segment Analysis
          </TabsTrigger>
          <TabsTrigger value="digital-twin" className="gap-2">
            <FlaskConical className="h-4 w-4" />
            Digital Twin
          </TabsTrigger>
        </TabsList>

        {/* Causal Impact Tab — no counterfactual time-series endpoint yet */}
        <TabsContent value="causal" className="space-y-6">
          <EmptyState
            title="No causal impact data available"
            description="Counterfactual time-series analysis requires an API-backed analysis run. No endpoint serves per-intervention causal-impact series yet."
          />
        </TabsContent>

        {/* Before/After Tab — no pre/post snapshot endpoint yet */}
        <TabsContent value="beforeafter" className="space-y-6">
          <EmptyState
            title="No before/after data available"
            description="Pre- and post-intervention metric snapshots will appear once a per-intervention analysis endpoint exists."
          />
        </TabsContent>

        {/* Treatment Effects Tab — no per-intervention ATE endpoint yet */}
        <TabsContent value="effects" className="space-y-6">
          <EmptyState
            title="No treatment effect estimates available"
            description="ATE, confidence intervals, p-values, and effect sizes will appear here once a per-intervention analysis endpoint exists."
          />
        </TabsContent>

        {/* Segment Analysis Tab — no per-intervention CATE endpoint yet */}
        <TabsContent value="segments" className="space-y-6">
          <EmptyState
            title="No segment heterogeneity data available"
            description="Heterogeneous treatment effects by segment will appear here once a per-intervention analysis endpoint exists."
          />
        </TabsContent>

        {/* Digital Twin Tab — REAL substrate, fully wired */}
        <TabsContent value="digital-twin" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Simulation Panel - Left Side */}
            <div className="lg:col-span-1">
              <SimulationPanel
                onSimulate={handleSimulate}
                isSimulating={isSimulating}
                initialBrand="Remibrutinib"
                brands={['Remibrutinib', 'Fabhalta', 'Kisqali']}
              />
            </div>

            {/* Results and Recommendations - Right Side */}
            <div className="lg:col-span-2 space-y-6">
              {/* The real simulation response is threaded through (the
                  former results={null} TODO hid every completed run). */}
              <ScenarioResults
                results={simulationResults}
                isLoading={isSimulating}
              />

              {/* No deployment / refinement / deep-analysis flows exist in
                  the backend yet, so no action callbacks are wired —
                  RecommendationCards hides its action buttons rather than
                  showing dead controls. */}
              <RecommendationCards recommendation={simulationRecommendation} />
            </div>
          </div>

          {/* Digital Twin Context Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                About Digital Twin Simulation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium text-blue-600">Pre-Screen Interventions</h4>
                  <p className="text-sm text-muted-foreground">
                    Test intervention scenarios virtually before committing real resources.
                    The digital twin models HCP behavior and market dynamics to predict outcomes.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-emerald-600">Causal Inference Engine</h4>
                  <p className="text-sm text-muted-foreground">
                    Powered by DoWhy and EconML, the simulation uses causal models trained
                    on historical data to estimate treatment effects and confidence intervals.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-amber-600">Fidelity Metrics</h4>
                  <p className="text-sm text-muted-foreground">
                    Each simulation includes fidelity scores indicating how well the model
                    represents your specific market conditions and data coverage.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default InterventionImpact;
