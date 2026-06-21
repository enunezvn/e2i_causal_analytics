# Unified Agent-Led Causal Page (P0) Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax. Execute in an isolated git worktree; TDD red-first; commit per task; drive to a green fixed point and use codex:codex-rescue when stuck.

**Goal:** Collapse the two causal pages into ONE agent-led page at `/causal-analysis` — landing on the validated-effects leaderboard (grain + brand facets, brand + summary per row, drill-down into the existing deep view incl. #1030's estimator-comparison panel) with a secondary "Pose your own question" manual panel — and retire `/causal-discovery` via a redirect.

**Architecture:** `CausalAnalysis.tsx` is rebuilt as a leaderboard-led page that reuses the discover-effects job hook (`useDiscoverEffects`) for the landing leaderboard and the agent-analyze drill-down (`getCausalAgentAnalysis`) for per-row depth; the manual treatment/outcome/brand path (sourced from `/causal/variables`) moves into a collapsible secondary panel. The drill-down body and the manual-result body are unified through a shared `CausalAnalysisDetail` component (extracted from the post-#1030 `CausalDiscovery.tsx` drill-down, including its `EstimatorComparisonPanel`). The `/causal-discovery` route becomes a `<Navigate to="/causal-analysis" replace />` and is dropped from the sidebar nav config; `CausalDiscovery.tsx` and its test are deleted.

**Tech Stack:** React 18 + TypeScript, react-router-dom (`Navigate`), TanStack Query, Radix/shadcn UI, lucide-react, Vitest + Testing Library (unit), Playwright (e2e). No new deps.

**Depends on (sequencing):** PR #1030 merged to `main` (estimator-comparison panel, `EstimatorComparison`/`summary` types, connected-DAG fix — plan against the worktree `/home/enunez/Projects/wt_causal_discovery_revamp`). P1 backend merged (adds `brand` + `adjustment_set` + `summary` to `DiscoveredEffect` and brand-scoped discover-effects). This plan consumes those response shapes — it does not add backend fields.

---

## File Structure

| File | Create/Modify/Delete | Responsibility |
|---|---|---|
| `frontend/src/types/causal.ts` | Modify | Add `adjustment_set?: string[]` to `DiscoveredEffect` (P1 backend already returns it; `brand` + `summary` already present post-#1030). |
| `frontend/src/api/causal.ts` | Modify | `discoverCausalEffects(dataset, brand)` already param-encodes `dataset` — no signature change; the page passes the grain's dataset as `dataset`. (Read-only confirmation step.) |
| `frontend/src/hooks/api/use-causal.ts` | Modify | No signature change — `useDiscoverEffects(dataset, brand)` already threads dataset; the page selects the dataset from the grain facet. (Read-only confirmation step.) |
| `frontend/src/components/causal/CausalAnalysisDetail.tsx` | Create | Shared deep-view body: ATE/CI/p, estimator, gate, discovered confounders, DAG (`CausalDiscoveryViz` + per-test refutation), `EstimatorComparisonPanel`, interpretation (executive_summary/narrative/key_insights/recommendations). Consumed by both the leaderboard drill-down and the manual-result panel. |
| `frontend/src/components/causal/CausalAnalysisDetail.test.tsx` | Create | Unit tests: renders ATE/estimator/gate, feeds DAG nodes/edges + refutation count to the viz, renders the estimator-comparison panel + key insights + recommendations. |
| `frontend/src/pages/CausalAnalysis.tsx` | Modify (rewrite) | Unified agent-led page: leaderboard landing (grain + brand facets, brand + summary columns, drill-down via `CausalAnalysisDetail`) + secondary "Pose your own question" manual panel (treatment/outcome/brand/estimator from `/variables`, result via `CausalAnalysisDetail`) + Estimators/History tabs retained. |
| `frontend/src/pages/CausalAnalysis.test.tsx` | Modify (rewrite) | Unit tests for the unified page: leaderboard empty/running/ranked + brand+summary columns + brand facet default-null + grain facet + drill-down + manual panel run. |
| `frontend/src/pages/CausalDiscovery.tsx` | Delete | Page merged into `CausalAnalysis.tsx`. |
| `frontend/src/pages/CausalDiscovery.test.tsx` | Delete | Page deleted. |
| `frontend/src/router/routes.tsx` | Modify | `/causal-discovery` route element → `<Navigate to="/causal-analysis" replace />`; drop the `/causal-discovery` `routeConfigs` entry; drop the now-unused `CausalDiscovery` lazy import. |
| `frontend/src/router/routes.test.ts` | Modify | Update the `causal` IA section expectation (drop "Causal Discovery"); the `routes.test.tsx` recovery-route file is untouched. |
| `frontend/src/router/routes.redirect.test.tsx` | Create | Lock: `/causal-discovery` is routed to a `Navigate` redirect (not `NotFound`) and is NOT in the nav config. |
| `frontend/e2e/pages/causal-analysis.page.ts` | Modify | Add leaderboard POM locators (header, agent-driven badge, leaderboard empty state, Discover button, "Pose your own question" trigger) consumed by the realigned spec. |
| `frontend/e2e/pages/causal-discovery.page.ts` | Modify (redirect POM) | Repoint to assert the redirect lands on `/causal-analysis` (the page no longer exists at `/causal-discovery`). |
| `frontend/e2e/specs/causal-analysis.spec.ts` | Modify | Realign to the rebuilt page (leaderboard landing + manual panel) — grep + fix desynced `getByText` locators. |
| `frontend/e2e/specs/causal-discovery.spec.ts` | Modify | Replace page-load assertions with a single redirect assertion (`/causal-discovery` → `/causal-analysis`). |
| `frontend/e2e/fixtures/test-data.ts` | Modify | `ROUTES.CAUSAL_ANALYSIS = '/causal-analysis'` added; `CAUSAL_DISCOVERY` retained (smoke spec hits it to assert the redirect serves < 400). |

---

### Task 1: Add `adjustment_set` to the `DiscoveredEffect` type (consume P1's contract)

P1's backend adds `brand`, `adjustment_set`, and `summary` to `DiscoveredEffect`. Post-#1030 `types/causal.ts` already has `summary` (line 304) but NOT `brand` or `adjustment_set` on `DiscoveredEffect`. Add both so the leaderboard can surface the brand column and (later) the modeled adjustment set without an `any` cast.

**Files:**
- Modify: `frontend/src/types/causal.ts:285-308` (the `DiscoveredEffect` interface)
- Test: `frontend/src/types/causal.types.test.ts` (create — a compile-time shape lock)

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/types/causal.types.test.ts
import { describe, it, expect } from 'vitest';
import type { DiscoveredEffect } from './causal';

describe('DiscoveredEffect carries the P1 SSOT fields', () => {
  it('accepts brand + adjustment_set + summary (compile-time shape lock)', () => {
    const e: DiscoveredEffect = {
      treatment: 'treatment_arm',
      outcome: 'persistent_180d',
      status: 'completed',
      statistical_significance: true,
      confidence_score: 0.9,
      n_rows: 1500,
      brand: 'Kisqali',
      adjustment_set: ['disease_severity', 'academic_hcp'],
      summary: 'treatment_arm raises persistent_180d by +0.088.',
    };
    expect(e.brand).toBe('Kisqali');
    expect(e.adjustment_set).toEqual(['disease_severity', 'academic_hcp']);
    expect(e.summary).toContain('+0.088');
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd frontend && npx vitest run src/types/causal.types.test.ts`
Expected: FAIL — `tsc`/vitest type error: `brand` and `adjustment_set` do not exist on `DiscoveredEffect` (only `summary` does).

- [ ] **Step 3: Add the fields**

In `frontend/src/types/causal.ts`, inside `interface DiscoveredEffect` (after the `outcome: string;` line at the top of the interface), add:

```typescript
  /** Brand this question is scoped to (SSOT-derived; null = all brands). */
  brand?: string | null;
  /** Modeled backdoor set used for this estimate (SSOT confounders_controlled). */
  adjustment_set?: string[];
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/types/causal.types.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types/causal.ts frontend/src/types/causal.types.test.ts
git commit -m "feat(causal-fe): DiscoveredEffect carries SSOT brand + adjustment_set (P1 contract)"
```

---

### Task 2: Extract the shared `CausalAnalysisDetail` deep-view component

The post-#1030 `CausalDiscovery.tsx` drill-down (lines 438-577) and its `EstimatorComparisonPanel` (lines 138-198) are the canonical deep view. Extract them into `frontend/src/components/causal/CausalAnalysisDetail.tsx` so both the unified leaderboard drill-down AND the manual "Pose your own question" result render the SAME deep view (DAG + per-test refutation + estimator-comparison + interpretation). This is a pure refactor — copy the existing code verbatim, parameterized by an `AgentCausalAnalysisResponse`.

**Files:**
- Create: `frontend/src/components/causal/CausalAnalysisDetail.tsx`
- Test: `frontend/src/components/causal/CausalAnalysisDetail.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/components/causal/CausalAnalysisDetail.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CausalAnalysisDetail } from './CausalAnalysisDetail';
import type { AgentCausalAnalysisResponse } from '@/types/causal';

// Stub the heavy DAG viz — assert the detail feeds it the agent's graph + refutation.
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({
    nodes,
    edges,
    refutationResults,
  }: {
    nodes: unknown[];
    edges: unknown[];
    refutationResults?: unknown[];
  }) => (
    <div
      data-testid="causal-dag"
      data-nodes={nodes.length}
      data-edges={edges.length}
      data-refutations={refutationResults?.length ?? 0}
    />
  ),
}));

const RESULT: AgentCausalAnalysisResponse = {
  analysis_id: 'a1',
  status: 'completed',
  treatment_var: 'treatment_arm',
  outcome_var: 'persistent_180d',
  dataset: 'patient_journeys',
  n_rows: 1500,
  data_source: 'synthetic',
  dag: {
    nodes: ['treatment_arm', 'persistent_180d', 'disease_severity'],
    edges: [
      ['treatment_arm', 'persistent_180d'],
      ['disease_severity', 'persistent_180d'],
    ],
    treatment_nodes: ['treatment_arm'],
    outcome_nodes: ['persistent_180d'],
    adjustment_sets: [['disease_severity']],
    dag_dot: null,
  },
  discovered_confounders: ['disease_severity'],
  ate: 0.0875,
  ate_ci_lower: 0.0867,
  ate_ci_upper: 0.0884,
  p_value: 0,
  statistical_significance: true,
  selected_estimator: 'LinearDML',
  estimator_comparison: {
    candidates: [
      { estimator: 'causal_forest', success: true, energy_score: 0.51, ate: 0.1, error: null, is_selected: false },
      { estimator: 'linear_dml', success: true, energy_score: 0.48, ate: 0.0875, error: null, is_selected: true },
    ],
    selection_reason: 'confounding-robust preferred over OLS',
    energy_score_gap: 0.03,
    n_evaluated: 2,
    n_succeeded: 2,
    quality_tier: 'good',
    requires_review: false,
  },
  refutation: {
    gate_decision: 'proceed',
    passed: true,
    needs_review: false,
    tests_passed: 2,
    tests_total: 3,
    sensitivity_e_value: 1.6,
    tests: [
      { test_name: 'placebo_treatment', passed: true, original_effect: 0.0875, new_effect: 0.001, p_value: 0.6 },
      { test_name: 'random_common_cause', passed: true, original_effect: 0.0875, new_effect: 0.086, p_value: 0.9 },
      { test_name: 'unobserved_common_cause', passed: true, original_effect: 0.0875, new_effect: 0.0875, p_value: 0 },
    ],
  },
  narrative: 'Treatment raises persistence.',
  executive_summary: 'Positive, robust effect.',
  recommendations: ['Monitor outcomes closely'],
  key_insights: ['Estimated causal effect: 0.09 (small)'],
  warnings: [],
  latency_ms: 4200,
};

describe('CausalAnalysisDetail', () => {
  it('renders the effect, estimator, gate, and discovered confounders', () => {
    render(<CausalAnalysisDetail result={RESULT} />);
    // ATE renders in the headline AND the selected-estimator comparison row (same
    // value by design — the headline IS the selected estimator's ATE) → getAllByText.
    expect(screen.getAllByText('0.0875').length).toBeGreaterThan(0);
    expect(screen.getByText(/Linear dml/i)).toBeInTheDocument();
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText(/disease_severity/)).toBeInTheDocument();
  });

  it('feeds the DAG (nodes + edges) and per-test refutation into the viz', () => {
    render(<CausalAnalysisDetail result={RESULT} />);
    const dag = screen.getByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-edges', '2');
    expect(dag).toHaveAttribute('data-refutations', '3');
  });

  it('renders the estimator-comparison panel (the #1030 data-driven evaluation)', () => {
    render(<CausalAnalysisDetail result={RESULT} />);
    expect(screen.getByText('Estimator selection (data-driven)')).toBeInTheDocument();
    expect(screen.getByText(/2\/2 estimators fit/)).toBeInTheDocument();
    expect(screen.getByText(/confounding-robust preferred over OLS/)).toBeInTheDocument();
  });

  it('renders interpretation: key insights + recommendations', () => {
    render(<CausalAnalysisDetail result={RESULT} />);
    expect(screen.getByText('Positive, robust effect.')).toBeInTheDocument();
    expect(screen.getByText('Key insights')).toBeInTheDocument();
    expect(screen.getByText('Recommended actions')).toBeInTheDocument();
    expect(screen.getByText('Monitor outcomes closely')).toBeInTheDocument();
  });

  it('shows an honest empty-state when no DAG was produced', () => {
    render(<CausalAnalysisDetail result={{ ...RESULT, dag: { ...RESULT.dag, nodes: [], edges: [] } }} />);
    expect(screen.getByText('No DAG produced')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd frontend && npx vitest run src/components/causal/CausalAnalysisDetail.test.tsx`
Expected: FAIL — `Failed to resolve import "./CausalAnalysisDetail"`.

- [ ] **Step 3: Create the component** (lifts the post-#1030 `CausalDiscovery.tsx` drill-down body + `EstimatorComparisonPanel` verbatim, parameterized by `result`)

```typescript
// frontend/src/components/causal/CausalAnalysisDetail.tsx
/**
 * CausalAnalysisDetail — the shared deep view for one validated causal effect.
 * ===========================================================================
 *
 * Lifted verbatim from the post-#1030 CausalDiscovery drill-down so BOTH the
 * unified page's leaderboard drill-down and its "Pose your own question" manual
 * result render the identical deep view: discovered DAG + per-test refutation,
 * the data-driven estimator-comparison panel (#1030), and the interpretation
 * (executive summary / narrative / key insights / recommendations). All values
 * are REAL agent outputs surfaced from the response — never fabricated.
 *
 * @module components/causal/CausalAnalysisDetail
 */

import { useMemo } from 'react';

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type {
  RefutationResult,
  RefutationMethod,
} from '@/components/visualizations/causal/RefutationTests';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/EmptyState';
import type {
  AgentCausalAnalysisResponse,
  EstimatorComparison,
  RefutationTestDetail,
} from '@/types/causal';

function formatEffect(ate: number | null | undefined): string {
  if (ate === null || ate === undefined || Number.isNaN(ate)) return 'N/A';
  return ate.toFixed(4);
}

function formatCI(lower?: number | null, upper?: number | null): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return '—';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
}

function gateBadge(decision?: string | null) {
  if (decision === 'proceed') return <Badge variant="default">Proceed</Badge>;
  if (decision === 'review') return <Badge variant="secondary">Review</Badge>;
  if (decision === 'block') return <Badge variant="destructive">Blocked</Badge>;
  return <Badge variant="outline">—</Badge>;
}

// Backend refutation test_name -> the viz's RefutationMethod union. The backend
// surfaces the contract key, so the sensitivity test arrives as
// `unobserved_common_cause`; `sensitivity_e_value` (the raw enum) is mapped too
// as defense-in-depth so it can never fall through to "Random Common Cause".
const REFUTATION_METHOD_MAP: Record<string, RefutationMethod> = {
  placebo_treatment: 'placebo_treatment',
  random_common_cause: 'random_common_cause',
  data_subset: 'data_subset',
  bootstrap: 'bootstrap',
  unobserved_common_cause: 'add_unobserved_common_cause',
  add_unobserved_common_cause: 'add_unobserved_common_cause',
  sensitivity_e_value: 'add_unobserved_common_cause',
};

function toRefutationResults(tests: RefutationTestDetail[] | undefined | null): RefutationResult[] {
  if (!tests) return [];
  return tests.map((t, i) => ({
    id: `${t.test_name}-${i}`,
    method: REFUTATION_METHOD_MAP[t.test_name] ?? 'random_common_cause',
    originalEstimate: t.original_effect ?? 0,
    refutedEstimate: t.new_effect ?? 0,
    pValue: t.p_value ?? 0,
    passed: t.passed,
    description: t.details ?? undefined,
  }));
}

// The agent fits and energy-scores several estimators and picks the lowest score
// with a robust-over-fast tie-break. Surface that evaluation so the analyst sees
// WHAT was compared and WHY the winner won — not just the winner's name.
function EstimatorComparisonPanel({ comparison }: { comparison: EstimatorComparison }) {
  const ranked = [...comparison.candidates].sort((a, b) => {
    if (a.energy_score == null) return 1;
    if (b.energy_score == null) return -1;
    return a.energy_score - b.energy_score;
  });
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">Estimator selection (data-driven)</p>
        <p className="text-xs text-muted-foreground">
          {comparison.n_succeeded}/{comparison.n_evaluated} estimators fit · lower energy score is
          better
        </p>
      </div>
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="border-b bg-muted/40 text-left text-xs uppercase text-muted-foreground">
            <tr>
              <th className="p-2 font-medium">Estimator</th>
              <th className="p-2 font-medium">Energy score</th>
              <th className="p-2 font-medium">ATE</th>
              <th className="p-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((c) => (
              <tr
                key={c.estimator}
                className={`border-b last:border-0 ${c.is_selected ? 'bg-muted/50 font-medium' : ''}`}
              >
                <td className="p-2 capitalize">
                  {c.estimator.replace(/_/g, ' ')}
                  {c.is_selected && (
                    <Badge variant="default" className="ml-2 align-middle">
                      Selected
                    </Badge>
                  )}
                </td>
                <td className="p-2">{c.energy_score != null ? c.energy_score.toFixed(4) : '—'}</td>
                <td className="p-2">{c.ate != null ? c.ate.toFixed(4) : '—'}</td>
                <td className="p-2 text-xs text-muted-foreground">
                  {c.success ? 'fit' : `failed${c.error ? `: ${c.error}` : ''}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {comparison.selection_reason && (
        <p className="text-xs text-muted-foreground">
          <span className="font-medium">Why this estimator:</span> {comparison.selection_reason}
          {comparison.quality_tier ? ` (quality: ${comparison.quality_tier})` : ''}
        </p>
      )}
    </div>
  );
}

/** The shared deep view for one validated causal effect (agent-analyze result). */
export function CausalAnalysisDetail({ result }: { result: AgentCausalAnalysisResponse }) {
  // Map the analysis's DAG onto the shared causal-graph visualization. The
  // treatment->outcome edge carries the estimated effect.
  const { vizNodes, vizEdges } = useMemo((): {
    vizNodes: CausalNode[];
    vizEdges: CausalEdge[];
  } => {
    const dag = result.dag;
    const treatmentSet = new Set(dag.treatment_nodes);
    const outcomeSet = new Set(dag.outcome_nodes);
    const confounderSet = new Set(dag.adjustment_sets.flat());
    const nodes: CausalNode[] = dag.nodes.map((name) => ({
      id: name,
      label: name,
      type: treatmentSet.has(name)
        ? 'treatment'
        : outcomeSet.has(name)
          ? 'outcome'
          : confounderSet.has(name)
            ? 'confounder'
            : 'variable',
    }));
    const edges: CausalEdge[] = dag.edges.map(([source, target]) => {
      const isEffectEdge = treatmentSet.has(source) && outcomeSet.has(target);
      return {
        id: `${source}->${target}`,
        source,
        target,
        type: 'causal' as const,
        ...(isEffectEdge && result.ate !== null && result.ate !== undefined
          ? { effect: result.ate }
          : {}),
      };
    });
    return { vizNodes: nodes, vizEdges: edges };
  }, [result]);

  const refutationResults = useMemo(
    () => toRefutationResults(result.refutation?.tests),
    [result]
  );

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-3 gap-6">
        <div className="text-center">
          <div className="text-3xl font-bold text-primary">{formatEffect(result.ate)}</div>
          <div className="text-sm text-muted-foreground mt-1">
            ATE · 95% CI {formatCI(result.ate_ci_lower, result.ate_ci_upper)}
          </div>
          {result.p_value !== null && result.p_value !== undefined && (
            <div className="text-xs text-muted-foreground mt-1">p = {result.p_value.toFixed(4)}</div>
          )}
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold capitalize">
            {result.selected_estimator ? result.selected_estimator.replace(/_/g, ' ') : 'N/A'}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Estimator (data-driven) · {result.n_rows.toLocaleString()} rows
          </div>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center gap-2">
            {gateBadge(result.refutation.gate_decision)}
            {result.statistical_significance ? (
              <Badge variant="default">Significant</Badge>
            ) : (
              <Badge variant="secondary">Not significant</Badge>
            )}
          </div>
          <div className="text-xs text-muted-foreground mt-2">
            Refutation: {result.refutation.tests_passed ?? '—'}
            {result.refutation.tests_total !== null && result.refutation.tests_total !== undefined
              ? ` / ${result.refutation.tests_total}`
              : ''}{' '}
            passed
          </div>
        </div>
      </div>

      {result.discovered_confounders && result.discovered_confounders.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Confounders the data identified (adjusted for in the estimate):{' '}
          <span className="font-medium">{result.discovered_confounders.join(', ')}</span>
        </p>
      )}

      {vizNodes.length > 0 ? (
        <div className="space-y-2">
          <CausalDiscoveryViz
            nodes={vizNodes}
            edges={vizEdges}
            refutationResults={refutationResults}
            showEffectsTable={false}
          />
          <p className="text-xs text-muted-foreground">
            Confounders point into both treatment and outcome (the backdoor paths the estimate
            adjusts for). A node drawn without edges has no detected causal link to this question.
          </p>
        </div>
      ) : (
        <EmptyState
          title="No DAG produced"
          description="The agent did not return a causal graph for this run."
        />
      )}

      {result.estimator_comparison && (
        <EstimatorComparisonPanel comparison={result.estimator_comparison} />
      )}

      {(result.executive_summary ||
        result.narrative ||
        result.key_insights.length > 0 ||
        result.recommendations.length > 0) && (
        <div className="space-y-4 text-sm">
          {result.executive_summary && <p className="font-medium">{result.executive_summary}</p>}
          {result.narrative && (
            <p className="text-muted-foreground whitespace-pre-line">{result.narrative}</p>
          )}
          {result.key_insights.length > 0 && (
            <div>
              <p className="mb-1 font-medium">Key insights</p>
              <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
                {result.key_insights.map((k, i) => (
                  <li key={i}>{k}</li>
                ))}
              </ul>
            </div>
          )}
          {result.recommendations.length > 0 && (
            <div>
              <p className="mb-1 font-medium">Recommended actions</p>
              <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
                {result.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default CausalAnalysisDetail;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/causal/CausalAnalysisDetail.test.tsx`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/causal/CausalAnalysisDetail.tsx frontend/src/components/causal/CausalAnalysisDetail.test.tsx
git commit -m "refactor(causal-fe): extract shared CausalAnalysisDetail deep view (DAG + refutation + estimator-comparison)"
```

---

### Task 3: Rewrite `CausalAnalysis.tsx` as the unified agent-led page

Rebuild the page so the LANDING is the validated-effects leaderboard (no empty form). Facets: **grain** (Patient / HCP / Trigger) + **brand**. Patient grain (`patient_journeys`) is live now; HCP (`hcp_adoption`) and Trigger (`nba_triggers`) datasets are added by P2/P3 — P0 wires the grain facet UI but disables the not-yet-backed grains with an honest note (in scope: the facet shell; out of scope: the HCP/Trigger loaders). Each row surfaces its **brand** + **summary** and drills into `CausalAnalysisDetail`. A secondary **"Pose your own question"** panel keeps the manual treatment/outcome/brand/estimator path (sourced from `/causal/variables`) and renders its result through the same `CausalAnalysisDetail`. The Estimators + History tabs are retained.

**Files:**
- Modify (rewrite): `frontend/src/pages/CausalAnalysis.tsx`
- Test: `frontend/src/pages/CausalAnalysis.test.tsx` (rewritten in Task 4)

- [ ] **Step 1: Write the failing tests** (rewrite `CausalAnalysis.test.tsx`)

```typescript
// frontend/src/pages/CausalAnalysis.test.tsx
/**
 * CausalAnalysis Page — unified agent-led page
 * ============================================
 *
 * The page LANDS on the validated-effects leaderboard (discover-effects job),
 * faceted by grain + brand, each row surfacing its brand + plain-language
 * summary and drilling into the deep view (DAG + refutation + estimator
 * comparison). A secondary "Pose your own question" panel keeps the manual
 * treatment/outcome path sourced from /causal/variables. These tests lock the
 * honest empty/running states, the ranked leaderboard (brand + summary), the
 * facets, the drill-down, and the manual run.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalAnalysis from './CausalAnalysis';

// Stub the shared deep view — assert the page mounts it for the selected row /
// manual result (its internals are covered by CausalAnalysisDetail.test.tsx).
vi.mock('@/components/causal/CausalAnalysisDetail', () => ({
  CausalAnalysisDetail: ({ result }: { result: { analysis_id: string } }) => (
    <div data-testid="causal-detail" data-analysis-id={result.analysis_id} />
  ),
}));

vi.mock('@/hooks/api', () => ({
  useCausalHealth: vi.fn(),
  useCausalAnalysisHistory: vi.fn(),
  useCausalVariables: vi.fn(),
  useCausalBrands: vi.fn(),
  useDiscoverEffects: vi.fn(),
  useRunCausalAgentAnalysis: vi.fn(),
  useEstimators: vi.fn(),
}));

vi.mock('@/api/causal', () => ({
  getCausalAgentAnalysis: vi.fn(),
}));

import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useCausalBrands,
  useDiscoverEffects,
  useRunCausalAgentAnalysis,
  useEstimators,
} from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';

const VARIABLES = {
  dataset: 'patient_journeys',
  treatment_candidates: ['treatment_arm', 'treatment_initiated'],
  outcome_candidates: ['persistent_180d', 'discontinued_180d'],
  covariate_candidates: ['disease_severity', 'engagement_score'],
  columns: [],
};

const EFFECTS = [
  {
    treatment: 'treatment_arm',
    outcome: 'persistent_180d',
    status: 'completed',
    ate: 0.0875,
    ate_ci_lower: 0.0867,
    ate_ci_upper: 0.0884,
    p_value: 0,
    statistical_significance: true,
    selected_estimator: 'LinearDML',
    gate_decision: 'proceed',
    confidence_score: 0.9,
    impact: 0.0875,
    n_rows: 1500,
    brand: 'Kisqali',
    summary: 'treatment_arm raises persistent_180d by +0.088 — survived all robustness checks.',
    analysis_id: 'a1',
  },
  {
    treatment: 'treatment_arm',
    outcome: 'treatment_initiated',
    status: 'blocked',
    ate: -0.006,
    statistical_significance: true,
    selected_estimator: 'LinearDML',
    gate_decision: 'block',
    confidence_score: 0.4,
    impact: 0.006,
    n_rows: 1500,
    brand: 'Fabhalta',
    analysis_id: 'a3',
  },
];

const COMPLETED_JOB = {
  job_id: 'j1',
  status: 'completed',
  dataset: 'patient_journeys',
  brand: null,
  total: 2,
  completed: 2,
  effects: EFFECTS,
  note: 'ranked',
};

const DETAIL = { analysis_id: 'a1', status: 'completed' };

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function mockDiscover(overrides: Record<string, unknown> = {}) {
  (useDiscoverEffects as ReturnType<typeof vi.fn>).mockReturnValue({
    start: vi.fn(),
    isStarting: false,
    startError: null,
    job: null,
    ...overrides,
  });
}

describe('CausalAnalysis — unified agent-led page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useCausalHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
    (useCausalAnalysisHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
    });
    (useCausalVariables as ReturnType<typeof vi.fn>).mockReturnValue({ data: VARIABLES });
    (useCausalBrands as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { dataset: 'patient_journeys', brands: ['Remibrutinib', 'Kisqali', 'Fabhalta'] },
      isLoading: false,
      error: null,
    });
    (useEstimators as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
    });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockResolvedValue(DETAIL);
    mockDiscover();
  });

  it('lands on the leaderboard with an honest empty state before any run', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/No discovery run yet/i)).toBeInTheDocument();
  }, 20000);

  it('offers grain + brand facets; brand defaults to all (null) for the patient grain', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByLabelText('Grain')).toBeInTheDocument();
    expect(screen.getByLabelText('Brand')).toBeInTheDocument();
    // Patient grain (patient_journeys) is the default; brand null = all brands.
    expect(useDiscoverEffects).toHaveBeenCalledWith('patient_journeys', null);
  }, 20000);

  it('renders the ranked leaderboard with the brand column and per-row summary', () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText('persistent_180d')).toBeInTheDocument();
    expect(screen.getByText('treatment_initiated')).toBeInTheDocument();
    // Brand surfaced per row (SSOT-derived scope).
    expect(screen.getByText('Kisqali')).toBeInTheDocument();
    expect(screen.getByText('Fabhalta')).toBeInTheDocument();
    // Plain-language summary surfaced.
    expect(screen.getByText(/raises persistent_180d by \+0\.088/)).toBeInTheDocument();
    // Honest verdicts.
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText('Blocked')).toBeInTheDocument();
  }, 20000);

  it('shows progress while the agent is validating', () => {
    mockDiscover({ job: { ...COMPLETED_JOB, status: 'running', completed: 1 } });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/Validating… \(1\/2\)/)).toBeInTheDocument();
  }, 20000);

  it('drills a validated row into the shared deep view', async () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('persistent_180d'));
    const detail = await screen.findByTestId('causal-detail');
    expect(detail).toHaveAttribute('data-analysis-id', 'a1');
    expect(getCausalAgentAnalysis).toHaveBeenCalledWith('a1');
  }, 20000);

  it('keeps a "Pose your own question" panel and runs the manual agent path with it', () => {
    const mutateAsync = vi.fn().mockResolvedValue({ analysis_id: 'm1', status: 'completed' });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync,
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // The secondary manual panel is present (its trigger), defaulting collapsed.
    expect(screen.getByRole('button', { name: /Pose your own question/i })).toBeInTheDocument();
    // Expand it, then run the manual analysis with the data-driven defaults.
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    expect(screen.getByLabelText('Treatment variable')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
        dataset: 'patient_journeys',
        brand: undefined,
      })
    );
  }, 20000);

  it('explains why the candidate-question set is the size it is', () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/Why these 2 questions\?/)).toBeInTheDocument();
  }, 20000);

  it('renders the live estimator-registry total on the overview card', () => {
    (useEstimators as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { estimators: [], total: 12, by_library: {} },
      isLoading: false,
      isError: false,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText('12')).toBeInTheDocument();
  }, 20000);
});
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `cd frontend && npx vitest run src/pages/CausalAnalysis.test.tsx`
Expected: FAIL — the current page has no leaderboard / grain facet / "Pose your own question" panel; assertions like `No discovery run yet`, `getByLabelText('Grain')`, and `Why these 2 questions?` are not found.

- [ ] **Step 3: Rewrite the page**

Replace the entire contents of `frontend/src/pages/CausalAnalysis.tsx` with:

```typescript
/**
 * Causal Analysis Page — unified, agent-led
 * =========================================
 *
 * ONE page (the former /causal-discovery + /causal-analysis collapsed). The
 * LANDING is the validated-effects leaderboard: the causal_impact agent
 * proposes the causal questions from the gold-standard SSOT (no empty form),
 * validates each (guided DAG discovery + data-driven estimator + refutation
 * gate), and ranks them by confidence then impact. Facets: grain (Patient /
 * HCP / Trigger) + brand. Each row surfaces its brand + plain-language summary
 * and drills into the deep view (DAG + per-test refutation + estimator
 * comparison + interpretation).
 *
 * A secondary "Pose your own question" panel keeps the manual treatment /
 * outcome / brand / estimator path (sourced from GET /causal/variables) for
 * power users; its result renders through the SAME deep view.
 *
 * @module pages/CausalAnalysis
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  ChevronRight,
  GitBranch,
  Layers,
  Loader2,
  Network,
  Play,
  Settings,
  Sparkles,
  TrendingUp,
} from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { KPICard } from '@/components/visualizations';
import { CausalAnalysisDetail } from '@/components/causal/CausalAnalysisDetail';
import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useCausalBrands,
  useDiscoverEffects,
  useRunCausalAgentAnalysis,
  useEstimators,
} from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';
import type { DiscoveredEffect } from '@/types/causal';

// =============================================================================
// CONSTANTS
// =============================================================================

// Each grain is a `dataset` the agent estimates over. Patient (patient_journeys)
// is live; HCP (hcp_adoption) + Trigger (nba_triggers) datasets land in P2/P3,
// so their facet options are present but disabled until then (honest, not faked).
interface GrainOption {
  value: string;
  dataset: string;
  label: string;
  ready: boolean;
}
const GRAINS: GrainOption[] = [
  { value: 'patient', dataset: 'patient_journeys', label: 'Patient', ready: true },
  { value: 'hcp', dataset: 'hcp_adoption', label: 'HCP', ready: false },
  { value: 'trigger', dataset: 'nba_triggers', label: 'Trigger', ready: false },
];

// "All brands" sentinel — the Select needs a non-empty value; null is sent to the API.
const ALL_BRANDS = '__all__';

// Estimator selection for the manual panel. "auto" = the agent's data-driven
// energy-score routing — the DEFAULT. The override is an expert escape hatch;
// values MUST be members of the backend's AGENT_FORCEABLE_ESTIMATORS allowlist.
const AUTO_ESTIMATOR = 'auto';
const ESTIMATOR_OPTIONS: Array<{ value: string; label: string }> = [
  { value: AUTO_ESTIMATOR, label: 'Auto — agent decides (recommended)' },
  { value: 'CausalForestDML', label: 'Causal Forest — EconML' },
  { value: 'LinearDML', label: 'Linear DML — EconML' },
  { value: 'drlearner', label: 'DR-Learner — EconML' },
  { value: 'ols', label: 'Linear Regression (OLS)' },
  { value: 'propensity_score_weighting', label: 'Propensity Score Weighting — DoWhy' },
];

const LIBRARY_COLORS: Record<string, string> = {
  dowhy: '#3b82f6',
  econml: '#8b5cf6',
  causalml: '#06b6d4',
  networkx: '#f59e0b',
};

const DEFAULT_HEALTH = {
  status: 'unknown',
  libraries_available: { dowhy: false, econml: false, causalml: false, networkx: false },
  estimators_loaded: 0,
  pipeline_orchestrator_ready: false,
  hierarchical_analyzer_ready: false,
  analysis_count_24h: 0,
  average_latency_ms: null as number | null,
};

// =============================================================================
// HELPERS
// =============================================================================

function formatEffect(ate: number | null | undefined): string {
  if (ate === null || ate === undefined || Number.isNaN(ate)) return 'N/A';
  return ate.toFixed(4);
}

function formatCI(lower?: number | null, upper?: number | null): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return '—';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
}

// One column conveys both the run state and the robustness verdict. A computed
// effect is shown by its gate (Proceed/Review/Blocked); a run that produced no
// estimate is Failed; in-flight rows are Queued/Running.
function verdictBadge(e: DiscoveredEffect) {
  switch (e.status) {
    case 'completed':
      return <Badge variant="default">Proceed</Badge>;
    case 'needs_review':
      return <Badge variant="secondary">Review</Badge>;
    case 'blocked':
      return <Badge variant="destructive">Blocked</Badge>;
    case 'running':
      return (
        <Badge variant="outline" className="gap-1">
          <Loader2 className="h-3 w-3 animate-spin" /> Running
        </Badge>
      );
    case 'pending':
      return <Badge variant="outline">Queued</Badge>;
    default:
      return <Badge variant="destructive">Failed</Badge>;
  }
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function CausalAnalysis() {
  // ── Facets ────────────────────────────────────────────────────────────────
  const [grain, setGrain] = useState<string>('patient');
  const activeGrain = GRAINS.find((g) => g.value === grain) ?? GRAINS[0];
  const dataset = activeGrain.dataset;
  const [selectedBrand, setSelectedBrand] = useState<string>(ALL_BRANDS);
  const brandArg = selectedBrand === ALL_BRANDS ? null : selectedBrand;

  // ── Leaderboard (landing): the agent's validated, ranked effects ───────────
  const brandsQuery = useCausalBrands(dataset);
  const { start, isStarting, startError, job } = useDiscoverEffects(dataset, brandArg);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: ['causal', 'agent-analyze', selectedId],
    queryFn: () => getCausalAgentAnalysis(selectedId as string),
    enabled: !!selectedId,
  });
  const detailResult = detail.data;
  const effects: DiscoveredEffect[] = useMemo(() => job?.effects ?? [], [job]);
  const running = !!job && job.status !== 'completed';

  // ── Manual "Pose your own question" panel ──────────────────────────────────
  const [manualOpen, setManualOpen] = useState(false);
  const { data: variables } = useCausalVariables(dataset);
  const treatmentCandidates = variables?.treatment_candidates ?? ['treatment_arm'];
  const outcomeCandidates = variables?.outcome_candidates ?? ['persistent_180d'];
  const [treatmentVar, setTreatmentVar] = useState('treatment_arm');
  const [outcomeVar, setOutcomeVar] = useState('persistent_180d');
  const [estimator, setEstimator] = useState(AUTO_ESTIMATOR);
  const confounders = useMemo(
    () =>
      (variables?.covariate_candidates ?? []).filter(
        (c) => c !== treatmentVar && c !== outcomeVar
      ),
    [variables, treatmentVar, outcomeVar]
  );
  const runAgent = useRunCausalAgentAnalysis();
  const manualResult = runAgent.data;

  const handleRunManual = async () => {
    try {
      await runAgent.mutateAsync({
        treatment_var: treatmentVar,
        outcome_var: outcomeVar,
        dataset,
        estimator: estimator === AUTO_ESTIMATOR ? undefined : estimator,
        brand: brandArg ?? undefined,
      });
    } catch (error) {
      console.error('Causal agent analysis failed:', error);
    }
  };

  // ── Estimators + History tabs ──────────────────────────────────────────────
  const { data: healthData } = useCausalHealth();
  const {
    data: historyData,
    isLoading: historyLoading,
    isError: historyError,
  } = useCausalAnalysisHistory();
  const {
    data: estimatorsData,
    isLoading: estimatorsLoading,
    isError: estimatorsError,
  } = useEstimators();
  const [selectedLibrary, setSelectedLibrary] = useState<string>('all');
  const health = healthData ?? DEFAULT_HEALTH;
  const estimators = estimatorsData?.estimators ?? [];
  const visibleEstimators = estimators.filter(
    (e) => selectedLibrary === 'all' || e.library === selectedLibrary
  );
  const overviewMetrics = useMemo(() => {
    const availableLibraries = Object.values(health.libraries_available).filter(Boolean).length;
    const totalLibraries = Object.keys(health.libraries_available).length;
    return {
      librariesAvailable: `${availableLibraries}/${totalLibraries}`,
      estimatorsLoaded: estimatorsData?.total ?? health.estimators_loaded,
      analysisCount: health.analysis_count_24h,
      avgLatency: health.average_latency_ms
        ? `${(health.average_latency_ms / 1000).toFixed(1)}s`
        : 'N/A',
    };
  }, [health, estimatorsData]);

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
            <GitBranch className="h-8 w-8" />
            Causal Analysis
          </h1>
          <p className="text-muted-foreground">
            The agent proposes the causal questions from the gold-standard data, validates each
            (DAG + estimator + refutation gate), and ranks them by confidence and impact. No empty
            form — the agent decides; you read the ranked, validated results.
          </p>
        </div>
        <Badge variant="outline" className="flex items-center gap-1 self-start">
          <Sparkles className="h-3 w-3" />
          Agent-driven
        </Badge>
      </div>

      {/* Service Health Banner */}
      {health.status === 'healthy' ? (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertTitle className="text-green-800">Causal Engine Healthy</AlertTitle>
          <AlertDescription className="text-green-700">
            All {Object.values(health.libraries_available).filter(Boolean).length} causal libraries
            available. {health.analysis_count_24h} analyses completed in the last 24 hours.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Service Issue</AlertTitle>
          <AlertDescription>
            Some causal libraries may be unavailable. Check service health for details.
          </AlertDescription>
        </Alert>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          title="Libraries"
          value={overviewMetrics.librariesAvailable}
          icon={<Layers className="h-5 w-5" />}
        />
        <KPICard
          title="Estimators"
          value={overviewMetrics.estimatorsLoaded}
          icon={<Settings className="h-5 w-5" />}
        />
        <KPICard
          title="Analyses (24h)"
          value={overviewMetrics.analysisCount}
          icon={<Activity className="h-5 w-5" />}
        />
        <KPICard
          title="Avg Latency"
          value={overviewMetrics.avgLatency}
          icon={<TrendingUp className="h-5 w-5" />}
        />
      </div>

      <Tabs defaultValue="leaderboard" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="leaderboard">Validated effects</TabsTrigger>
          <TabsTrigger value="estimators">Estimators</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Leaderboard Tab (landing) */}
        <TabsContent value="leaderboard" className="space-y-6">
          {/* Facets + run control */}
          <Card>
            <CardHeader>
              <CardTitle>Discovered causal effects</CardTitle>
              <CardDescription>
                For each candidate question the agent builds the DAG, selects the estimator
                data-drivenly, and runs the refutation gate — then ranks the validated effects. This
                takes a few minutes; results fill in as each completes.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <label htmlFor="grain-select" className="text-sm font-medium text-muted-foreground">
                    Grain
                  </label>
                  <Select value={grain} onValueChange={setGrain} disabled={isStarting || running}>
                    <SelectTrigger id="grain-select" className="w-44" aria-label="Grain">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {GRAINS.map((g) => (
                        <SelectItem key={g.value} value={g.value} disabled={!g.ready}>
                          {g.label}
                          {g.ready ? '' : ' (coming soon)'}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center gap-2">
                  <label htmlFor="brand-select" className="text-sm font-medium text-muted-foreground">
                    Brand
                  </label>
                  <Select
                    value={selectedBrand}
                    onValueChange={setSelectedBrand}
                    disabled={isStarting || running}
                  >
                    <SelectTrigger id="brand-select" className="w-48" aria-label="Brand">
                      <SelectValue placeholder="All brands" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={ALL_BRANDS}>All brands</SelectItem>
                      {(brandsQuery.data?.brands ?? []).map((b) => (
                        <SelectItem key={b} value={b}>
                          {b}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={() => start()} disabled={isStarting || running}>
                  {isStarting || running ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {running && job
                        ? `Validating… (${job.completed}/${job.total})`
                        : 'Starting…'}
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      {job ? 'Re-run discovery' : 'Discover causal effects'}
                    </>
                  )}
                </Button>
                {job && (
                  <span className="text-sm text-muted-foreground">
                    {job.completed}/{job.total} questions validated
                  </span>
                )}
              </div>
              {!activeGrain.ready && (
                <p className="mt-3 text-xs text-muted-foreground">
                  The {activeGrain.label} grain&rsquo;s gold-standard loader is not wired yet — it
                  arrives in a later phase. The Patient grain is live now.
                </p>
              )}
              {startError && (
                <Alert variant="destructive" className="mt-4">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Discovery could not start</AlertTitle>
                  <AlertDescription>Please try again.</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Leaderboard */}
          {!job ? (
            <EmptyState
              title="No discovery run yet"
              description="Click Discover causal effects. The agent validates each candidate question and ranks the effects by confidence (robustness gate + significance) and impact (effect size)."
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Ranked causal effects</CardTitle>
                <CardDescription>
                  Validated by the agent (discovered DAG + estimator + refutation gate), ranked by
                  confidence then impact. Click a validated row for its DAG and robustness detail.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="border-b bg-muted/40 text-left text-xs uppercase text-muted-foreground">
                      <tr>
                        <th className="p-3 font-medium">#</th>
                        <th className="p-3 font-medium">Causal question</th>
                        <th className="p-3 font-medium">Brand</th>
                        <th className="p-3 font-medium">Confidence</th>
                        <th className="p-3 font-medium">Impact (ATE)</th>
                        <th className="p-3 font-medium">95% CI</th>
                        <th className="p-3 font-medium">Estimator</th>
                        <th className="p-3" />
                      </tr>
                    </thead>
                    <tbody>
                      {effects.map((e, i) => {
                        const clickable =
                          e.status === 'completed' ||
                          e.status === 'needs_review' ||
                          e.status === 'blocked';
                        const isSelected = !!e.analysis_id && e.analysis_id === selectedId;
                        return (
                          <tr
                            key={`${e.treatment}->${e.outcome}->${e.brand ?? 'all'}`}
                            className={`border-b last:border-0 ${
                              clickable ? 'cursor-pointer hover:bg-muted/40' : 'opacity-80'
                            } ${isSelected ? 'bg-muted/60' : ''}`}
                            onClick={() => {
                              if (clickable && e.analysis_id) setSelectedId(e.analysis_id);
                            }}
                          >
                            <td className="p-3 text-muted-foreground">{i + 1}</td>
                            <td className="p-3 font-medium">
                              <span>{e.treatment}</span>{' '}
                              <span className="text-muted-foreground">&rarr;</span>{' '}
                              <span>{e.outcome}</span>
                              {e.summary && (
                                <div className="mt-1 max-w-md text-xs font-normal text-muted-foreground">
                                  {e.summary}
                                </div>
                              )}
                            </td>
                            <td className="p-3 text-muted-foreground">{e.brand ?? 'All'}</td>
                            <td className="p-3">{verdictBadge(e)}</td>
                            <td className="p-3 font-medium">{formatEffect(e.ate)}</td>
                            <td className="p-3 text-muted-foreground">
                              {formatCI(e.ate_ci_lower, e.ate_ci_upper)}
                            </td>
                            <td className="p-3 capitalize">
                              {e.selected_estimator ? e.selected_estimator.replace(/_/g, ' ') : '—'}
                            </td>
                            <td className="p-3 text-right">
                              {clickable && (
                                <ChevronRight className="inline h-4 w-4 text-muted-foreground" />
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="border-t px-3 py-3 text-xs text-muted-foreground">
                  Why these {effects.length} question{effects.length === 1 ? '' : 's'}? The agent
                  only proposes the treatment&rarr;outcome relationships this grain&rsquo;s
                  gold-standard causal spec defines, scoped per brand, and collapses complementary
                  outcomes (e.g. &ldquo;discontinued&rdquo; is the inverse of
                  &ldquo;persistent&rdquo;) and self-pairs. Clinical markers (eGFR, LDH, …) are
                  designated adjustment covariates, not treatments or outcomes, so they enter the
                  model as confounders rather than as questions.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Drill-down detail for the selected effect */}
          {selectedId && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Network className="h-5 w-5" />
                  {detailResult ? (
                    <>
                      {detailResult.treatment_var} &rarr; {detailResult.outcome_var}
                    </>
                  ) : (
                    'Loading effect…'
                  )}
                </CardTitle>
                <CardDescription>
                  The agent&apos;s validated causal model: discovered DAG, estimated effect, and
                  robustness gate.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {detail.isLoading || !detailResult ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" /> Loading the validated analysis…
                  </div>
                ) : (
                  <CausalAnalysisDetail result={detailResult} />
                )}
              </CardContent>
            </Card>
          )}

          {/* Secondary: Pose your own question (manual path) */}
          <Card>
            <CardHeader>
              <button
                type="button"
                onClick={() => setManualOpen((o) => !o)}
                className="flex w-full items-center justify-between text-left"
                aria-expanded={manualOpen}
              >
                <div>
                  <CardTitle>Pose your own question</CardTitle>
                  <CardDescription>
                    Power users: pick a treatment + outcome from the gold-standard frame and run the
                    agent on that single hypothesis. Segmentation and method are decided by the
                    engine, not set by hand.
                  </CardDescription>
                </div>
                <ChevronRight
                  className={`h-5 w-5 shrink-0 text-muted-foreground transition-transform ${manualOpen ? 'rotate-90' : ''}`}
                />
              </button>
            </CardHeader>
            {manualOpen && (
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Treatment variable</label>
                    <Select value={treatmentVar} onValueChange={setTreatmentVar}>
                      <SelectTrigger aria-label="Treatment variable">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {treatmentCandidates.map((c) => (
                          <SelectItem key={c} value={c}>
                            {c}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">Outcome variable</label>
                    <Select value={outcomeVar} onValueChange={setOutcomeVar}>
                      <SelectTrigger aria-label="Outcome variable">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {outcomeCandidates.map((c) => (
                          <SelectItem key={c} value={c}>
                            {c}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Estimator <span className="text-muted-foreground">(optional override)</span>
                    </label>
                    <Select value={estimator} onValueChange={setEstimator}>
                      <SelectTrigger aria-label="Estimator">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {ESTIMATOR_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {confounders.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Controlling for (confounders, data-driven):{' '}
                    <span className="font-medium">{confounders.join(', ')}</span>
                  </p>
                )}
                <div className="flex items-center gap-3">
                  <Button onClick={handleRunManual} disabled={runAgent.isPending}>
                    <Play className="mr-2 h-4 w-4" />
                    {runAgent.isPending ? 'Running…' : 'Run analysis'}
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    Scoped to: {brandArg ?? 'All brands'} · {activeGrain.label} grain
                  </span>
                </div>

                {runAgent.isError && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Analysis could not run</AlertTitle>
                    <AlertDescription>
                      {runAgent.error?.message ? `${runAgent.error.message} ` : ''}
                      The causal agent is fail-closed: it estimates on real gold-standard data and
                      will not fabricate an effect. Try a different treatment / outcome pairing.
                    </AlertDescription>
                  </Alert>
                )}

                {runAgent.isPending && (
                  <Alert className="border-blue-200 bg-blue-50">
                    <Activity className="h-4 w-4 text-blue-600 animate-pulse" />
                    <AlertTitle className="text-blue-800">Analyzing…</AlertTitle>
                    <AlertDescription className="text-blue-700">
                      The agent is building the causal DAG, selecting an estimator across the
                      registry, and running robustness checks. This can take a minute or two.
                    </AlertDescription>
                  </Alert>
                )}

                {manualResult && <CausalAnalysisDetail result={manualResult} />}
              </CardContent>
            )}
          </Card>
        </TabsContent>

        {/* Estimators Tab */}
        <TabsContent value="estimators" className="space-y-6">
          <div className="flex gap-4 mb-4">
            <Select value={selectedLibrary} onValueChange={setSelectedLibrary}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by library" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Libraries</SelectItem>
                <SelectItem value="econml">EconML</SelectItem>
                <SelectItem value="causalml">CausalML</SelectItem>
                <SelectItem value="dowhy">DoWhy</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {estimatorsError ? (
            <EmptyState
              icon={<AlertTriangle className="h-8 w-8" aria-hidden="true" />}
              title="Estimator registry unavailable"
              description="Could not load the estimator registry from /causal/estimators. Try refreshing."
            />
          ) : estimatorsLoading ? (
            <EmptyState
              icon={<Activity className="h-8 w-8" aria-hidden="true" />}
              title="Loading estimators…"
              description="Fetching the supported estimator registry."
            />
          ) : visibleEstimators.length === 0 ? (
            <EmptyState
              icon={<GitBranch className="h-8 w-8" aria-hidden="true" />}
              title="No estimators for this library"
              description="No registered estimators match the selected library filter."
            />
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {visibleEstimators.map((est) => (
                <Card key={est.name} className="hover:shadow-md transition-shadow">
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-base capitalize">
                        {est.name.replace(/_/g, ' ')}
                      </CardTitle>
                      <Badge
                        style={{ backgroundColor: LIBRARY_COLORS[est.library] }}
                        className="text-white"
                      >
                        {est.library}
                      </Badge>
                    </div>
                    <CardDescription>{est.estimator_type}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <p className="text-sm text-muted-foreground">{est.description}</p>
                    <div className="flex gap-4 text-sm">
                      <div className="flex items-center gap-1">
                        {est.supports_confidence_intervals ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-gray-400" />
                        )}
                        <span>CI</span>
                      </div>
                      <div className="flex items-center gap-1">
                        {est.supports_heterogeneous_effects ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-gray-400" />
                        )}
                        <span>HTE</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Analysis History</CardTitle>
              <CardDescription>
                Recent completed causal analyses, newest first
                {historyData ? ` (${historyData.total})` : ''}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Loading analysis history…
                </div>
              ) : historyError ? (
                <EmptyState
                  title="Could not load analysis history"
                  description="A server error occurred while reading recent analyses. Try refreshing the page."
                />
              ) : historyData && historyData.items.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Completed</TableHead>
                      <TableHead>Summary</TableHead>
                      <TableHead>Agent</TableHead>
                      <TableHead className="text-right">ATE</TableHead>
                      <TableHead className="text-right">Confidence</TableHead>
                      <TableHead>Model</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {historyData.items.map((item) => (
                      <TableRow key={item.memory_id}>
                        <TableCell className="whitespace-nowrap text-sm">
                          {new Date(item.occurred_at).toLocaleString()}
                        </TableCell>
                        <TableCell className="max-w-md truncate text-sm">
                          {item.description ?? '—'}
                        </TableCell>
                        <TableCell className="text-sm">{item.agent_name ?? '—'}</TableCell>
                        <TableCell className="text-right text-sm">
                          {formatEffect(item.ate_estimate)}
                        </TableCell>
                        <TableCell className="text-right text-sm">
                          {item.confidence === null || item.confidence === undefined
                            ? 'N/A'
                            : `${(item.confidence * 100).toFixed(0)}%`}
                        </TableCell>
                        <TableCell className="text-sm">{item.model_used ?? '—'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <EmptyState
                  title="No analyses recorded yet"
                  description="Completed causal analyses will appear here as they are run."
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/pages/CausalAnalysis.test.tsx`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/CausalAnalysis.tsx frontend/src/pages/CausalAnalysis.test.tsx
git commit -m "feat(causal-fe): unify into one agent-led page (leaderboard landing + grain/brand facets + manual panel)"
```

---

### Task 4: Retire `/causal-discovery` (redirect) and delete the merged page

`/causal-discovery` now redirects to `/causal-analysis` and is removed from the sidebar nav. The old page + its test are deleted. The `routes.test.ts` IA-section expectation drops "Causal Discovery"; a new redirect test locks the behavior. (`_smoke.spec.ts` still hits `/causal-discovery` and expects `< 400` — a client-side `<Navigate>` keeps the SPA HTML 200, so it stays green.)

**Files:**
- Modify: `frontend/src/router/routes.tsx:12` (drop `CausalDiscovery` lazy import), `:77-84` (drop `routeConfigs` entry), `:338-347` (redirect element)
- Modify: `frontend/src/router/routes.test.ts:38-48` (causal IA section)
- Create: `frontend/src/router/routes.redirect.test.tsx`
- Delete: `frontend/src/pages/CausalDiscovery.tsx`, `frontend/src/pages/CausalDiscovery.test.tsx`

- [ ] **Step 1: Write the failing tests**

First, create the redirect test:

```tsx
// frontend/src/router/routes.redirect.test.tsx
import { describe, it, expect } from 'vitest';
import type { RouteObject } from 'react-router-dom';
import { Navigate } from 'react-router-dom';
import { routes, getNavigationRoutes } from './routes';

function findRoute(path: string): RouteObject | undefined {
  return routes.find((r) => r.path === path);
}

describe('/causal-discovery retirement (unified into /causal-analysis)', () => {
  it('still routes /causal-discovery (so the smoke spec gets HTML, not a 404)', () => {
    expect(findRoute('/causal-discovery')).toBeDefined();
  });

  it('redirects /causal-discovery to /causal-analysis (not NotFound)', () => {
    const route = findRoute('/causal-discovery');
    const el = route?.element as React.ReactElement;
    expect(el.type).toBe(Navigate);
    expect((el.props as { to: string }).to).toBe('/causal-analysis');
    expect((el.props as { replace?: boolean }).replace).toBe(true);
  });

  it('drops /causal-discovery from the sidebar nav (no dead link)', () => {
    const navPaths = getNavigationRoutes().map((r) => r.path);
    expect(navPaths).not.toContain('/causal-discovery');
    expect(navPaths).toContain('/causal-analysis');
  });
});
```

Then update the existing `routes.test.ts` causal IA expectation (it currently lists "Causal Discovery" first). Replace the `orders Causal Analytics by the analytical workflow` test body's array:

```typescript
  it('orders Causal Analytics by the analytical workflow', () => {
    const causal = getNavigationSections().find((s) => s.key === 'causal');
    expect(causal?.routes.map((r) => r.title)).toEqual([
      'Knowledge Graph',
      'Causal Analysis',
      'Intervention Impact',
      'Segment Analysis',
      'Expert Reviews',
    ]);
  });
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `cd frontend && npx vitest run src/router/routes.redirect.test.tsx src/router/routes.test.ts`
Expected: FAIL — `routes.redirect.test.tsx`: `/causal-discovery`'s element is the lazy `CausalDiscovery` page (not `Navigate`), and the nav still contains `/causal-discovery`; `routes.test.ts`: the causal section still starts with "Causal Discovery".

- [ ] **Step 3: Apply the route + nav changes**

In `frontend/src/router/routes.tsx`:

(a) Add `Navigate` to the react-router import and drop the `CausalDiscovery` lazy import. Replace line 2:

```typescript
import type { RouteObject } from 'react-router-dom';
import { Navigate } from 'react-router-dom';
```

and delete line 12 (`const CausalDiscovery = lazy(() => import('@/pages/CausalDiscovery'));`).

(b) Delete the `/causal-discovery` entry from `routeConfigs` (the object at lines 77-84):

```typescript
  {
    path: '/causal-discovery',
    title: 'Causal Discovery',
    description: 'Causal analysis and DAG visualization',
    icon: 'git-branch',
    section: 'causal',
    showInNav: true,
  },
```

(c) Replace the `/causal-discovery` route element (lines 338-347) with a redirect:

```typescript
  // /causal-discovery retired — unified into the agent-led /causal-analysis.
  // Kept as a redirect so bookmarks + the e2e smoke spec resolve (not a 404).
  {
    path: '/causal-discovery',
    element: <Navigate to="/causal-analysis" replace />,
  },
```

- [ ] **Step 4: Delete the merged page + its test**

```bash
git rm frontend/src/pages/CausalDiscovery.tsx frontend/src/pages/CausalDiscovery.test.tsx
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/router/routes.redirect.test.tsx src/router/routes.test.ts src/router/routes.test.tsx`
Expected: PASS (redirect lock + IA grouping + recovery routes all green; no lingering import of the deleted page).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/router/routes.tsx frontend/src/router/routes.test.ts frontend/src/router/routes.redirect.test.tsx
git commit -m "feat(causal-fe): retire /causal-discovery (redirect to /causal-analysis) + drop nav link"
```

---

### Task 5: Realign the e2e page-objects + specs (page rewrite desyncs POM `getByText` locators)

A page rewrite desyncs the e2e page-object exact-text locators (known trap). The causal-analysis POM still asserts the old "Run Analysis" header button + "No analysis run yet" landing; the page now lands on the leaderboard. The causal-discovery POM/spec must become a redirect assertion. Grep the POMs and realign.

**Files:**
- Modify: `frontend/e2e/fixtures/test-data.ts:98-114` (add `CAUSAL_ANALYSIS`)
- Modify: `frontend/e2e/pages/causal-analysis.page.ts`
- Modify: `frontend/e2e/pages/causal-discovery.page.ts`
- Modify: `frontend/e2e/specs/causal-discovery.spec.ts`
- Modify: `frontend/e2e/specs/causal-analysis.spec.ts`

- [ ] **Step 1: Grep the POM locators against the rewritten page (find the desync)**

Run:
```bash
cd frontend && grep -n "getByText\|getByRole\|getByLabel" e2e/pages/causal-analysis.page.ts e2e/pages/causal-discovery.page.ts
grep -rn "No analysis run yet\|Run Analysis\|No discovery run yet\|Discover causal effects\|Pose your own question\|Validated effects" src/pages/CausalAnalysis.tsx
```
Expected: confirms the POM still references the old `runAnalysisButton` (`Run Analysis`) header + `No analysis run yet` empty state, while the rewritten page has `Discover causal effects`, `No discovery run yet`, `Pose your own question`, and the `Run analysis` button inside the manual panel. This is the desync to fix.

- [ ] **Step 2: Add `CAUSAL_ANALYSIS` to the ROUTES fixture**

In `frontend/e2e/fixtures/test-data.ts`, in the `ROUTES` object (after the `CAUSAL_DISCOVERY` line), add:

```typescript
  CAUSAL_ANALYSIS: '/causal-analysis',
```

- [ ] **Step 3: Realign the causal-analysis POM to the unified page**

Replace `frontend/e2e/pages/causal-analysis.page.ts` with:

```typescript
import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the unified Causal Analysis page (`/causal-analysis`).
 *
 * ONE agent-led page (the former /causal-discovery is now a redirect here). The
 * LANDING is the validated-effects leaderboard: the analyst clicks "Discover
 * causal effects" and the causal_impact agent validates + ranks each candidate
 * question. A secondary "Pose your own question" panel keeps the manual
 * treatment/outcome path. Honest states this POM exposes:
 *  - header + "Agent-driven" badge
 *  - healthy banner: "Causal Engine Healthy" / degraded: "Service Issue"
 *  - the leaderboard "Discover causal effects" run control + grain/brand facets
 *  - empty: EmptyState "No discovery run yet" before a run
 *  - the "Pose your own question" manual panel trigger
 */
export class CausalAnalysisPage extends BasePage {
  readonly url = '/causal-analysis'
  readonly pageTitle = /Causal Analysis|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Causal Analysis/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Causal Analysis/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/ranks them by confidence and impact/i).first()
  }

  get agentDrivenBadge(): Locator {
    return this.page.getByText('Agent-driven', { exact: false }).first()
  }

  get healthyBanner(): Locator {
    return this.page.getByText('Causal Engine Healthy', { exact: true }).first()
  }

  get serviceIssueBanner(): Locator {
    return this.page.getByText('Service Issue', { exact: true }).first()
  }

  get librariesCard(): Locator {
    return this.page.getByText('Libraries', { exact: true }).first()
  }

  get estimatorsCard(): Locator {
    return this.page.getByText('Estimators', { exact: true }).first()
  }

  // The leaderboard run control (landing).
  get discoverButton(): Locator {
    return this.page.getByRole('button', { name: /Discover causal effects/i }).first()
  }

  get grainSelect(): Locator {
    return this.page.getByLabel('Grain')
  }

  get brandSelect(): Locator {
    return this.page.getByLabel('Brand')
  }

  // Honest empty state on the leaderboard before a run.
  get emptyState(): Locator {
    return this.page.getByText('No discovery run yet', { exact: true }).first()
  }

  // The secondary manual path trigger.
  get poseYourOwnQuestion(): Locator {
    return this.page.getByRole('button', { name: /Pose your own question/i }).first()
  }
}
```

- [ ] **Step 4: Repoint the causal-discovery POM + spec to the redirect**

Replace `frontend/e2e/pages/causal-discovery.page.ts` with:

```typescript
import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the RETIRED Causal Discovery route.
 *
 * `/causal-discovery` was unified into the agent-led `/causal-analysis` page and
 * is now a client-side redirect. This POM only asserts the redirect behavior;
 * the leaderboard + manual panel live on `CausalAnalysisPage`.
 */
export class CausalDiscoveryPage extends BasePage {
  readonly url = ROUTES.CAUSAL_DISCOVERY
  readonly pageTitle = /Causal Analysis|Causal Discovery|E2I/i

  constructor(page: Page) {
    super(page)
  }

  // After the redirect, the unified page's header renders.
  get redirectedHeader(): Locator {
    return this.page.getByRole('heading', { name: /Causal Analysis/i }).first()
  }
}
```

Replace `frontend/e2e/specs/causal-discovery.spec.ts` with:

```typescript
import { test, expect } from '@playwright/test'
import { CausalDiscoveryPage } from '../pages/causal-discovery.page'
import { harnessBase } from '../fixtures/page-harness'

/**
 * `/causal-discovery` is retired — unified into the agent-led `/causal-analysis`
 * page. This spec asserts the redirect: visiting the old route lands on
 * `/causal-analysis` with its header (NOT a 404 / NotFound). The leaderboard +
 * manual panel behavior is covered by causal-analysis.spec.ts and the component
 * unit tests.
 */
test.describe('Causal Discovery (retired → redirect)', () => {
  test('redirects /causal-discovery to /causal-analysis', async ({ page }) => {
    await harnessBase(page)
    const causalPage = new CausalDiscoveryPage(page)
    await page.goto(causalPage.url)
    await page.waitForLoadState('networkidle')
    await expect(page).toHaveURL(/causal-analysis/)
    await expect(causalPage.redirectedHeader).toBeVisible()
  })
})
```

- [ ] **Step 5: Realign the causal-analysis spec to the unified page**

Replace the `Healthy service` describe block's content tests + add leaderboard/manual assertions. In `frontend/e2e/specs/causal-analysis.spec.ts`, replace the `'shows honest empty state before any analysis is run'` test and add two more, inside the `Healthy service` describe:

```typescript
    test('lands on the leaderboard with an honest empty state before a run', async () => {
      await expect(causalPage.emptyState).toBeVisible()
    })

    test('shows the Discover causal effects run control + grain/brand facets', async () => {
      await expect(causalPage.discoverButton).toBeVisible()
      await expect(causalPage.grainSelect).toBeVisible()
      await expect(causalPage.brandSelect).toBeVisible()
    })

    test('keeps the secondary "Pose your own question" manual panel', async () => {
      await expect(causalPage.poseYourOwnQuestion).toBeVisible()
    })
```

(The existing `loads at /causal-analysis`, `displays the page header`, `renders the healthy service banner`, `renders the analyses-24h count`, `renders the KPI overview cards`, and the degraded-service falsifiability test stay as-is — they match the rebuilt page's retained health banner + KPI cards. The `pageDescription` locator in the POM was updated in Step 3 to the new copy.)

- [ ] **Step 6: Run the e2e type/lint check (the full e2e suite runs in CI against a served bundle)**

Run:
```bash
cd frontend && npx tsc -p tsconfig.json --noEmit && npx eslint e2e/pages/causal-analysis.page.ts e2e/pages/causal-discovery.page.ts e2e/specs/causal-analysis.spec.ts e2e/specs/causal-discovery.spec.ts
```
Expected: clean (no type errors from the POM/spec edits; no lint errors). The Playwright runs themselves are a CI gate (served prod bundle); locally just typecheck + lint the touched e2e files.

- [ ] **Step 7: Commit**

```bash
git add frontend/e2e/fixtures/test-data.ts frontend/e2e/pages/causal-analysis.page.ts frontend/e2e/pages/causal-discovery.page.ts frontend/e2e/specs/causal-analysis.spec.ts frontend/e2e/specs/causal-discovery.spec.ts
git commit -m "test(causal-fe): realign e2e POMs + specs to the unified page + /causal-discovery redirect"
```

---

## Verification (whole plan)

- [ ] Unit tests — all green:
  ```bash
  cd frontend && npx vitest run \
    src/types/causal.types.test.ts \
    src/components/causal/CausalAnalysisDetail.test.tsx \
    src/pages/CausalAnalysis.test.tsx \
    src/router/routes.redirect.test.tsx \
    src/router/routes.test.ts \
    src/router/routes.test.tsx
  ```
- [ ] No dangling reference to the deleted page:
  ```bash
  cd frontend && ! grep -rn "pages/CausalDiscovery'" src/ && echo "no CausalDiscovery page import remains"
  ```
- [ ] Type check (whole FE) — clean: `cd frontend && npx tsc -p tsconfig.json --noEmit`
- [ ] Lint — clean: `cd frontend && npx eslint src/pages/CausalAnalysis.tsx src/components/causal/CausalAnalysisDetail.tsx src/router/routes.tsx`
- [ ] Full FE unit suite (CI parity): `cd frontend && npx vitest run` — green (no other test imports the deleted page; `routes.test.tsx` recovery-route file untouched).
- [ ] **Faithful live run** (after #1030 + P1 merged + deployed): authenticate, open `/causal-analysis` → lands on the leaderboard; submit discover-effects for brand=Kisqali (Patient grain) → rows fill in, each showing its **brand** + **summary**; click a `proceed` row → the deep view renders the connected DAG + per-test refutation + estimator-comparison panel + interpretation; expand "Pose your own question" → run `treatment_arm → persistent_180d` → the same deep view renders. Then hit `/causal-discovery` directly → browser lands on `/causal-analysis`. Confirm via `docker logs e2i_api` that the served requests are `POST /causal/discover-effects?dataset=patient_journeys&brand=Kisqali` and `GET /causal/agent-analyze/{id}` (not double-wrapped params).
- [ ] Adversarial multi-lens review before PR (the redesign has repeatedly surfaced CI-passing honesty/presentation bugs here).

## Self-Review (done)

- **Spec coverage (§5.1):** route `/causal-analysis` kept + `/causal-discovery` redirect (Task 4); landing = agent-led leaderboard with **grain** + **brand** facets (Task 3); row → existing deep view incl. #1030 estimator-comparison panel (Tasks 2+3, via `CausalAnalysisDetail`); "pose your own question" secondary panel sourced from `/variables` (Task 3); each row surfaces **brand** + **summary** (Task 3 leaderboard columns, consuming the P1 fields added in Task 1). Grain facet UI is wired but HCP/Trigger are honestly disabled until P2/P3 land their datasets — in scope (the page shell); out of scope (the loaders), matching §9 phasing.
- **Cross-phase boundaries:** This plan adds NO backend fields — it consumes `DiscoveredEffect.brand`/`adjustment_set`/`summary` (P1) and `AgentCausalAnalysisResponse.estimator_comparison` (#1030). `discoverCausalEffects(dataset, brand)` and `useDiscoverEffects(dataset, brand)` already thread `dataset` (verified in the worktree), so the grain facet needs no hook/api change — only passing the grain's dataset string. No reinvented helpers: `verdictBadge`, `REFUTATION_METHOD_MAP`, `toRefutationResults`, and `EstimatorComparisonPanel` are lifted verbatim from the post-#1030 `CausalDiscovery.tsx`.
- **Placeholder scan:** none. Every code/test step is complete and grounded in the actual post-#1030 files (real component/hook/api/type signatures quoted: `useDiscoverEffects`, `getCausalAgentAnalysis`, `CausalDiscovery as CausalDiscoveryViz`, `KPICard`, `EmptyState`, Radix `Select`/`Tabs`/`Table`, `Navigate`, `getNavigationRoutes`). Grep step in Task 5 is a real desync-finder, not a placeholder.
- **Type consistency:** `CausalAnalysisDetail` takes `result: AgentCausalAnalysisResponse` (the exact type both the drill-down `getCausalAgentAnalysis` and the manual `runAgent.data` produce). `DiscoveredEffect.brand`/`adjustment_set` added in Task 1 are consumed in Task 3's leaderboard (`e.brand`); `summary` (already in the type post-#1030) consumed too. The redirect element type is asserted as `Navigate` in Task 4. Vitest mock of `@/components/causal/CausalAnalysisDetail` in the page test isolates the page from the detail's internals (which its own test covers) — no duplicate assertions.
- **CI traps pre-empted:** (1) page-rewrite POM desync handled explicitly (Task 5 grep + realign) — the known repeated failure mode. (2) `_smoke.spec.ts` hits `/causal-discovery` expecting `< 400`; a client-side `<Navigate>` serves the SPA HTML (200) then redirects → stays green (no testIgnore/quarantine edit needed). (3) `home.spec.ts` "navigate to Causal Discovery via link" asserts only `url contains 'causal'` and is wrapped in an `isVisible()` guard; the sidebar link is now "Causal Analysis" (`/causal-analysis`) → still contains 'causal' → green. (4) `routes.test.ts` IA expectation updated in lockstep (Task 4) so dropping the nav entry doesn't red the grouping test. (5) Deleted-page import scan in Verification prevents a stale-import build break.
