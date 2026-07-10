# Documentation Page (`/documentation`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an illustrated, interactive `/documentation` page ("Understanding E2I") explaining the platform's purpose, methodology, best practices, and expected impact; link it from the footer (4th link) and the sidebar ("Data & Reference"); delete the stale `AgenticMethodology.tsx` it supersedes.

**Architecture:** A lazy-loaded React page composed of small presentational components under `frontend/src/components/documentation/`, all content in typed constants (`content.ts`). The page makes exactly ONE network call (`useKPIList`) for a live KPI-count chip that silently disappears on error. The per-page CapabilityIndex derives its groups from `getNavigationSections()` in `routes.tsx` at runtime, so it can never advertise a retired page; a content-invariant test enforces that every nav page has a capability question.

**Tech Stack:** React 18 + TypeScript (Vite), react-router-dom v7, TanStack Query (`useKPIList` only), Radix Collapsible (`@/components/ui/collapsible`), lucide-react icons, Tailwind v4 CSS-var tokens. Tests: vitest + React Testing Library (co-located), Playwright e2e.

**Spec:** `docs/superpowers/specs/2026-07-10-documentation-page-design.md`

---

## Plan decisions (deviations from spec, with reasoning)

1. **CSS transitions instead of framer-motion.** framer-motion is installed but used only by chat components (`E2IChatSidebar`, `E2IChatPopup`). Every animation here (expand/collapse, highlight) is achievable with Tailwind `transition-*` classes plus the built-in `motion-reduce:` variant, which satisfies the spec's reduced-motion requirement with zero added chunk weight and no jsdom animation flakiness. The spec's intent (interactive, reduced-motion-safe, no new deps) is fully preserved.
2. **CapabilityIndex derives groups from the router.** The spec says card copy "is authored at implementation time from each page's routeConfig". We go one step further: the component maps over `getNavigationSections()` directly and looks up per-path questions from `CAPABILITY_QUESTIONS` in `content.ts`. Titles, grouping, and page membership can never drift from the sidebar; only the question text is maintained content, and `content.test.ts` fails the build if a nav page lacks a question (or a question references a dead path).
3. **Component behavior tests live in `Documentation.test.tsx`** (per spec's testing section), not one test file per component. `content.test.ts` holds the anti-drift invariants. `Footer.test.tsx` is new and separate.

## Environment notes (read before starting)

- Frontend commands run from `/home/enunez/Projects/e2i_causal_analytics/frontend`. Git commands from the repo root.
- Type check is `npx tsc -p tsconfig.app.json` — bare `npx tsc --noEmit` is a known FALSE GREEN in this repo.
- Run only targeted vitest files locally (this box is the prod droplet; CI runs the full suite). vitest 4: no `basic` reporter — use default.
- There is NO prettier gate — do not run `prettier --write`.
- Do not push to `main` directly: pushing `main` triggers the ~12-min production deploy. All work happens on the feature branch until the PR merges.
- Before any `git push`/`gh` call: `git config --global http.https://github.com.proxy ""` (corporate proxy bypass).

## File structure

```
frontend/src/pages/Documentation.tsx              page shell: header, SectionNav, 4 sections, stat chips
frontend/src/pages/Documentation.test.tsx         page + component behavior tests
frontend/src/components/documentation/
  index.ts                                        barrel export
  content.ts                                      ALL typed content constants
  content.test.ts                                 anti-drift invariants
  SectionNav.tsx                                  sticky scroll-spy nav
  CausalScopeMap.tsx                              §1 three-level capability map
  CorrelationCausationToggle.tsx                  §1 correlation-vs-causation illustration
  CapabilityIndex.tsx                             §1 per-page capability grid
  CausalPipeline.tsx                              §2 5-stage pipeline
  AgentTierStack.tsx                              §2 6-tier / 21-agent stack
  ClinicalGrounding.tsx                           §2 clinical knowledge sources strip
  PracticeCards.tsx                               §3 do/don't cards with role filter
  ImpactPathways.tsx                              §4 impact pathway cards
frontend/src/components/layout/Footer.tsx         MODIFY: add 4th link
frontend/src/components/layout/Footer.test.tsx    NEW: footer link tests
frontend/src/components/layout/Sidebar.tsx        MODIFY: add 'graduation-cap' icon
frontend/src/router/routes.tsx                    MODIFY: lazy import + routeConfig + RouteObject
frontend/src/components/kpi/AgenticMethodology.tsx  DELETE
frontend/src/components/kpi/index.ts              MODIFY: update superseded comment
frontend/e2e/fixtures/test-data.ts                MODIFY: add DOCUMENTATION route
frontend/e2e/pages/documentation.page.ts          NEW: page object
frontend/e2e/specs/documentation.spec.ts          NEW: e2e spec
```

Content facts were verified against the backend on 2026-07-10 (do not "improve" them without re-verifying):
- 21 agents / 6 tiers: `src/agents/factory.py` `AGENT_REGISTRY_CONFIG` (Tier 0 has 8 incl. `cohort_constructor`)
- 5 refutation tests: `src/api/schemas/causal.py` (placebo_treatment, random_common_cause, data_subset, bootstrap, unobserved_common_cause)
- 8 intervention channels: `src/digital_twin/effect/provider.py` `INTERVENTION_CATALOG`
- 4 predictive cohorts: `src/insights/predictive_cohort.py` (hcp_adoption, initiation, persistence, discontinuation)
- 3 brands / 4 indications: `src/agents/cohort_constructor/configs.py` (Remibrutinib CSU, Fabhalta PNH, Fabhalta C3G, Kisqali HR+/HER2− BC)
- Scope-map node labels: `src/insights/causal_context.py` `_NODE_LABELS`
- Clinical sources: UMLS `src/data/kg/umls_uts.py`; OpenFDA/ClinicalTrials.gov/PubMed/ChEMBL `src/services/clinical_context/`

---

### Task 1: Feature branch

**Files:** none

- [ ] **Step 1: Create the branch from up-to-date main**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git checkout main && git pull --ff-only
git checkout -b feat/documentation-page
```

Expected: `Switched to a new branch 'feat/documentation-page'`. If the tree is dirty, STOP and report — do not stash around an in-flight deploy.

---

### Task 2: Content constants + anti-drift invariants (`content.ts`)

**Files:**
- Create: `frontend/src/components/documentation/content.ts`
- Test: `frontend/src/components/documentation/content.test.ts`

- [ ] **Step 1: Write the failing invariant test**

Create `frontend/src/components/documentation/content.test.ts`:

```ts
/**
 * Anti-drift invariants for Documentation page content.
 * These tests are tripwires: they fail when the platform changes shape
 * (new nav page, retired page, agent roster change) without the
 * Documentation content being updated — the exact failure mode that made
 * the old AgenticMethodology component go stale.
 */
import { describe, it, expect } from 'vitest';
import { getNavigationRoutes } from '@/router/routes';
import {
  AGENT_TIERS,
  CAPABILITY_EXEMPT_PATHS,
  CAPABILITY_QUESTIONS,
  CLINICAL_SOURCES,
  IMPACT_PATHWAYS,
  PIPELINE_STAGES,
  PRACTICES,
  SCOPE_LEVELS,
  STAT_CHIPS,
} from './content';

describe('content invariants', () => {
  it('covers every nav page with a capability question (no gaps)', () => {
    const navPaths = getNavigationRoutes().map((r) => r.path);
    const missing = navPaths.filter(
      (p) => !(CAPABILITY_EXEMPT_PATHS as readonly string[]).includes(p) && !(p in CAPABILITY_QUESTIONS)
    );
    expect(missing).toEqual([]);
  });

  it('has no capability question for a dead path (no orphans)', () => {
    const navPaths = new Set(getNavigationRoutes().map((r) => r.path));
    const orphans = Object.keys(CAPABILITY_QUESTIONS).filter((p) => !navPaths.has(p));
    expect(orphans).toEqual([]);
  });

  it('models exactly 21 agents across 6 tiers with unique ids', () => {
    expect(AGENT_TIERS).toHaveLength(6);
    const ids = AGENT_TIERS.flatMap((t) => t.agents.map((a) => a.id));
    expect(ids).toHaveLength(21);
    expect(new Set(ids).size).toBe(21);
  });

  it('lists five clinical sources with UMLS and OpenFDA prominent', () => {
    expect(CLINICAL_SOURCES).toHaveLength(5);
    const prominent = CLINICAL_SOURCES.filter((s) => s.prominent).map((s) => s.name);
    expect(prominent).toEqual(['UMLS', 'OpenFDA']);
  });

  it('has 5 pipeline stages, 3 scope levels, 4 static chips, 4 impact pathways', () => {
    expect(PIPELINE_STAGES).toHaveLength(5);
    expect(SCOPE_LEVELS).toHaveLength(3);
    expect(STAT_CHIPS).toHaveLength(4);
    expect(IMPACT_PATHWAYS).toHaveLength(4);
    expect(PRACTICES.length).toBeGreaterThanOrEqual(5);
  });

  it('impact pathways contain no fabricated digits in their copy', () => {
    for (const p of IMPACT_PATHWAYS) {
      expect(p.mechanism).not.toMatch(/\d/);
      expect(p.title).not.toMatch(/\d/);
    }
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/components/documentation/content.test.ts
```

Expected: FAIL — `Cannot find module './content'` (or equivalent resolve error).

- [ ] **Step 3: Create `content.ts`**

Create `frontend/src/components/documentation/content.ts`:

```ts
/**
 * Documentation page content — typed constants.
 * =============================================
 *
 * STALENESS RULE (from the spec): structural facts only. Anything that
 * drifts weekly (KPI counts, table counts) is either fetched live on the
 * page or omitted. Facts below were verified against the backend on
 * 2026-07-10 — sources cited per block. content.test.ts enforces the
 * router-facing invariants.
 */

export type CausalLevel = 'hcp' | 'patient' | 'market';

export const LEVEL_LABELS: Record<CausalLevel, string> = {
  hcp: 'HCP',
  patient: 'Patient',
  market: 'Market',
};

// ── §1 Purpose ──────────────────────────────────────────────────────────────

export interface ScopeLevelDef {
  id: CausalLevel;
  title: string;
  summary: string;
  /** Humanized node labels drawn from src/insights/causal_context.py _NODE_LABELS. */
  nodes: string[];
}

export const SCOPE_LEVELS: ScopeLevelDef[] = [
  {
    id: 'hcp',
    title: 'HCP prescribing behavior',
    summary:
      'Which promotional levers actually change prescribing — rep detailing, speaker programs, sampling, peer influence, digital engagement — with eight intervention channels simulatable in the Digital Twin.',
    nodes: [
      'rep detailing frequency',
      'speaker program attendance',
      'sampling',
      'HCP coverage',
      'intent to prescribe',
    ],
  },
  {
    id: 'patient',
    title: 'Patient journey outcomes',
    summary:
      'What drives patients to start therapy, stay on it, or stop — treatment initiation, persistence, and discontinuation, each with its own predictive cohort.',
    nodes: [
      'treatment initiation',
      'patient persistence',
      'treatment discontinuation',
      'copay support program',
    ],
  },
  {
    id: 'market',
    title: 'Market & brand performance',
    summary:
      'How upstream behaviors and market dynamics — formulary status, competitor activity — aggregate into the brand outcomes executives track.',
    nodes: [
      'TRx volume',
      'NBRx volume',
      'TRx market share',
      'ROI',
      'formulary status',
      'competitor activity',
    ],
  },
];

export interface StatChip {
  value: string;
  label: string;
}

/** Static structural chips. The live KPI-count chip is fetched on the page. */
export const STAT_CHIPS: StatChip[] = [
  { value: '3 / 4', label: 'brands / indications' },
  { value: '4', label: 'predictive cohorts' },
  { value: '8', label: 'intervention channels' },
  { value: '5', label: 'refutation tests' },
];

export interface CapabilityInfo {
  question: string;
  levels?: CausalLevel[];
}

/**
 * Pages excluded from the capability index: the dashboard itself and this
 * documentation page.
 */
export const CAPABILITY_EXEMPT_PATHS = ['/', '/documentation'] as const;

/**
 * One question per nav page. Grouping/titles/links come from
 * getNavigationSections() at render time, so retired pages can never appear;
 * content.test.ts fails when a page is added or removed without updating this map.
 */
export const CAPABILITY_QUESTIONS: Record<string, CapabilityInfo> = {
  '/knowledge-graph': {
    question: 'How are HCPs, patients, brands, and outcomes connected — and what evidence backs each causal path?',
    levels: ['hcp', 'patient', 'market'],
  },
  '/causal-analysis': {
    question: 'What is the measured effect of an intervention on an outcome — and does it survive refutation?',
    levels: ['hcp', 'market'],
  },
  '/segment-analysis': {
    question: 'Who responds above or below average, and by how much?',
    levels: ['hcp', 'patient'],
  },
  '/expert-reviews': {
    question: 'Do human experts agree with the causal findings, and where do they push back?',
  },
  '/predictive-analytics': {
    question: 'Which patients or HCPs are most likely to initiate, persist, discontinue, or adopt?',
    levels: ['hcp', 'patient'],
  },
  '/model-performance': {
    question: 'Can we trust the models behind the predictions?',
  },
  '/feature-importance': {
    question: 'Which factors drive the model predictions, and in which direction?',
  },
  '/time-series': {
    question: 'How are the business KPIs trending over time, per brand?',
    levels: ['market'],
  },
  '/digital-twin': {
    question: 'What would happen if we ran this intervention — before we spend on it?',
    levels: ['hcp'],
  },
  '/gap-analysis': {
    question: 'Where are we underperforming relative to potential, and what is closing the gap worth?',
    levels: ['market'],
  },
  '/resource-optimization': {
    question: 'How should budget be allocated across channels for maximum causal impact?',
    levels: ['hcp', 'market'],
  },
  '/experiments': {
    question: 'How do we design and monitor field experiments to confirm an effect?',
  },
  '/ai-insights': {
    question: 'What do the agents conclude when they analyze the data end to end?',
  },
  '/kpi-dictionary': {
    question: 'What does each KPI mean, and how exactly is it calculated?',
  },
  '/data-quality': {
    question: 'Is the underlying data complete, consistent, and fresh enough to trust?',
  },
  '/system-health': {
    question: 'Are all platform services healthy right now?',
  },
  '/monitoring': {
    question: 'Are models drifting, and which alerts are firing?',
  },
  '/analytics': {
    question: 'How is the platform itself being used?',
  },
  '/agent-orchestration': {
    question: 'Which agents ran, in what order, and at what cost?',
  },
  '/memory-architecture': {
    question: 'How does the platform remember context across sessions and analyses?',
  },
  '/audit-chain': {
    question: 'Can every insight be traced back to its evidence?',
  },
  '/feedback-learning': {
    question: 'Does the system actually learn from user feedback?',
  },
};

// ── §2 Methodology ──────────────────────────────────────────────────────────

export interface PipelineStage {
  id: string;
  name: string;
  /** Plain-language summary (always shown when the stage is expanded). */
  plain: string;
  /** "For analysts" collapsible content. */
  analyst: string;
}

/** The causal pipeline. Refutation test names: src/api/schemas/causal.py. */
export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: 'frame',
    name: 'Frame',
    plain:
      'Turn a business question into a precise causal question: which intervention, on which population, affecting which outcome.',
    analyst:
      'The question is encoded as a directed acyclic graph (DAG) of treatment, outcome, and covariates. Cohorts are indication-specific eligible populations (e.g. Remibrutinib CSU, Fabhalta PNH/C3G, Kisqali HR+/HER2− BC) resolved through a single canonical cohort-loading path that fails closed rather than fabricating a population.',
  },
  {
    id: 'identify',
    name: 'Identify',
    plain:
      'Work out whether the question is answerable from the available data — and which variables must be adjusted for to avoid confounding.',
    analyst:
      'Backdoor adjustment over the DAG identifies the confounders to control (e.g. physician specialty confounding the calls→prescriptions relationship). If no valid adjustment set exists, the analysis stops here instead of producing a biased number.',
  },
  {
    id: 'estimate',
    name: 'Estimate',
    plain:
      'Measure the effect: on average (ATE), and for whom it differs (CATE) — so segments that respond above or below average become visible.',
    analyst:
      'Estimation runs through two independent libraries — EconML and CausalML — and the platform cross-validates their agreement before trusting a heterogeneous-effect result. Uplift models power segment-level expected-lift figures, which are gated: only segments whose effect is credibly above the average are surfaced as opportunities.',
  },
  {
    id: 'refute',
    name: 'Refute',
    plain:
      'Attack the estimate before believing it. Effects that fail these attacks are blocked or flagged for review — they never silently reach a recommendation.',
    analyst:
      'Five refutation tests: placebo treatment (fake treatment should show no effect), random common cause (adding noise confounders should not move the estimate), data subset (the effect should hold on subsamples), bootstrap (stability across resamples), and unobserved-common-cause sensitivity mapped to an E-value (how strong would a hidden confounder need to be to explain the effect away). Results feed a proceed / review / block gate.',
  },
  {
    id: 'act',
    name: 'Act',
    plain:
      'Only gated, refutation-tested effects flow into recommendations: budget allocation, segment targeting, experiment designs, and executive insights.',
    analyst:
      'Downstream surfaces (Resource Optimization, Gap Analysis, Digital Twin simulation, AI Insights) consume gated estimates with provenance labels. Narrative insight surfaces are digit-guarded: language models never invent figures, they interpret server-injected validated numbers.',
  },
];

export interface AgentDef {
  id: string;
  role: string;
}

export interface AgentTier {
  tier: number;
  name: string;
  blurb: string;
  agents: AgentDef[];
}

/** Roster source: src/agents/factory.py AGENT_REGISTRY_CONFIG (21 agents). */
export const AGENT_TIERS: AgentTier[] = [
  {
    tier: 0,
    name: 'ML Foundation',
    blurb: 'Builds and ships the models everything else relies on.',
    agents: [
      { id: 'scope_definer', role: 'Turns a request into a scoped ML problem' },
      { id: 'data_preparer', role: 'Assembles and validates training data' },
      { id: 'feature_analyzer', role: 'Selects and audits features' },
      { id: 'model_selector', role: 'Picks the algorithm for the problem' },
      { id: 'model_trainer', role: 'Trains and evaluates candidate models' },
      { id: 'model_deployer', role: 'Promotes models to serving' },
      { id: 'observability_connector', role: 'Wires telemetry for every run' },
      { id: 'cohort_constructor', role: 'Builds indication-specific eligible populations' },
    ],
  },
  {
    tier: 1,
    name: 'Coordination',
    blurb: 'Routes questions to the right specialists and composes tools.',
    agents: [
      { id: 'orchestrator', role: 'Routes each query to the right agents' },
      { id: 'tool_composer', role: 'Chains analysis tools into workflows' },
    ],
  },
  {
    tier: 2,
    name: 'Causal Analytics',
    blurb: 'The estimation core: effects, gaps, and heterogeneity.',
    agents: [
      { id: 'causal_impact', role: 'Estimates and refutes causal effects' },
      { id: 'gap_analyzer', role: 'Quantifies performance vs potential' },
      { id: 'heterogeneous_optimizer', role: 'Finds who responds differently (CATE)' },
    ],
  },
  {
    tier: 3,
    name: 'Monitoring',
    blurb: 'Watches models, experiments, and platform health.',
    agents: [
      { id: 'drift_monitor', role: 'Detects data and model drift' },
      { id: 'experiment_designer', role: 'Designs valid field experiments' },
      { id: 'experiment_monitor', role: 'Tracks running experiments' },
      { id: 'health_score', role: 'Scores end-to-end system health' },
    ],
  },
  {
    tier: 4,
    name: 'ML Predictions',
    blurb: 'Turns models into forward-looking answers.',
    agents: [
      { id: 'prediction_synthesizer', role: 'Combines model outputs into predictions' },
      { id: 'resource_optimizer', role: 'Allocates budget under constraints' },
    ],
  },
  {
    tier: 5,
    name: 'Self-Improvement',
    blurb: 'Explains results and learns from feedback.',
    agents: [
      { id: 'explainer', role: 'Produces evidence-grounded explanations' },
      { id: 'feedback_learner', role: 'Improves behavior from user feedback' },
    ],
  },
];

export interface ClinicalSource {
  name: string;
  role: string;
  prominent?: boolean;
}

/** Sources: src/data/kg/umls_uts.py and src/services/clinical_context/. */
export const CLINICAL_SOURCES: ClinicalSource[] = [
  {
    name: 'UMLS',
    role: 'Terminology backbone for knowledge-graph entity linking: concept search, CUI lookup, and ICD-10-CM / RxNorm / LOINC crosswalks.',
    prominent: true,
  },
  {
    name: 'OpenFDA',
    role: 'Official drug-label indications feeding the on-label gate, so insights stay inside approved labeling.',
    prominent: true,
  },
  {
    name: 'ClinicalTrials.gov',
    role: 'Real trial endpoints per brand and indication ground outcome definitions.',
  },
  {
    name: 'PubMed',
    role: 'Real-world-evidence literature citations attach to clinical claims.',
  },
  {
    name: 'ChEMBL',
    role: 'Mechanism-of-action context for each brand’s molecule.',
  },
];

// ── §3 Best Practices ───────────────────────────────────────────────────────

export type PracticeRole = 'exec' | 'analyst';

export interface Practice {
  id: string;
  doText: string;
  dontText: string;
  why: string;
  roles: PracticeRole[];
}

export const PRACTICES: Practice[] = [
  {
    id: 'refutation-gate',
    doText: 'Check the refutation gate (proceed / review / block) before acting on an estimated effect.',
    dontText: 'Treat every estimated effect as actionable just because it has a confidence interval.',
    why: 'An estimate that fails placebo or sensitivity tests is likely confounded; the gate exists to stop it from reaching a decision.',
    roles: ['exec', 'analyst'],
  },
  {
    id: 'informational-kpis',
    doText: 'Read "Informational" KPIs as context about the environment.',
    dontText: 'Manage teams against Informational KPIs as if they were performance targets.',
    why: 'KPIs without a defensible target are labeled Informational on purpose — inventing a target for them rewards gaming, not improvement.',
    roles: ['exec'],
  },
  {
    id: 'honest-null',
    doText: 'Treat "no significant effect" as a finding that saves money.',
    dontText: 'Rerun an analysis with different settings until an effect appears.',
    why: 'A credible null on a promotional lever means budget can move to levers that do work. Torturing the data until it confesses produces effects that will not replicate in the field.',
    roles: ['exec', 'analyst'],
  },
  {
    id: 'whatif-ranges',
    doText: 'Keep what-if simulation inputs inside the observed data ranges shown on each control.',
    dontText: 'Extrapolate simulations far beyond any scenario the models have seen.',
    why: 'Twin models interpolate well and extrapolate poorly; inputs outside observed ranges produce numbers with no evidential basis.',
    roles: ['analyst'],
  },
  {
    id: 'brand-scope',
    doText: 'Confirm the brand selector before comparing metrics across pages.',
    dontText: 'Compare numbers captured under different brand scopes.',
    why: 'Models, cohorts, and KPIs are per-brand; a metric from the wrong brand context looks plausible but answers a different question.',
    roles: ['exec', 'analyst'],
  },
];

// ── §4 Expected Impact ──────────────────────────────────────────────────────

export interface ImpactPathway {
  title: string;
  mechanism: string;
  href: string;
  linkLabel: string;
}

/**
 * Mechanism-focused, digit-free by design (enforced in content.test.ts):
 * the platform's honesty discipline forbids fabricated ROI figures. Each card
 * links to the live page where users see their own numbers.
 */
export const IMPACT_PATHWAYS: ImpactPathway[] = [
  {
    title: 'Sharper targeting',
    mechanism:
      'Heterogeneous-effect segments identify who responds above average, so field effort concentrates where it causally moves outcomes.',
    href: '/segment-analysis',
    linkLabel: 'See your segments',
  },
  {
    title: 'Better budget allocation',
    mechanism:
      'Constrained optimization reallocates spend across channels using gated causal effects instead of last-touch attribution.',
    href: '/resource-optimization',
    linkLabel: 'See your allocation',
  },
  {
    title: 'Cheaper experimentation',
    mechanism:
      'Digital-twin simulation pre-screens interventions, so expensive field pilots are reserved for the candidates most likely to work.',
    href: '/digital-twin',
    linkLabel: 'Run a simulation',
  },
  {
    title: 'Faster time-to-insight',
    mechanism:
      'Natural-language chat over governed KPIs and causal results answers questions in minutes that previously took an analyst request cycle.',
    href: '/',
    linkLabel: 'Open the dashboard',
  },
];

// ── Section nav ─────────────────────────────────────────────────────────────

export interface DocSection {
  id: string;
  label: string;
}

export const DOC_SECTIONS: DocSection[] = [
  { id: 'purpose', label: 'Purpose' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'practices', label: 'Best Practices' },
  { id: 'impact', label: 'Expected Impact' },
];
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/components/documentation/content.test.ts
```

Expected: PASS (6 tests). If the "covers every nav page" test fails, a nav page was added/removed since this plan was written — add/remove the matching `CAPABILITY_QUESTIONS` entry (do NOT loosen the test).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/content.ts frontend/src/components/documentation/content.test.ts
git commit -m "feat(docs-page): typed content constants with router anti-drift invariants"
```

---

### Task 3: Page shell + SectionNav

**Files:**
- Create: `frontend/src/components/documentation/SectionNav.tsx`
- Create: `frontend/src/components/documentation/index.ts`
- Create: `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx`

- [ ] **Step 1: Write the failing page test**

Create `frontend/src/pages/Documentation.test.tsx`:

```tsx
/**
 * Documentation Page Tests
 * ========================
 * Page shell + interactive component behaviors (per the spec's testing
 * section, component behavior tests live here; content invariants live in
 * components/documentation/content.test.ts).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Documentation from './Documentation';

// The page's ONLY network hook. Mocked in every test; the "live chip" tests
// flip its return value.
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
}));

import { useKPIList } from '@/hooks/api/use-kpi';

// jsdom has neither scrollIntoView nor IntersectionObserver.
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(useKPIList).mockReturnValue({
    data: { kpis: [], total: 46 },
    isLoading: false,
    isError: false,
  } as ReturnType<typeof useKPIList>);
});

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Documentation />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Documentation page shell', () => {
  it('renders the page header', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /understanding e2i/i })).toBeInTheDocument();
  });

  it('renders all four sections', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /^purpose/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^methodology/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^best practices/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /^expected impact/i })).toBeInTheDocument();
  });

  it('renders the section nav and scrolls on click', async () => {
    renderPage();
    const nav = screen.getByRole('navigation', { name: /on this page/i });
    expect(nav).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^methodology$/i }));
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

Expected: FAIL — cannot resolve `./Documentation`.

- [ ] **Step 3: Create `SectionNav.tsx`**

Create `frontend/src/components/documentation/SectionNav.tsx`:

```tsx
/**
 * SectionNav — sticky scroll-spy navigation for the Documentation page.
 * Scroll-spy uses IntersectionObserver, guarded so environments without it
 * (jsdom, very old browsers) degrade to click-to-scroll with no highlight.
 */
import { useEffect, useState } from 'react';
import type { DocSection } from './content';

interface SectionNavProps {
  sections: DocSection[];
}

export function SectionNav({ sections }: SectionNavProps) {
  const [activeId, setActiveId] = useState<string>(sections[0]?.id ?? '');

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActiveId(entry.target.id);
        }
      },
      // Trigger when a section's top crosses the upper third of the viewport.
      { rootMargin: '-20% 0px -70% 0px' }
    );
    for (const s of sections) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [sections]);

  return (
    <nav
      aria-label="On this page"
      className="sticky top-0 z-10 -mx-4 mb-8 overflow-x-auto border-b border-[var(--color-border)] bg-[var(--color-background)]/95 px-4 py-2 backdrop-blur"
    >
      <ul className="flex items-center gap-1">
        {sections.map((s) => (
          <li key={s.id}>
            <button
              type="button"
              onClick={() => document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
              aria-current={activeId === s.id ? 'true' : undefined}
              className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors motion-reduce:transition-none ${
                activeId === s.id
                  ? 'bg-[var(--color-primary)]/10 font-medium text-[var(--color-primary)]'
                  : 'text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]'
              }`}
            >
              {s.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
```

- [ ] **Step 4: Create the barrel export**

Create `frontend/src/components/documentation/index.ts`:

```ts
/**
 * Documentation page components.
 * Superseded frontend/src/components/kpi/AgenticMethodology.tsx (deleted) —
 * this directory is the current home of platform methodology content.
 */
export { SectionNav } from './SectionNav';
export * from './content';
```

(Components added in later tasks each append one `export` line here — the later tasks show the exact line.)

- [ ] **Step 5: Create the page shell**

Create `frontend/src/pages/Documentation.tsx`:

```tsx
/**
 * Documentation Page — "Understanding E2I"
 * ========================================
 *
 * Scroll narrative in four sections: Purpose, Methodology, Best Practices,
 * Expected Impact. Spec: docs/superpowers/specs/2026-07-10-documentation-page-design.md
 *
 * Honesty constraints (platform-wide):
 * - ONE network call (useKPIList) for the live KPI-count chip; on error the
 *   chip silently disappears — no error UI on a docs page.
 * - No fabricated performance/ROI digits anywhere; illustrative content is
 *   visually labeled "illustrative".
 */
import { useKPIList } from '@/hooks/api/use-kpi';
import { SectionNav, DOC_SECTIONS, STAT_CHIPS } from '@/components/documentation';

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-heading`} className="scroll-mt-16 space-y-6 pb-12">
      <h2 id={`${id}-heading`} className="text-xl font-semibold text-[var(--color-foreground)]">
        {title}
      </h2>
      {children}
    </section>
  );
}

function StatChipView({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] px-4 py-3 text-center">
      <div className="text-2xl font-bold text-[var(--color-foreground)]">{value}</div>
      <div className="text-xs text-[var(--color-muted-foreground)]">{label}</div>
    </div>
  );
}

export function Documentation() {
  // Live chip: renders only on success with a positive count; degrades to
  // absence (never a spinner, error banner, or placeholder number).
  const { data: kpiData } = useKPIList(undefined, { retry: false });
  const kpiTotal = kpiData?.total;

  return (
    <div className="space-y-6 px-1">
      <div>
        <h1 className="text-2xl font-bold text-[var(--color-foreground)]">Understanding E2I</h1>
        <p className="text-[var(--color-muted-foreground)]">
          What this platform is for, how its causal methodology works, how to use it well, and what
          impact to expect.
        </p>
      </div>

      <SectionNav sections={DOC_SECTIONS} />

      <Section id="purpose" title="Purpose — why E2I exists">
        <p className="max-w-3xl text-sm leading-6 text-[var(--color-foreground)]">
          Commercial pharma teams are surrounded by correlations: calls correlate with
          prescriptions, programs correlate with adoption. Correlation is cheap — and often wrong
          about what to do next. E2I applies formal causal inference, checked by adversarial
          refutation tests and grounded in clinical context, to answer the question that matters:{' '}
          <em>what actually causes the outcomes we care about?</em> It operates at three linked
          levels — HCP prescribing behavior, patient journey outcomes, and market &amp; brand
          performance.
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:max-w-3xl md:[grid-template-columns:repeat(5,minmax(0,1fr))]">
          {STAT_CHIPS.map((chip) => (
            <StatChipView key={chip.label} value={chip.value} label={chip.label} />
          ))}
          {typeof kpiTotal === 'number' && kpiTotal > 0 && (
            <StatChipView value={String(kpiTotal)} label="governed KPIs" />
          )}
        </div>
        {/* CausalScopeMap (Task 6), CorrelationCausationToggle (Task 7), CapabilityIndex (Task 8) mount here */}
      </Section>

      <Section id="methodology" title="Methodology — how it works">
        {/* CausalPipeline (Task 9), AgentTierStack (Task 10), ClinicalGrounding (Task 11) mount here */}
      </Section>

      <Section id="practices" title="Best Practices — using E2I well">
        {/* PracticeCards (Task 12) mounts here */}
      </Section>

      <Section id="impact" title="Expected Impact — what good looks like">
        {/* ImpactPathways (Task 13) mounts here */}
      </Section>
    </div>
  );
}

export default Documentation;
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx frontend/src/components/documentation/SectionNav.tsx frontend/src/components/documentation/index.ts
git commit -m "feat(docs-page): page shell with sticky scroll-spy section nav and stat chips"
```

---

### Task 4: Router + sidebar wiring

**Files:**
- Modify: `frontend/src/router/routes.tsx` (three insertions)
- Modify: `frontend/src/components/layout/Sidebar.tsx` (one iconMap entry)

- [ ] **Step 1: Add the lazy import in `routes.tsx`**

After the line `const Analytics = lazy(() => import('@/pages/Analytics'));` (end of the lazy-import block, ~line 34), add:

```tsx
const Documentation = lazy(() => import('@/pages/Documentation'));
```

- [ ] **Step 2: Add the routeConfig entry**

In `routeConfigs`, between the `/kpi-dictionary` entry (ends ~line 193) and the `/data-quality` entry, insert:

```tsx
  {
    path: '/documentation',
    title: 'Documentation',
    description: 'Platform purpose, methodology, and best practices',
    icon: 'graduation-cap',
    section: 'data',
    showInNav: true,
  },
```

- [ ] **Step 3: Add the RouteObject**

In the `routes` array, immediately after the `/kpi-dictionary` RouteObject (the `{ path: '/kpi-dictionary', element: (...) }` block), insert:

```tsx
  {
    path: '/documentation',
    element: (
      <ProtectedRoute>
        <LazyPage>
          <Documentation />
        </LazyPage>
      </ProtectedRoute>
    ),
  },
```

- [ ] **Step 4: Add the sidebar icon**

In `frontend/src/components/layout/Sidebar.tsx`, inside the `iconMap` in `NavIcon` (after the `'book-open'` entry, ~line 100), add — `book-open` is taken by KPI Dictionary, so Documentation uses a graduation cap (lucide `graduation-cap` paths, same stroke style as the existing icons):

```tsx
    'graduation-cap': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M22 10v6M2 10l10-5 10 5-10 5z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 12v5c3 3 9 3 12 0v-5" />
      </svg>
    ),
```

- [ ] **Step 5: Verify types and existing tests still pass**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx tsc -p tsconfig.app.json
npx vitest run src/components/layout/Sidebar.test.tsx src/components/documentation/content.test.ts
```

Expected: tsc clean; Sidebar tests PASS (the new route appears in "Data & Reference"); content invariants still PASS (`/documentation` is exempt).

- [ ] **Step 6: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/router/routes.tsx frontend/src/components/layout/Sidebar.tsx
git commit -m "feat(docs-page): wire /documentation route and sidebar entry (Data & Reference)"
```

---

### Task 5: Footer link (+ new Footer test)

**Files:**
- Test: `frontend/src/components/layout/Footer.test.tsx` (NEW — footer has zero coverage today)
- Modify: `frontend/src/components/layout/Footer.tsx:89-94` (insert after the System Status link)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/layout/Footer.test.tsx`:

```tsx
/**
 * Footer Tests — all four quick links.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Footer } from './Footer';

function renderFooter() {
  return render(
    <MemoryRouter>
      <Footer />
    </MemoryRouter>
  );
}

describe('Footer', () => {
  it('renders all four quick links', () => {
    renderFooter();
    expect(screen.getByRole('link', { name: /dashboard/i })).toHaveAttribute('href', '/');
    expect(screen.getByRole('link', { name: /system status/i })).toHaveAttribute('href', '/system-health');
    expect(screen.getByRole('link', { name: /documentation/i })).toHaveAttribute('href', '/documentation');
    expect(screen.getByRole('link', { name: /api docs/i })).toHaveAttribute('href', '/api/docs');
  });

  it('keeps API Docs external (new tab) and Documentation internal', () => {
    renderFooter();
    expect(screen.getByRole('link', { name: /api docs/i })).toHaveAttribute('target', '_blank');
    expect(screen.getByRole('link', { name: /documentation/i })).not.toHaveAttribute('target');
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/components/layout/Footer.test.tsx
```

Expected: FAIL — "Unable to find an accessible element with the role link and name /documentation/i" (the other assertions pass).

- [ ] **Step 3: Add the link**

In `frontend/src/components/layout/Footer.tsx`, inside `<nav aria-label="Footer navigation">`, after the System Status `</Link>` (line 94) and before the API Docs `<a>`, insert:

```tsx
            <Link
              to="/documentation"
              className="text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)] transition-colors"
            >
              Documentation
            </Link>
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/components/layout/Footer.test.tsx
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/layout/Footer.tsx frontend/src/components/layout/Footer.test.tsx
git commit -m "feat(docs-page): 4th footer link to /documentation, with first Footer tests"
```

---

### Task 6: CausalScopeMap (§1)

**Files:**
- Create: `frontend/src/components/documentation/CausalScopeMap.tsx`
- Modify: `frontend/src/components/documentation/index.ts`
- Modify: `frontend/src/pages/Documentation.tsx` (mount in §1)
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block to `Documentation.test.tsx`**

```tsx
describe('CausalScopeMap', () => {
  it('renders the three causal levels', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /hcp prescribing behavior/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /patient journey outcomes/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /market & brand performance/i })).toBeInTheDocument();
  });

  it('shows a level summary and its registry nodes on selection', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /patient journey outcomes/i }));
    expect(screen.getByText(/treatment initiation, persistence, and discontinuation/i)).toBeInTheDocument();
    expect(screen.getByText('patient persistence')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify the new block fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

Expected: the two new tests FAIL (buttons not found); earlier tests still pass.

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/CausalScopeMap.tsx`:

```tsx
/**
 * CausalScopeMap — the three linked causal levels E2I operates on.
 * Node labels are drawn verbatim from the causal registry's _NODE_LABELS
 * (src/insights/causal_context.py): every node shown is a modeled node.
 */
import { useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { SCOPE_LEVELS } from './content';
import type { CausalLevel } from './content';

const LEVEL_STYLES: Record<CausalLevel, { active: string; dot: string }> = {
  hcp: { active: 'border-blue-500/60 bg-blue-500/5', dot: 'bg-blue-500' },
  patient: { active: 'border-emerald-500/60 bg-emerald-500/5', dot: 'bg-emerald-500' },
  market: { active: 'border-purple-500/60 bg-purple-500/5', dot: 'bg-purple-500' },
};

export function CausalScopeMap() {
  const [selected, setSelected] = useState<CausalLevel>('hcp');
  const active = SCOPE_LEVELS.find((l) => l.id === selected) ?? SCOPE_LEVELS[0];

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="flex flex-col items-stretch">
        {SCOPE_LEVELS.map((level, i) => (
          <div key={level.id} className="flex flex-col items-center">
            {i > 0 && (
              <ArrowDown className="my-1 h-4 w-4 text-[var(--color-muted-foreground)]" aria-hidden="true" />
            )}
            <button
              type="button"
              onClick={() => setSelected(level.id)}
              aria-pressed={selected === level.id}
              className={`w-full rounded-lg border px-4 py-3 text-left transition-colors motion-reduce:transition-none ${
                selected === level.id
                  ? LEVEL_STYLES[level.id].active
                  : 'border-[var(--color-border)] bg-[var(--color-card)] hover:border-[var(--color-muted-foreground)]/40'
              }`}
            >
              <span className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${LEVEL_STYLES[level.id].dot}`} aria-hidden="true" />
                <span className="text-sm font-medium text-[var(--color-foreground)]">{level.title}</span>
              </span>
            </button>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
        <p className="text-sm leading-6 text-[var(--color-foreground)]">{active.summary}</p>
        <p className="mt-3 text-xs font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]">
          Modeled nodes at this level
        </p>
        <ul className="mt-2 flex flex-wrap gap-1.5">
          {active.nodes.map((node) => (
            <li
              key={node}
              className="rounded-full border border-[var(--color-border)] px-2.5 py-0.5 text-xs text-[var(--color-muted-foreground)]"
            >
              {node}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Export and mount**

Append to `frontend/src/components/documentation/index.ts`:

```ts
export { CausalScopeMap } from './CausalScopeMap';
```

In `frontend/src/pages/Documentation.tsx`: extend the import to `import { SectionNav, CausalScopeMap, DOC_SECTIONS, STAT_CHIPS } from '@/components/documentation';` and replace the `{/* CausalScopeMap (Task 6), ... */}` comment inside the Purpose section with:

```tsx
        <CausalScopeMap />
        {/* CorrelationCausationToggle (Task 7), CapabilityIndex (Task 8) mount here */}
```

- [ ] **Step 5: Run to verify it passes, then commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/CausalScopeMap.tsx frontend/src/components/documentation/index.ts frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): CausalScopeMap — three-level interactive capability map"
```

---

### Task 7: CorrelationCausationToggle (§1)

**Files:**
- Create: `frontend/src/components/documentation/CorrelationCausationToggle.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block**

```tsx
describe('CorrelationCausationToggle', () => {
  it('starts on the correlation view, labeled illustrative', () => {
    renderPage();
    expect(screen.getByText(/calls correlate with trx/i)).toBeInTheDocument();
    expect(screen.getAllByText(/illustrative/i).length).toBeGreaterThan(0);
  });

  it('reveals the confounder on toggle', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /reveal the confounder/i }));
    expect(screen.getByText(/specialty drives both/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify the new tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

Expected: 2 new FAILs.

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/CorrelationCausationToggle.tsx`:

```tsx
/**
 * CorrelationCausationToggle — the platform's core pitch, made visible.
 * View A: a convincing spurious scatter ("calls correlate with TRx").
 * View B: the confounder (specialty) revealed as a small DAG.
 * All coordinates are hand-authored and labeled "illustrative example" —
 * they are NOT real metrics (platform honesty discipline).
 */
import { useState } from 'react';

// Hand-authored scatter: two specialty clusters that create an overall upward
// trend even though within each cluster the relationship is flat.
const SPECIALISTS: Array<[number, number]> = [
  [120, 40], [135, 44], [150, 38], [165, 45], [180, 41], [195, 46],
];
const GENERALISTS: Array<[number, number]> = [
  [30, 95], [45, 99], [60, 93], [75, 100], [90, 96], [105, 101],
];

function Dot({ x, y, cls }: { x: number; y: number; cls: string }) {
  return <circle cx={x} cy={y} r={4} className={cls} />;
}

export function CorrelationCausationToggle() {
  const [revealed, setRevealed] = useState(false);

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-[var(--color-foreground)]">
          {revealed ? 'Causation: specialty drives both' : 'Correlation: calls correlate with TRx'}
        </h3>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-amber-500/50 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-600 dark:text-amber-400">
            Illustrative example
          </span>
          <button
            type="button"
            onClick={() => setRevealed((v) => !v)}
            className="rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs font-medium text-[var(--color-foreground)] transition-colors hover:bg-[var(--color-muted)] motion-reduce:transition-none"
          >
            {revealed ? 'Back to the raw scatter' : 'Reveal the confounder'}
          </button>
        </div>
      </div>

      {!revealed ? (
        <div>
          <svg viewBox="0 0 240 130" role="img" aria-label="Illustrative scatter plot: HCP calls versus TRx, trending upward" className="w-full max-w-xl">
            <line x1="20" y1="115" x2="230" y2="115" className="stroke-[var(--color-border)]" strokeWidth="1" />
            <line x1="20" y1="115" x2="20" y2="10" className="stroke-[var(--color-border)]" strokeWidth="1" />
            <text x="125" y="128" textAnchor="middle" className="fill-[var(--color-muted-foreground)] text-[8px]">HCP calls →</text>
            <text x="10" y="60" textAnchor="middle" transform="rotate(-90 10 60)" className="fill-[var(--color-muted-foreground)] text-[8px]">TRx →</text>
            {/* One visual population — the trend looks real */}
            {[...GENERALISTS, ...SPECIALISTS].map(([x, y]) => (
              <Dot key={`${x}-${y}`} x={x} y={y} cls="fill-[var(--color-primary)] opacity-70" />
            ))}
            <line x1="30" y1="100" x2="200" y2="35" className="stroke-[var(--color-primary)]" strokeWidth="1.5" strokeDasharray="4 3" />
          </svg>
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            More calls, more prescriptions — so call everyone more? Not so fast.
          </p>
        </div>
      ) : (
        <div>
          <svg viewBox="0 0 240 130" role="img" aria-label="Illustrative DAG: physician specialty causes both call targeting and TRx" className="w-full max-w-xl">
            <defs>
              <marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="fill-[var(--color-muted-foreground)]" />
              </marker>
            </defs>
            <rect x="85" y="10" width="70" height="24" rx="6" className="fill-amber-500/15 stroke-amber-500" strokeWidth="1.5" />
            <text x="120" y="26" textAnchor="middle" className="fill-[var(--color-foreground)] text-[9px] font-medium">Specialty</text>
            <rect x="15" y="90" width="70" height="24" rx="6" className="fill-[var(--color-muted)] stroke-[var(--color-border)]" strokeWidth="1.5" />
            <text x="50" y="106" textAnchor="middle" className="fill-[var(--color-foreground)] text-[9px] font-medium">HCP calls</text>
            <rect x="155" y="90" width="70" height="24" rx="6" className="fill-[var(--color-muted)] stroke-[var(--color-border)]" strokeWidth="1.5" />
            <text x="190" y="106" textAnchor="middle" className="fill-[var(--color-foreground)] text-[9px] font-medium">TRx</text>
            <line x1="100" y1="36" x2="58" y2="88" className="stroke-[var(--color-muted-foreground)]" strokeWidth="1.5" markerEnd="url(#dag-arrow)" />
            <line x1="140" y1="36" x2="182" y2="88" className="stroke-[var(--color-muted-foreground)]" strokeWidth="1.5" markerEnd="url(#dag-arrow)" />
            <line x1="87" y1="102" x2="153" y2="102" className="stroke-[var(--color-border)]" strokeWidth="1.5" strokeDasharray="4 3" markerEnd="url(#dag-arrow)" />
            <text x="120" y="97" textAnchor="middle" className="fill-[var(--color-muted-foreground)] text-[8px]">much weaker, adjusted</text>
          </svg>
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            Specialty drives both: specialists get more calls AND their patients need this therapy
            more. Adjust for specialty and the calls→TRx effect shrinks dramatically. E2I finds the
            adjustment automatically — and refutation-tests what remains.
          </p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { CorrelationCausationToggle } from './CorrelationCausationToggle';`
In `Documentation.tsx`, replace the `{/* CorrelationCausationToggle (Task 7), CapabilityIndex (Task 8) mount here */}` comment with:

```tsx
        <CorrelationCausationToggle />
        {/* CapabilityIndex (Task 8) mounts here */}
```

and add `CorrelationCausationToggle` to the `@/components/documentation` import.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): correlation-vs-causation toggle illustration"
```

---

### Task 8: CapabilityIndex (§1)

**Files:**
- Create: `frontend/src/components/documentation/CapabilityIndex.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block**

```tsx
describe('CapabilityIndex', () => {
  it('renders the five sidebar groups', () => {
    renderPage();
    const index = screen.getByRole('region', { name: /where to go for each question/i });
    for (const label of [
      'Causal Analytics',
      'Predictive Modeling',
      'Decisions & Optimization',
      'Data & Reference',
      'System & Platform',
    ]) {
      expect(within(index).getByRole('heading', { name: label })).toBeInTheDocument();
    }
  });

  it('links each card to its live page and excludes exempt/retired routes', () => {
    renderPage();
    const index = screen.getByRole('region', { name: /where to go for each question/i });
    expect(within(index).getByRole('link', { name: /segment analysis/i })).toHaveAttribute('href', '/segment-analysis');
    expect(within(index).queryByRole('link', { name: /causal discovery/i })).not.toBeInTheDocument();
    expect(within(index).queryByRole('link', { name: /^documentation$/i })).not.toBeInTheDocument();
  });

  it('shows causal-level badges on analysis surfaces', () => {
    renderPage();
    const index = screen.getByRole('region', { name: /where to go for each question/i });
    const twinCard = within(index).getByRole('link', { name: /digital twin/i }).closest('li');
    expect(twinCard).not.toBeNull();
    expect(within(twinCard as HTMLElement).getByText('HCP')).toBeInTheDocument();
  });
});
```

Also extend the RTL import at the top of the file to include `within`:

```tsx
import { render, screen, within } from '@testing-library/react';
```

- [ ] **Step 2: Run to verify the new tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/CapabilityIndex.tsx`:

```tsx
/**
 * CapabilityIndex — "where do I go for each question?"
 * Groups and page titles come from getNavigationSections() at render time, so
 * a retired page can never appear here and a new page shows up automatically
 * (content.test.ts fails the build until it gets a question).
 */
import { Link } from 'react-router-dom';
import { getNavigationSections } from '@/router/routes';
import { CAPABILITY_EXEMPT_PATHS, CAPABILITY_QUESTIONS, LEVEL_LABELS } from './content';

export function CapabilityIndex() {
  const groups = getNavigationSections()
    .filter((g) => g.label !== null)
    .map((g) => ({
      ...g,
      routes: g.routes.filter((r) => !(CAPABILITY_EXEMPT_PATHS as readonly string[]).includes(r.path)),
    }))
    .filter((g) => g.routes.length > 0);

  return (
    <section aria-label="Where to go for each question" className="space-y-6">
      <h3 className="text-sm font-semibold text-[var(--color-foreground)]">
        Where to go for each question
      </h3>
      {groups.map((group) => (
        <div key={group.key}>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]">
            {group.label}
          </h4>
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {group.routes.map((route) => {
              const info = CAPABILITY_QUESTIONS[route.path];
              return (
                <li
                  key={route.path}
                  className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-3 transition-colors hover:border-[var(--color-primary)]/50 motion-reduce:transition-none"
                >
                  <div className="flex items-start justify-between gap-2">
                    <Link
                      to={route.path}
                      className="text-sm font-medium text-[var(--color-primary)] hover:underline"
                    >
                      {route.title}
                    </Link>
                    {info?.levels && (
                      <span className="flex shrink-0 gap-1">
                        {info.levels.map((level) => (
                          <span
                            key={level}
                            className="rounded-full border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
                          >
                            {LEVEL_LABELS[level]}
                          </span>
                        ))}
                      </span>
                    )}
                  </div>
                  {info && (
                    <p className="mt-1 text-xs leading-5 text-[var(--color-muted-foreground)]">
                      {info.question}
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </section>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { CapabilityIndex } from './CapabilityIndex';`
In `Documentation.tsx`, replace `{/* CapabilityIndex (Task 8) mounts here */}` with `<CapabilityIndex />` and extend the import.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx src/components/documentation/content.test.ts
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): per-page CapabilityIndex derived from live nav sections"
```

---

### Task 9: CausalPipeline (§2)

**Files:**
- Create: `frontend/src/components/documentation/CausalPipeline.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block**

```tsx
describe('CausalPipeline', () => {
  it('renders the five stages', () => {
    renderPage();
    for (const name of ['Frame', 'Identify', 'Estimate', 'Refute', 'Act']) {
      expect(screen.getByRole('button', { name: new RegExp(`^${name}`, 'i') })).toBeInTheDocument();
    }
  });

  it('expands a stage with plain language and a "For analysts" collapsible', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /^refute/i }));
    expect(screen.getByText(/attack the estimate before believing it/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /for analysts/i }));
    expect(screen.getByText(/placebo treatment/i)).toBeInTheDocument();
    expect(screen.getByText(/e-value/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify the new tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/CausalPipeline.tsx`:

```tsx
/**
 * CausalPipeline — the 5-stage causal workflow (Frame → Identify → Estimate →
 * Refute → Act), visually kin to visualizations/QueryProcessingFlow. Clicking
 * a stage expands a two-layer panel: plain language + "For analysts"
 * (Radix Collapsible).
 */
import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { PIPELINE_STAGES } from './content';

export function CausalPipeline() {
  const [openId, setOpenId] = useState<string | null>(null);
  const openStage = PIPELINE_STAGES.find((s) => s.id === openId);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        {PIPELINE_STAGES.map((stage, i) => (
          <div key={stage.id} className="flex items-center gap-2">
            {i > 0 && (
              <ChevronRight className="h-4 w-4 text-[var(--color-muted-foreground)]" aria-hidden="true" />
            )}
            <button
              type="button"
              onClick={() => setOpenId(openId === stage.id ? null : stage.id)}
              aria-expanded={openId === stage.id}
              className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors motion-reduce:transition-none ${
                openId === stage.id
                  ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                  : 'border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-foreground)] hover:border-[var(--color-primary)]/50'
              }`}
            >
              {stage.name}
            </button>
          </div>
        ))}
      </div>

      {openStage && (
        <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
          <p className="text-sm leading-6 text-[var(--color-foreground)]">{openStage.plain}</p>
          <Collapsible className="mt-3">
            <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-[var(--color-primary)] hover:underline">
              <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
              For analysts
            </CollapsibleTrigger>
            <CollapsibleContent>
              <p className="mt-2 border-l-2 border-[var(--color-border)] pl-3 text-xs leading-5 text-[var(--color-muted-foreground)]">
                {openStage.analyst}
              </p>
            </CollapsibleContent>
          </Collapsible>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { CausalPipeline } from './CausalPipeline';`
In `Documentation.tsx`, replace the Methodology placeholder comment with:

```tsx
        <CausalPipeline />
        {/* AgentTierStack (Task 10), ClinicalGrounding (Task 11) mount here */}
```

and extend the import.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): clickable 5-stage causal pipeline with analyst layer"
```

---

### Task 10: AgentTierStack (§2)

**Files:**
- Create: `frontend/src/components/documentation/AgentTierStack.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block**

```tsx
describe('AgentTierStack', () => {
  it('renders all six tiers', () => {
    renderPage();
    for (const name of ['ML Foundation', 'Coordination', 'Causal Analytics', 'Monitoring', 'ML Predictions', 'Self-Improvement']) {
      expect(screen.getByRole('button', { name: new RegExp(name, 'i') })).toBeInTheDocument();
    }
  });

  it('expands a tier to list its agents', async () => {
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /causal analytics.*3 agents/i }));
    expect(screen.getByText('causal_impact')).toBeInTheDocument();
    expect(screen.getByText('heterogeneous_optimizer')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify the new tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/AgentTierStack.tsx`:

```tsx
/**
 * AgentTierStack — the 6-tier / 21-agent architecture. The corrected
 * successor to the deleted kpi/AgenticMethodology.tsx; roster source:
 * src/agents/factory.py AGENT_REGISTRY_CONFIG (content.test.ts enforces 21/6).
 */
import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { AGENT_TIERS } from './content';

export function AgentTierStack() {
  const [openTier, setOpenTier] = useState<number | null>(null);

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--color-foreground)]">
        The agent system: 21 agents in 6 tiers
      </h3>
      <div className="space-y-2">
        {AGENT_TIERS.map((tier) => {
          const open = openTier === tier.tier;
          return (
            <div key={tier.tier} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
              <button
                type="button"
                onClick={() => setOpenTier(open ? null : tier.tier)}
                aria-expanded={open}
                className="flex w-full items-center justify-between px-4 py-2.5 text-left"
              >
                <span className="flex items-baseline gap-2">
                  <span className="text-xs font-mono text-[var(--color-muted-foreground)]">T{tier.tier}</span>
                  <span className="text-sm font-medium text-[var(--color-foreground)]">{tier.name}</span>
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    ({tier.agents.length} agent{tier.agents.length > 1 ? 's' : ''})
                  </span>
                </span>
                <ChevronDown
                  className={`h-4 w-4 text-[var(--color-muted-foreground)] transition-transform motion-reduce:transition-none ${open ? 'rotate-180' : ''}`}
                  aria-hidden="true"
                />
              </button>
              {open && (
                <div className="border-t border-[var(--color-border)] px-4 py-3">
                  <p className="mb-2 text-xs text-[var(--color-muted-foreground)]">{tier.blurb}</p>
                  <ul className="grid gap-1.5 sm:grid-cols-2">
                    {tier.agents.map((agent) => (
                      <li key={agent.id} className="text-xs leading-5">
                        <span className="font-mono text-[var(--color-foreground)]">{agent.id}</span>
                        <span className="text-[var(--color-muted-foreground)]"> — {agent.role}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { AgentTierStack } from './AgentTierStack';`
In `Documentation.tsx`, replace the remaining Methodology placeholder comment with:

```tsx
        <AgentTierStack />
        {/* ClinicalGrounding (Task 11) mounts here */}
```

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): 6-tier/21-agent interactive tier stack (supersedes AgenticMethodology content)"
```

---

### Task 11: ClinicalGrounding (§2)

**Files:**
- Create: `frontend/src/components/documentation/ClinicalGrounding.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block**

```tsx
describe('ClinicalGrounding', () => {
  it('renders all five clinical sources with UMLS and OpenFDA present', () => {
    renderPage();
    const strip = screen.getByRole('region', { name: /grounded in clinical reality/i });
    for (const name of ['UMLS', 'OpenFDA', 'ClinicalTrials.gov', 'PubMed', 'ChEMBL']) {
      expect(within(strip).getByText(name)).toBeInTheDocument();
    }
  });
});
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/ClinicalGrounding.tsx`:

```tsx
/**
 * ClinicalGrounding — "Business insights, grounded in clinical reality."
 * DESCRIBES the platform's external clinical knowledge integrations (UMLS,
 * OpenFDA, ClinicalTrials.gov, PubMed, ChEMBL) as static constants — this
 * page never calls those APIs itself.
 */
import { BookMarked, ShieldCheck } from 'lucide-react';
import { CLINICAL_SOURCES } from './content';

export function ClinicalGrounding() {
  const prominent = CLINICAL_SOURCES.filter((s) => s.prominent);
  const rest = CLINICAL_SOURCES.filter((s) => !s.prominent);

  return (
    <section aria-label="Business insights, grounded in clinical reality">
      <h3 className="mb-2 text-sm font-semibold text-[var(--color-foreground)]">
        Business insights, grounded in clinical reality
      </h3>
      <p className="mb-3 max-w-3xl text-xs leading-5 text-[var(--color-muted-foreground)]">
        Commercial signals only mean something inside their clinical context. E2I links entities
        through medical terminology and gates insight language against official drug labeling,
        drawing on five authoritative external sources.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {prominent.map((source) => (
          <div
            key={source.name}
            className="rounded-lg border border-[var(--color-primary)]/40 bg-[var(--color-primary)]/5 p-4"
          >
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-[var(--color-primary)]" aria-hidden="true" />
              <span className="text-sm font-semibold text-[var(--color-foreground)]">{source.name}</span>
            </div>
            <p className="mt-1.5 text-xs leading-5 text-[var(--color-muted-foreground)]">{source.role}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {rest.map((source) => (
          <div key={source.name} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-3">
            <div className="flex items-center gap-2">
              <BookMarked className="h-3.5 w-3.5 text-[var(--color-muted-foreground)]" aria-hidden="true" />
              <span className="text-xs font-semibold text-[var(--color-foreground)]">{source.name}</span>
            </div>
            <p className="mt-1 text-xs leading-5 text-[var(--color-muted-foreground)]">{source.role}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { ClinicalGrounding } from './ClinicalGrounding';`
In `Documentation.tsx`, replace `{/* ClinicalGrounding (Task 11) mounts here */}` with `<ClinicalGrounding />`.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): clinical grounding strip (UMLS/OpenFDA prominent)"
```

---

### Task 12: PracticeCards (§3)

**Files:**
- Create: `frontend/src/components/documentation/PracticeCards.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append block)

- [ ] **Step 1: Append the failing test block**

```tsx
describe('PracticeCards', () => {
  it('renders do/don’t pairs', () => {
    renderPage();
    expect(screen.getByText(/check the refutation gate/i)).toBeInTheDocument();
    expect(screen.getByText(/rerun an analysis with different settings/i)).toBeInTheDocument();
  });

  it('filters by role', async () => {
    renderPage();
    // whatif-ranges is analyst-only; informational-kpis is exec-only.
    await userEvent.click(screen.getByRole('button', { name: /^exec$/i }));
    expect(screen.queryByText(/what-if simulation inputs/i)).not.toBeInTheDocument();
    expect(screen.getByText(/informational.*kpis as if they were performance targets/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /^all$/i }));
    expect(screen.getByText(/what-if simulation inputs/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/PracticeCards.tsx`:

```tsx
/**
 * PracticeCards — paired Do/Don't cards grounded in real product behavior
 * (refutation gates, Informational KPIs, honest nulls, what-if ranges,
 * per-brand scope), filterable by audience role.
 */
import { useState } from 'react';
import { Check, X } from 'lucide-react';
import { PRACTICES } from './content';
import type { PracticeRole } from './content';

type Filter = 'all' | PracticeRole;

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'exec', label: 'Exec' },
  { id: 'analyst', label: 'Analyst' },
];

export function PracticeCards() {
  const [filter, setFilter] = useState<Filter>('all');
  const visible = PRACTICES.filter((p) => filter === 'all' || p.roles.includes(filter));

  return (
    <div>
      <div className="mb-3 flex items-center gap-1.5" role="group" aria-label="Filter practices by role">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFilter(f.id)}
            aria-pressed={filter === f.id}
            className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors motion-reduce:transition-none ${
              filter === f.id
                ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                : 'border-[var(--color-border)] text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>
      <ul className="grid gap-3 md:grid-cols-2">
        {visible.map((p) => (
          <li key={p.id} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
            <div className="flex items-start gap-2">
              <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
              <p className="text-sm leading-6 text-[var(--color-foreground)]">{p.doText}</p>
            </div>
            <div className="mt-2 flex items-start gap-2">
              <X className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
              <p className="text-sm leading-6 text-[var(--color-muted-foreground)]">{p.dontText}</p>
            </div>
            <p className="mt-2 border-t border-[var(--color-border)] pt-2 text-xs leading-5 text-[var(--color-muted-foreground)]">
              {p.why}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { PracticeCards } from './PracticeCards';`
In `Documentation.tsx`, replace the Best Practices placeholder comment with `<PracticeCards />`.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): do/don't practice cards with role filter"
```

---

### Task 13: ImpactPathways (§4) + live-chip degradation tests

**Files:**
- Create: `frontend/src/components/documentation/ImpactPathways.tsx`
- Modify: `frontend/src/components/documentation/index.ts`, `frontend/src/pages/Documentation.tsx`
- Test: `frontend/src/pages/Documentation.test.tsx` (append two blocks)

- [ ] **Step 1: Append the failing test blocks**

```tsx
describe('ImpactPathways', () => {
  it('renders four pathway cards linking to live pages', () => {
    renderPage();
    const region = screen.getByRole('region', { name: /expected impact pathways/i });
    expect(within(region).getByRole('link', { name: /see your segments/i })).toHaveAttribute('href', '/segment-analysis');
    expect(within(region).getByRole('link', { name: /see your allocation/i })).toHaveAttribute('href', '/resource-optimization');
    expect(within(region).getByRole('link', { name: /run a simulation/i })).toHaveAttribute('href', '/digital-twin');
    expect(within(region).getByRole('link', { name: /open the dashboard/i })).toHaveAttribute('href', '/');
  });
});

describe('live KPI chip degradation', () => {
  it('shows the governed-KPIs chip when the query succeeds', () => {
    renderPage();
    expect(screen.getByText('46')).toBeInTheDocument();
    expect(screen.getByText(/governed kpis/i)).toBeInTheDocument();
  });

  it('silently omits the chip on error — no error UI', () => {
    vi.mocked(useKPIList).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    } as ReturnType<typeof useKPIList>);
    renderPage();
    expect(screen.queryByText(/governed kpis/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    // Static chips unaffected:
    expect(screen.getByText(/intervention channels/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify the ImpactPathways tests fail**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
```

Expected: ImpactPathways block FAILS; the two live-chip tests already PASS (behavior shipped in Task 3) — they are locked in here because this is the last page-assembly task.

- [ ] **Step 3: Create the component**

Create `frontend/src/components/documentation/ImpactPathways.tsx`:

```tsx
/**
 * ImpactPathways — Expected Impact, honestly framed. Mechanism-focused cards
 * with NO fabricated ROI digits (enforced by content.test.ts); each links to
 * the live page where users see their own numbers.
 */
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { IMPACT_PATHWAYS } from './content';

export function ImpactPathways() {
  return (
    <section aria-label="Expected impact pathways">
      <p className="mb-3 max-w-3xl text-sm leading-6 text-[var(--color-foreground)]">
        E2I does not promise a number — fabricated ROI figures are exactly what this platform is
        built to eliminate. It promises mechanisms, each measurable on its own live page with your
        data:
      </p>
      <ul className="grid gap-3 sm:grid-cols-2">
        {IMPACT_PATHWAYS.map((p) => (
          <li key={p.title} className="flex flex-col rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">{p.title}</h3>
            <p className="mt-1 flex-1 text-xs leading-5 text-[var(--color-muted-foreground)]">{p.mechanism}</p>
            <Link
              to={p.href}
              className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-[var(--color-primary)] hover:underline"
            >
              {p.linkLabel}
              <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
```

- [ ] **Step 4: Export, mount, verify, commit**

Append to `index.ts`: `export { ImpactPathways } from './ImpactPathways';`
In `Documentation.tsx`, replace the Expected Impact placeholder comment with `<ImpactPathways />`.

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx vitest run src/pages/Documentation.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/documentation/ frontend/src/pages/Documentation.tsx frontend/src/pages/Documentation.test.tsx
git commit -m "feat(docs-page): honest impact pathways + live-chip degradation tests"
```

---

### Task 14: Delete AgenticMethodology (supersede)

**Files:**
- Delete: `frontend/src/components/kpi/AgenticMethodology.tsx`
- Modify: `frontend/src/components/kpi/index.ts:13-14`

- [ ] **Step 1: Verify it still has zero consumers (cheapest disproof)**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
grep -rn "AgenticMethodology" frontend/src frontend/e2e --include="*.ts" --include="*.tsx" | grep -v "components/kpi/AgenticMethodology.tsx"
```

Expected: ONLY the comment lines in `frontend/src/components/kpi/index.ts`. If anything else appears, STOP and report — do not delete.

- [ ] **Step 2: Delete and update the comment**

```bash
git rm frontend/src/components/kpi/AgenticMethodology.tsx
```

In `frontend/src/components/kpi/index.ts`, replace lines 13-14:

```ts
// AgenticMethodology component is available but not exported from page
// (removed from KPI Dictionary page per user request - content is outdated)
```

with:

```ts
// AgenticMethodology was deleted 2026-07 (content had gone stale); its role is
// superseded by the /documentation page (src/pages/Documentation.tsx).
```

- [ ] **Step 3: Verify nothing broke, commit**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx tsc -p tsconfig.app.json
npx vitest run src/pages/KPIDictionary.test.tsx
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/src/components/kpi/index.ts
git commit -m "chore(kpi): delete stale AgenticMethodology, superseded by /documentation"
```

Expected: tsc clean, KPIDictionary tests PASS.

---

### Task 15: Playwright e2e spec

**Files:**
- Modify: `frontend/e2e/fixtures/test-data.ts` (ROUTES map, ~line 113)
- Create: `frontend/e2e/pages/documentation.page.ts`
- Create: `frontend/e2e/specs/documentation.spec.ts`

Note: `_smoke.spec.ts` keeps its fixed 4 routes — do not touch it.

- [ ] **Step 1: Add the route constant**

In `frontend/e2e/fixtures/test-data.ts`, inside `ROUTES` after `KPI_DICTIONARY: '/kpi-dictionary',` add:

```ts
  DOCUMENTATION: '/documentation',
```

- [ ] **Step 2: Create the page object**

Create `frontend/e2e/pages/documentation.page.ts`:

```ts
import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Documentation page ("Understanding E2I").
 */
export class DocumentationPage extends BasePage {
  readonly url = ROUTES.DOCUMENTATION
  readonly pageTitle = /Documentation|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await this.page.goto(this.url)
    await this.page.waitForLoadState('domcontentloaded')
    await this.pageHeader.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {})
    await this.page.waitForTimeout(300)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Understanding E2I/i }).first()
  }

  get sectionNav(): Locator {
    return this.page.getByRole('navigation', { name: /on this page/i })
  }

  get refuteStage(): Locator {
    return this.page.getByRole('button', { name: /^Refute/i })
  }

  get capabilityIndex(): Locator {
    return this.page.getByRole('region', { name: /where to go for each question/i })
  }
}
```

- [ ] **Step 3: Create the spec**

Create `frontend/e2e/specs/documentation.spec.ts`:

```ts
import { test, expect } from '@playwright/test'
import { DocumentationPage } from '../pages/documentation.page'
import { mockApiRoutes } from '../fixtures/api-mocks'
import { TIMEOUTS } from '../fixtures/test-data'
import { assertNoErrors } from '../utils/assertions'

test.describe('Documentation Page', () => {
  let docPage: DocumentationPage

  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page)
    docPage = new DocumentationPage(page)
    await docPage.goto()
  })

  test.describe('Page Load', () => {
    test('should load successfully', async () => {
      await expect(docPage.pageHeader).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })

    test('should show no errors on load', async ({ page }) => {
      await assertNoErrors(page)
    })

    test('should display the section nav', async () => {
      await expect(docPage.sectionNav).toBeVisible()
    })
  })

  test.describe('Interactivity', () => {
    test('expands a pipeline stage', async ({ page }) => {
      await docPage.refuteStage.click()
      await expect(page.getByText(/Attack the estimate before believing it/i)).toBeVisible()
    })

    test('capability index links to live pages', async () => {
      await expect(docPage.capabilityIndex).toBeVisible()
      await expect(
        docPage.capabilityIndex.getByRole('link', { name: /Segment Analysis/i })
      ).toHaveAttribute('href', '/segment-analysis')
    })
  })

  test.describe('Footer entry point', () => {
    test('footer Documentation link navigates here', async ({ page }) => {
      await page.goto('/')
      await page.getByRole('contentinfo').getByRole('link', { name: /^Documentation$/i }).click()
      await expect(docPage.pageHeader).toBeVisible({ timeout: TIMEOUTS.PAGE_LOAD })
    })
  })
})
```

- [ ] **Step 4: Type-check the e2e code and commit**

Local Playwright runs are UNFAITHFUL unless `VITE_MSW_ENABLED=false` (memory 2026-07-07) and are heavy on this box — rely on CI's e2e job; locally just type-check:

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx tsc -p tsconfig.app.json
cd /home/enunez/Projects/e2i_causal_analytics
git add frontend/e2e/fixtures/test-data.ts frontend/e2e/pages/documentation.page.ts frontend/e2e/specs/documentation.spec.ts
git commit -m "test(docs-page): Playwright spec + page object for /documentation"
```

(If e2e files are covered by a different tsconfig and `tsc -p tsconfig.app.json` skips them, that matches the repo's existing e2e handling — CI's e2e job compiles them.)

---

### Task 16: Full local verification

**Files:** none (verification only)

- [ ] **Step 1: Type check (the real one)**

```bash
cd /home/enunez/Projects/e2i_causal_analytics/frontend
npx tsc -p tsconfig.app.json
```

Expected: exit 0, no output.

- [ ] **Step 2: Run every test file this branch touched or could affect**

```bash
npx vitest run \
  src/pages/Documentation.test.tsx \
  src/components/documentation/content.test.ts \
  src/components/layout/Footer.test.tsx \
  src/components/layout/Sidebar.test.tsx \
  src/pages/KPIDictionary.test.tsx
```

Expected: ALL PASS. (Do not run the whole suite locally — CI is the arbiter on this box.)

- [ ] **Step 3: Lint check (no --write)**

```bash
npx eslint src/pages/Documentation.tsx src/components/documentation src/components/layout/Footer.tsx 2>/dev/null || echo "check eslint config invocation in package.json scripts if this errors"
```

Fix any reported issues by hand.

- [ ] **Step 4: Commit any fixes**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git status --porcelain   # if dirty:
git add -A frontend/src frontend/e2e && git commit -m "fix(docs-page): verification round fixes"
```

---

### Task 17: PR, CI, merge, deploy, live verification

**Files:** none

- [ ] **Step 1: Push and open the PR**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
git config --global http.https://github.com.proxy ""
git push -u origin feat/documentation-page
gh pr create --title "feat: /documentation page — Understanding E2I (footer + sidebar), supersedes AgenticMethodology" --body "$(cat <<'EOF'
## Summary
- New `/documentation` page ("Understanding E2I"): scroll narrative with sticky scroll-spy nav across Purpose / Methodology / Best Practices / Expected Impact
- §1: three-level CausalScopeMap (nodes verbatim from the causal registry), correlation-vs-causation toggle, per-page CapabilityIndex derived from getNavigationSections() (cannot advertise retired pages), stat chips (static structural + live KPI count via useKPIList with silent degradation)
- §2: clickable 5-stage causal pipeline (plain language + "For analysts" collapsibles incl. the 5 refutation tests), 6-tier/21-agent AgentTierStack, ClinicalGrounding strip (UMLS + OpenFDA prominent; ClinicalTrials.gov, PubMed, ChEMBL)
- §3: Do/Don't practice cards with Exec/Analyst filter, grounded in real product behavior
- §4: mechanism-focused impact pathways, digit-free by test-enforced design
- 4th footer link + sidebar entry (Data & Reference, new graduation-cap icon)
- DELETED stale `kpi/AgenticMethodology.tsx` (superseded; zero consumers verified)
- Anti-drift invariants in content.test.ts (router coverage, 21/6 agent roster, no fabricated digits)

Spec: docs/superpowers/specs/2026-07-10-documentation-page-design.md

## Test plan
- [ ] CI green (type check, vitest, e2e)
- [ ] Live: footer link on several pages, sidebar entry, all interactives, both themes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch CI (TAB-separated output)**

```bash
gh pr checks --watch --interval 60 | awk -F'\t' '{print $1, $2}'
```

Expected: all checks pass. If the pip-audit gate fails on a CVE published after main's last green scan (known trap), compare `published_at` vs scan time and handle per the allowlist policy — do not blindly rerun.

- [ ] **Step 3: Merge (never squash; frontend PRs need --admin)**

```bash
gh pr merge --merge --admin
```

- [ ] **Step 4: Watch the deploy and verify conclusion (do not trust `| tail`)**

```bash
git checkout main && git pull --ff-only
gh run list --branch main --limit 3
# after ~12 min, for the deploy run id:
gh run view <run-id> --json conclusion,status
```

Expected: `"conclusion": "success"`. If the deploy refuses due to a dirty droplet tree, ensure the working tree is clean `git status` and `gh run rerun <run-id> --failed`.

- [ ] **Step 5: Live verification on https://eznomics.site/**

1. Bundle check (proves the new chunk shipped):
```bash
curl -s https://eznomics.site/ | grep -o 'assets/index-[^"]*\.js' | head -1
# fetch that bundle and confirm the lazy route is registered:
curl -s "https://eznomics.site/assets/<index-bundle>.js" | grep -c "documentation"
```
Expected: count ≥ 1.
2. In the user's logged-in Chrome (classifier blocks minting test users — memory 2026-07-09), or via chrome-devtools MCP against their session: visit `/documentation`; confirm header, section nav scroll-spy, scope map click, correlation toggle, pipeline expand + "For analysts", tier stack, clinical strip, practice filter, impact links, live KPI chip; check footer "Documentation" link from 2-3 other pages and the sidebar entry under Data & Reference; toggle dark mode and re-scan all sections; confirm `/kpi-dictionary` still loads clean (post-deletion).
3. Report results faithfully — if anything is off, fix on a follow-up branch; never edit tracked files on main while a deploy may be in flight.

---

## Self-review (performed at plan-writing time)

- **Spec coverage:** Footer link → T5; sidebar + route → T4; §1 narrative/chips/scope map/toggle/index → T3/T6/T7/T8; §2 pipeline/tiers/clinical → T9/T10/T11; §3 → T12; §4 + chip degradation → T13; deletion → T14; e2e + ROUTES → T15; tsc/vitest gates → T16; PR/merge/deploy/live → T17. Scroll-spy graceful degradation → SectionNav guard + jsdom tests. No-fabricated-digits → content.test.ts regex. ✔
- **Placeholder scan:** every code step contains complete code; no TBDs. The only intentionally deferred content is none. ✔
- **Type consistency:** `CausalLevel`/`LEVEL_LABELS`/`CAPABILITY_EXEMPT_PATHS` defined in T2 and consumed in T6/T8; `DocSection`/`DOC_SECTIONS` in T2, consumed in T3; `PracticeRole` in T2, consumed in T12; `useKPIList(undefined, { retry: false })` matches the hook's `(params?, options?)` signature; `KPIListResponse.total` exists (`types/kpi.ts:246`). ✔

---

## Appendix: Authorized deviations from plan code (recorded during execution — do NOT revert)

Appended 2026-07-10 by the execution coordinator. The plan code above is left untouched so task line references stay valid; where the deviations below conflict with plan code blocks, THE DEVIATIONS WIN. Later tasks that transcribe or extend the affected files must preserve them.

**Task 3 type fixes (commit 86eaa91c):**
1. `frontend/src/pages/Documentation.tsx` — `Section` helper props: `children?: React.ReactNode` (optional). The plan's required prop caused TS2741 on comment-only sections.
2. `frontend/src/pages/Documentation.test.tsx` — mock idiom is `(useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({...})` (repo convention, per KPIDictionary.test.tsx), NOT the plan's `vi.mocked(...).mockReturnValue({...} as ReturnType<typeof useKPIList>)`, which fails TS2352. **Task 13's planned test additions repeat the bad cast — use the repo idiom instead.**

**Task 3 code-quality review fixes (commit 3dbddf4c):**
3. `SectionNav.tsx` — nav is `sticky top-16` (NOT `top-0`; the app Header is `sticky top-0 z-40 h-16` and paints over a `top-0` nav).
4. `SectionNav.tsx` — full-bleed is `-mx-1 px-1` (NOT `-mx-4 px-4`; page wrapper uses `px-1`).
5. `SectionNav.tsx` — scroll-spy callback filters to intersecting entries, early-returns on an empty batch, and activates the entry with the smallest `boundingClientRect.top` (topmost wins; the plan's last-write-wins loop was a bug).
6. `SectionNav.tsx` — click handler respects reduced motion: `typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches` → `scrollIntoView` with `behavior: 'auto'` instead of `'smooth'`.
7. `frontend/src/pages/Documentation.tsx` — `Section` helper uses `scroll-mt-28` (NOT `scroll-mt-16`; must clear header 64px + docked nav ≈48px).
8. `frontend/src/pages/Documentation.tsx` — `const showLiveChip = typeof kpiTotal === 'number' && kpiTotal > 0;` derived once and used BOTH for rendering the live chip AND for the stat-chip grid columns (`repeat(5,…)` when true, `repeat(4,…)` when false — both literals statically present for Tailwind JIT).
9. NEW FILE `frontend/src/components/documentation/SectionNav.test.tsx` — 2 tests locking in the scroll-spy behavior (topmost-wins, ignore-empty-batch). Exists beyond the plan's file list; keep it passing.
10. `frontend/src/components/documentation/index.ts` — header comment reads "(removed by this feature)" not "(deleted)" (AgenticMethodology deletion happens in Task 14).

**Task 4 deviation (commit 6cd93ca2):**
11. `frontend/src/components/layout/Sidebar.test.tsx` — hardcoded nav-link count bumped 23 → 24 (assertion + test title). The plan's Step 5 wrongly predicted Sidebar tests would pass untouched; adding the `/documentation` nav entry legitimately raises the count. Task 16's verification list includes this test — expect 24.

**Task 6 code-quality review fix (commit a6faa870):**
12. `frontend/src/pages/Documentation.test.tsx` — the CausalScopeMap describe block contains ONE test beyond the plan's Step 1 code: "defaults to the HCP level active on mount" (HCP button `aria-pressed="true"`, Patient/Market buttons `aria-pressed="false"`, HCP node text "rep detailing frequency" visible pre-click), and the plan's click test additionally asserts the Patient button has `aria-pressed="true"` after the click. Added because the plan's two tests never pinned the `useState('hcp')` default. Keep these; Tasks 7–13 appending later describe blocks must not remove them. Consider the same default-state pinning pattern when transcribing later toggle-style components.

**Task 7 plan-bug fix (commit a7d0f157):**
13. `frontend/src/pages/Documentation.test.tsx` — the second CorrelationCausationToggle test asserts `expect(screen.getAllByText(/specialty drives both/i).length).toBeGreaterThan(0);` (NOT the plan's `getByText(...)...toBeInTheDocument()`). The plan's query is a bug: both the revealed h3 heading ('Causation: specialty drives both') and the paragraph ('Specialty drives both: …') match, so `getByText` throws "multiple elements found" — verbatim transcription cannot go green. Component copy stays exactly as planned; only the test query changed (mirroring the plan's own `getAllByText(/illustrative/i)` idiom from the first test).

**Task 9 code-quality review fix (commit cfe4b209):**
14. `frontend/src/components/documentation/CausalPipeline.tsx` — the expanded-panel div carries `key={openStage.id}` (plan code has no key). Without it the uncontrolled Radix Collapsible is reconciled in place across stage switches, so an expanded "For analysts" layer leaks into the next stage unrequested (empirically confirmed; also path-inconsistent with the close-then-reopen reset). The key forces a remount per stage. Additionally the CausalPipeline describe block in `frontend/src/pages/Documentation.test.tsx` contains a THIRD test beyond the plan's two: "resets the \"For analysts\" layer when switching stages" (expand Refute's analyst layer → click Estimate → trigger `aria-expanded="false"` and `queryByText(/econml/i)` absent). Later tasks appending describe blocks must not remove either.

**Task 10 code-quality review fix (commit de906493):**
15. `frontend/src/pages/Documentation.test.tsx` — the AgentTierStack describe block contains a THIRD test beyond the plan's two: "starts fully collapsed, closes the open tier when another opens, and toggles closed on re-click" (all six tier buttons `aria-expanded="false"` on mount; open Monitoring then Coordination → Monitoring flips back to `false`; re-click Coordination → `false`). Added per the recurring default-state/exclusivity regression pattern (see items 12 and 14); non-vacuousness verified by assertion inversion. Component code (`AgentTierStack.tsx`) is byte-exact to plan — only the test file deviates. Later tasks appending describe blocks must not remove it.

**Task 12 code-quality review fix (commit 6672abc0):**
16. `frontend/src/pages/Documentation.test.tsx` — the two PracticeCards tests carry FOUR assertions beyond the plan's code, per the same default-state regression pattern (items 12/15; third toggle-style component caught with an untested default). In "renders do/don't pairs": `expect(screen.getByText(/what-if simulation inputs/i)).toBeInTheDocument()` (analyst-only whatif-ranges content visible at mount proves the `useState<Filter>('all')` default) and `expect(screen.getByRole('button', { name: /^all$/i })).toHaveAttribute('aria-pressed', 'true')`. In "filters by role", immediately after clicking Exec: Exec button `aria-pressed="true"` and All button `aria-pressed="false"`. Non-vacuousness proven by two mutations (default `'exec'` → 1 test failed; hardcoded `aria-pressed={false}` → both tests failed; reverted, 20/20 green). Component code (`PracticeCards.tsx`) is byte-exact to plan — only the test file deviates. Later tasks appending describe blocks must not remove these assertions.

**Task 13 plan-bug fixes (commit b1fe4360):**
17. `frontend/src/pages/Documentation.test.tsx` — THREE assertion lines in the two live-KPI-chip tests use exact-string queries instead of the plan's case-insensitive regexes, because both regexes match static page prose in addition to the chip elements (RTL `getByText` then throws "multiple elements" / `queryByText` false-positives). (a) In "silently omits the chip on error — no error UI": `screen.getByText('intervention channels')` (NOT `/intervention channels/i`) — the regex also matches CausalScopeMap's default HCP summary sentence ("…eight intervention channels simulatable…", content.ts SCOPE_LEVELS), visible at mount since Task 6. (b) In "shows the governed-KPIs chip when the query succeeds": `screen.getByText('governed KPIs')` and (c) in the error test: `screen.queryByText('governed KPIs')).not.toBeInTheDocument()` (NOT `/governed kpis/i`) — the regex also matches ImpactPathways' always-rendered "Faster time-to-insight" mechanism sentence ("Natural-language chat over governed KPIs…", content.ts IMPACT_PATHWAYS), which only surfaces once §4 mounts. RTL's string matcher requires a FULL normalized-text exact match, so only the chip label divs (exact text `intervention channels` / `governed KPIs`) can match; the prose paragraphs are longer sentences and never match, in success or error state. `getByText('46')` verified collision-free (no "46" anywhere in content.ts). Component code (`ImpactPathways.tsx`) is byte-exact to plan; the mock-idiom line follows item 2. Content copy was deliberately NOT reworded — tests were sharpened instead.

**Task 14 gate-expectation deviation (commit fa41d1ad):**
18. Task 14's Step 1 zero-consumer grep legitimately returns THREE hits, not the plan's expected "ONLY comment lines in kpi/index.ts". The two extra hits are prose-only JSDoc in Documentation-page files created during Tasks 2–13 — i.e. AFTER the plan's expectation was authored — narrating the supersession: `frontend/src/components/documentation/AgentTierStack.tsx:3` ("successor to the deleted kpi/AgenticMethodology.tsx; roster source:") and `frontend/src/components/documentation/content.test.ts:6` ("the old AgenticMethodology component go stale."). A FOURTH prose mention, `frontend/src/components/documentation/index.ts:3` ("Supersedes frontend/src/components/kpi/AgenticMethodology.tsx (removed by this feature)", item 10), is invisible to the gate command because its line text contains the literal path `components/kpi/AgenticMethodology.tsx`, which the gate's `grep -v` (intended to exclude only the file itself) also filters. All four references were independently re-verified by the coordinator as comments with ZERO imports, JSX usages, or `lazy()` references anywhere in `frontend/src` / `frontend/e2e`; the deletion premise (user-approved supersession by /documentation) stands. Ruling: proceed with Steps 2–3 exactly as written; NO wording changes to any of the four comments — "deleted"/"removed" phrasing was written anticipating this task and becomes literally true at this commit; the rest stays accurate as history. `npx tsc -p tsconfig.app.json` clean and `KPIDictionary.test.tsx` 30/30 green post-deletion confirm no dangling references. Anyone re-running the Step 1 gate after this commit should expect exactly the three prose hits above (kpi/index.ts:13 now bearing the replacement comment) and treat them as authorized.

**Task 16 test-list adjustment (no code commit — verification ran green):**
19. Task 16's Step 2 vitest invocation ran SIX files, not the plan's printed five: the plan's list plus `src/components/documentation/SectionNav.test.tsx`. The step's own criterion is "every test file this branch touched or could affect"; `git diff --name-only main...HEAD -- '*.test.*'` shows the branch touched exactly five test files, one of which — SectionNav.test.tsx, created during Task 3's quality-review fixes (item 9) — postdates the plan's list and was omitted from it. (KPIDictionary.test.tsx remains in the run set as the untouched "could affect" case: Task 14 edited the kpi barrel it imports through.) Result: 6 files, 66/66 tests passed, tsc exit 0, eslint clean, no fix commit produced. Anyone re-running Task 16's Step 2 should use the six-file list.
