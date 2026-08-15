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
  '/admin': {
    question: 'Who has access to the platform, and what have they been doing? (admin only)',
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

/** Roster source: src/agents/factory.py AGENT_REGISTRY_CONFIG (22 agents).
 * Pinned against the registry by tests/unit/test_agents/test_agent_roster_ssot_1638.py —
 * cohort_profiler was missing here, so this page under-reported the system (#1638). */
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
      { id: 'cohort_profiler', role: 'Profiles cohorts from chat with real KPI counts' },
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
