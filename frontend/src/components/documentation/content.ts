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

// ── Predictive cohorts ──────────────────────────────────────────────────────
// SSOT: src/api/schemas/causal.py CohortName — order and ids mirror the enum;
// each cohort selects ONE outcome column as the model's label. Pinned by
// tests/unit/test_docs/test_documentation_cohorts_channels_ssot.py.

export type PredictiveCohortId = 'initiation' | 'persistence' | 'discontinuation' | 'hcp_adoption';
export type CohortEntity = 'patients' | 'prescribers (HCPs)';

export interface PredictiveCohortDef {
  id: PredictiveCohortId;
  name: string;
  /** Who gets a score. */
  entity: CohortEntity;
  /** The outcome the model predicts, in plain words. */
  outcome: string;
  /** The backend label column (table.column) the cohort is defined by. */
  labelColumn: string;
}

export const PREDICTIVE_COHORT_INTRO =
  'A predictive cohort is a population the platform scores one member at a time. A model trained on the cohort gives every patient or prescriber a probability of one specific outcome, and SHAP explains which features drove each score — so a team knows not just who to target, but why. Four cohorts are evaluated, one per outcome:';

export const PREDICTIVE_COHORTS: PredictiveCohortDef[] = [
  {
    id: 'initiation',
    name: 'Treatment initiation',
    entity: 'patients',
    outcome: 'starting treatment',
    labelColumn: 'patient_journeys.treatment_initiated',
  },
  {
    id: 'persistence',
    name: 'Persistence',
    entity: 'patients',
    outcome: 'staying on therapy at 180 days',
    labelColumn: 'patient_journeys.persistent_180d',
  },
  {
    id: 'discontinuation',
    name: 'Discontinuation',
    entity: 'patients',
    outcome: 'discontinuing therapy within 180 days',
    labelColumn: 'patient_journeys.discontinued_180d',
  },
  {
    id: 'hcp_adoption',
    name: 'HCP adoption',
    entity: 'prescribers (HCPs)',
    outcome: 'adopting the brand (intent to prescribe)',
    labelColumn: 'hcp_brand_adoption.adopted',
  },
];

// ── Intervention channels ───────────────────────────────────────────────────
// SSOT: src/digital_twin/effect/provider.py INTERVENTION_CATALOG (value + label,
// same order) and INTERVENTION_TREATMENT_MAP (the lever = treatment column).
// Six channel-level HCP interventions + two program-level levers modeled as
// HCP-level proxies (user-approved taxonomy, 2026-07-08).

export type InterventionChannelKind = 'hcp' | 'program';

export interface InterventionChannelDef {
  id: string;
  name: string;
  kind: InterventionChannelKind;
  /** What is actually varied — the treatment variable the engine estimates on. */
  lever: string;
}

export const INTERVENTION_CHANNEL_INTRO =
  'An intervention channel is a lever the commercial team can actually pull. Each one is a treatment the causal engine estimates and the Digital Twin simulates; where a channel\'s exposure is not recorded in the data, the platform reports it as unavailable rather than inventing an effect. Eight channels are modeled — six at the HCP level, plus two program-level levers represented as HCP-level proxies:';

export const INTERVENTION_CHANNELS: InterventionChannelDef[] = [
  { id: 'email_campaign', name: 'Email Campaign', kind: 'hcp', lever: 'email campaign count' },
  { id: 'call_frequency_increase', name: 'Increased Call Frequency', kind: 'hcp', lever: 'rep call frequency' },
  { id: 'speaker_program_invitation', name: 'Speaker Program Invitation', kind: 'hcp', lever: 'speaker program count' },
  { id: 'sample_distribution', name: 'Sample Distribution', kind: 'hcp', lever: 'sample volume' },
  { id: 'peer_influence_activation', name: 'Peer Influence Activation', kind: 'hcp', lever: 'peer influence score' },
  { id: 'digital_engagement', name: 'Digital Engagement', kind: 'hcp', lever: 'digital engagement score' },
  {
    id: 'patient_support_program',
    name: 'Patient Support Program',
    kind: 'program',
    lever: 'share of the HCP\'s patients enrolled in patient-support programs',
  },
  {
    id: 'rep_training_quality',
    name: 'Rep Training Quality',
    kind: 'program',
    lever: 'territory rep-training quality experienced by the HCP',
  },
];

/** Static structural chips. The live KPI-count chip is fetched on the page.
 * Cohort and channel counts are DERIVED from the lists above, never typed. */
export const STAT_CHIPS: StatChip[] = [
  { value: '3 / 4', label: 'brands / indications' },
  { value: String(PREDICTIVE_COHORTS.length), label: 'predictive cohorts' },
  { value: String(INTERVENTION_CHANNELS.length), label: 'intervention channels' },
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
      'Downstream surfaces (Resource Optimization, Gap Analysis, Digital Twin simulation, Executive Insights) consume gated estimates with provenance labels. Narrative insight surfaces are digit-guarded: language models never invent figures, they interpret server-injected validated numbers.',
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

// ── Causal Impact — variable types + the illustrative DAG ───────────────────
// Ported from docs/Archive/CausalImpactAgent.html ("CausalImpactAgent — Visual
// Explainer"): the color key, the Acceptance → Action → Revenue DAG and its
// path toggles. Edge labels are that explainer's own hand-authored figures —
// the figure is labeled "illustrative example" on the page; nothing here is a
// measured effect.

export type CausalVariableType = 'treatment' | 'mediator' | 'outcome' | 'confounder';

export interface CausalVariableDef {
  type: CausalVariableType;
  label: string;
  definition: string;
  /** One color per role, shared by the explainer chips and the DAG nodes. */
  color: string;
  /** DAG node shape — confounders are diamonds, everything else circles. */
  shape: 'circle' | 'diamond';
}

export const CAUSAL_EFFECTS_INTRO =
  'Causal effects quantify the impact of a treatment by comparing outcomes across different treatment levels.';

export const CAUSAL_VARIABLES_LEAD =
  'When thinking about questions of cause and effect, it is helpful to distinguish 4 types of variables:';

export const CAUSAL_VARIABLE_TYPES: CausalVariableDef[] = [
  {
    type: 'treatment',
    label: 'Treatment',
    definition: 'Variable whose causal effect you want to estimate.',
    color: '#0ea5e9',
    shape: 'circle',
  },
  {
    type: 'mediator',
    label: 'Mediator',
    definition: 'Transmits part of the effect from exposure to outcome.',
    color: '#a855f7',
    shape: 'circle',
  },
  {
    type: 'outcome',
    label: 'Outcome',
    definition: 'The result whose causal determinants you are studying.',
    color: '#22c55e',
    shape: 'circle',
  },
  {
    type: 'confounder',
    label: 'Confounder',
    definition:
      'Pre-exposure variable that affects both exposure and outcome, opening a back-door path.',
    color: '#f59e0b',
    shape: 'diamond',
  },
];

export type DagPathGroup =
  | 'core'
  | 'mediation'
  | 'path1'
  | 'path2'
  | 'path3'
  | 'confounders'
  | 'loop';

export interface DagNode {
  id: string;
  label: string;
  type: CausalVariableType;
  x: number;
  y: number;
  /** Put the caption above the node (top-row nodes whose edge below is captioned). */
  labelAbove?: boolean;
}

export interface DagEdge {
  from: string;
  to: string;
  label: string;
  group: DagPathGroup;
  dashed?: boolean;
  /** Where along the edge (0 → 1) the caption sits; default midpoint. Used to
   *  keep the captions of crossing edges apart. */
  labelT?: number;
}

export interface DagPath {
  group: DagPathGroup;
  label: string;
  description: string;
}

export const DAG_TITLE = 'Acceptance → Action → Multi-path Revenue Impact';

// Coordinates are in the SVG's 1100 × 400 viewBox.
export const DAG_NODES: DagNode[] = [
  { id: 'acceptance', label: 'Acceptance Rate', type: 'treatment', x: 140, y: 120 },
  { id: 'engagement', label: 'Engagement Quality', type: 'mediator', x: 340, y: 80, labelAbove: true },
  { id: 'action', label: 'Action Rate', type: 'mediator', x: 340, y: 160 },
  { id: 'patient_id', label: 'Patient Identification', type: 'mediator', x: 540, y: 110 },
  { id: 'hcp_sat', label: 'HCP Satisfaction', type: 'mediator', x: 540, y: 190 },
  { id: 'treat_init', label: 'Treatment Initiation', type: 'mediator', x: 740, y: 110 },
  { id: 'brand_adopt', label: 'Brand Adoption', type: 'mediator', x: 740, y: 190 },
  { id: 'revenue', label: 'Revenue', type: 'outcome', x: 940, y: 150 },
  { id: 'feedback', label: 'Data Feedback', type: 'mediator', x: 540, y: 280 },
  { id: 'model_improve', label: 'Model Improvement', type: 'mediator', x: 740, y: 280 },
  { id: 'future_precision', label: 'Future Precision', type: 'mediator', x: 940, y: 280 },
  { id: 'seasonality', label: 'Seasonality', type: 'confounder', x: 140, y: 270 },
  { id: 'channel_mix', label: 'Channel Mix', type: 'confounder', x: 340, y: 270 },
];

export const DAG_EDGES: DagEdge[] = [
  { from: 'acceptance', to: 'action', label: '+12pp / 10pp', group: 'core' },
  { from: 'acceptance', to: 'engagement', label: 'mediates ~30%', group: 'mediation' },
  { from: 'engagement', to: 'action', label: '+quality → +action', group: 'mediation' },
  { from: 'action', to: 'patient_id', label: '', group: 'path1' },
  { from: 'patient_id', to: 'treat_init', label: '42%', group: 'path1' },
  { from: 'treat_init', to: 'revenue', label: '', group: 'path1' },
  { from: 'action', to: 'hcp_sat', label: '', group: 'path2' },
  { from: 'hcp_sat', to: 'brand_adopt', label: '28%', group: 'path2' },
  { from: 'brand_adopt', to: 'revenue', label: '', group: 'path2' },
  { from: 'action', to: 'feedback', label: '', group: 'path3' },
  { from: 'feedback', to: 'model_improve', label: '', group: 'path3' },
  { from: 'model_improve', to: 'future_precision', label: '15%', group: 'path3' },
  { from: 'future_precision', to: 'revenue', label: '', group: 'path3' },
  { from: 'seasonality', to: 'acceptance', label: 'confounds', group: 'confounders' },
  { from: 'seasonality', to: 'action', label: 'confounds', group: 'confounders', labelT: 0.25 },
  { from: 'channel_mix', to: 'acceptance', label: 'confounds', group: 'confounders', labelT: 0.75 },
  { from: 'channel_mix', to: 'action', label: 'confounds', group: 'confounders' },
  { from: 'revenue', to: 'feedback', label: 'learning loop', group: 'loop', dashed: true },
];

export const DAG_PATHS: DagPath[] = [
  {
    group: 'core',
    label: 'Core: Acceptance → Action',
    description:
      'The direct effect the agent estimates: does accepting a trigger raise the action rate?',
  },
  {
    group: 'mediation',
    label: 'Mediation branch',
    description:
      'Part of the acceptance effect travels through engagement quality before it reaches action.',
  },
  {
    group: 'path1',
    label: 'Path 1: Patient → Treat → Revenue',
    description: 'Actions identify patients, some of whom go on to initiate treatment.',
  },
  {
    group: 'path2',
    label: 'Path 2: HCP → Brand → Revenue',
    description: 'Actions build HCP satisfaction, which feeds brand adoption.',
  },
  {
    group: 'path3',
    label: 'Path 3: Feedback → Precision → Revenue',
    description:
      'Actions generate data that improves the model and sharpens the precision of future triggers.',
  },
  {
    group: 'confounders',
    label: 'Backdoor confounders',
    description:
      'Seasonality and channel mix affect both acceptance and action — the back-door paths the estimator must close before the core effect can be read.',
  },
  {
    group: 'loop',
    label: 'Reinforcing loop',
    description:
      'Revenue funds the data feedback that improves future precision — a learning loop, not a one-shot effect.',
  },
];

// ── Section nav ─────────────────────────────────────────────────────────────

// ---------------------------------------------------------------------------
// Quality gate — the five refutation tests (Documentation §Quality Gate).
// Defaults and pass rules mirror src/causal_engine/refutation_runner.py
// (RefutationRunner.DEFAULT_CONFIG / PASS_THRESHOLDS / GATE_THRESHOLDS and
// _determine_gate_decision). Keep them in sync when the runner changes.
// ---------------------------------------------------------------------------

export type RefutationTestId =
  | 'placebo_treatment'
  | 'random_common_cause'
  | 'data_subset'
  | 'bootstrap'
  | 'sensitivity_e_value';

export interface RefutationTestDef {
  id: RefutationTestId;
  name: string;
  /** What the test does to the estimate (one line). */
  action: string;
  /** What must be true for the estimate to survive. */
  mustHold: string;
  /** Production default from RefutationRunner.DEFAULT_CONFIG. */
  defaults: string;
  /** Pass rule from RefutationRunner.PASS_THRESHOLDS. */
  passRule: string;
  /** A failing result on its own blocks the estimate. */
  critical: boolean;
  /** What a failing result means, in plain language. */
  failSign: string;
}

export const REFUTATION_INTRO =
  'No causal estimate is reported until it survives five refutation tests — adversarial attacks that try to break it. Three are critical: a single failure blocks the estimate outright. All five feed a weighted confidence score that decides the gate.';

export const REFUTATION_TESTS: RefutationTestDef[] = [
  {
    id: 'placebo_treatment',
    name: 'Placebo Treatment',
    action: 'Replace treatment with noise',
    mustHold: 'effect must vanish',
    defaults: '30 permutations of the treatment column',
    passRule: 'placebo p-value > 0.05',
    critical: true,
    failSign:
      'A shuffled treatment still "moves" the outcome — the estimator is reading structure that has nothing to do with the treatment.',
  },
  {
    id: 'random_common_cause',
    name: 'Random Common Cause',
    action: 'Add a random confounder',
    mustHold: 'effect must hold stable',
    defaults: '20 simulations, confounder strength 0.1',
    passRule: 'effect moves by < 20 %',
    critical: true,
    failSign:
      'The estimate swings with a confounder that carries no information — the adjustment is fragile.',
  },
  {
    id: 'data_subset',
    name: 'Data Subset',
    action: '80 % subset replications',
    mustHold: 'effect must reproduce',
    defaults: '5 subsets, each 80 % of the rows',
    passRule: '≥ 80 % of subset effects inside the original CI',
    critical: false,
    failSign:
      'Subset estimates scatter outside the original interval — the effect depends on which rows were used.',
  },
  {
    id: 'bootstrap',
    name: 'Bootstrap',
    action: 'Resample the data with replacement',
    mustHold: 'variance must stay bounded',
    defaults: '50 resamples',
    passRule: 'bootstrap CI ≤ 1.5× the original width',
    critical: false,
    failSign:
      'Resampled estimates spread far wider than the reported interval — the stated precision is overstated.',
  },
  {
    id: 'sensitivity_e_value',
    name: 'Sensitivity (E-value)',
    action: 'How strong would a hidden confounder have to be to explain the effect away?',
    mustHold: 'robustness to unmeasured confounding',
    defaults: 'E-value on the point estimate and on the CI bound',
    passRule: 'E-value ≥ 2.0',
    critical: true,
    failSign:
      'A weak unmeasured confounder could produce the whole effect — nothing in the data rules it out.',
  },
];

/**
 * Latent-confounding diagnostic (FCI) — shown beside the refutation tests
 * because it answers a sibling question. Mirrors the surfacing policy in
 * src/agents/causal_impact/nodes/interpretation.py and the measured limits in
 * tests/unit/test_causal_engine/test_discovery/test_structural_recovery.py
 * (docstring item 6).
 */
export const LATENT_DIAGNOSTIC_NOTE =
  'Alongside these tests, every discovery run includes an FCI-based latent-confounding ' +
  'diagnostic: a check for signs that the treatment–outcome relationship is driven by an ' +
  'unmeasured common cause. Its warning appears only when the E-value independently ' +
  'indicates limited robustness — a corroborated alarm, not a reflex. Two honest limits: ' +
  'the diagnostic can only detect confounding strong enough to account for the entire ' +
  'treatment–outcome dependence, and its silence is NOT evidence that no unmeasured ' +
  'confounding exists — quantitative robustness always rests with the E-value.';

export type GateDecision = 'proceed' | 'review' | 'block';

export interface GateBand {
  decision: GateDecision;
  label: string;
  rule: string;
  consequence: string;
}

export const GATE_BANDS: GateBand[] = [
  {
    decision: 'proceed',
    label: 'Proceed',
    rule: 'Confidence ≥ 0.70 and no critical test failed',
    consequence: 'The estimate is validated and flows into recommendations.',
  },
  {
    decision: 'review',
    label: 'Review',
    rule: 'Confidence between 0.50 and 0.70',
    consequence: 'Borderline — surfaced with a caveat and queued for expert review.',
  },
  {
    decision: 'block',
    label: 'Block',
    rule: 'Any critical test failed, or confidence < 0.50',
    consequence: 'The estimate is marked refuted and never reaches a decision.',
  },
];

export interface DocSection {
  id: string;
  label: string;
}

export const DOC_SECTIONS: DocSection[] = [
  { id: 'purpose', label: 'Purpose' },
  { id: 'causal-impact', label: 'Causal Impact' },
  { id: 'refutation-gate', label: 'Quality Gate' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'practices', label: 'Best Practices' },
  { id: 'impact', label: 'Expected Impact' },
];
