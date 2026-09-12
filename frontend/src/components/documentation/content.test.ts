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
  REFUTATION_INTRO,
  REFUTATION_TESTS,
  type RefutationTestId,
  GATE_BANDS,
  DOC_SECTIONS,
  PREDICTIVE_COHORTS,
  INTERVENTION_CHANNELS,
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

  // #1638: was 21, and cohort_profiler — a real, dispatched agent — was missing,
  // so this page under-reported the system. The Python-side pin
  // (tests/unit/test_agents/test_agent_roster_ssot_1638.py) compares this roster
  // against factory.AGENT_REGISTRY_CONFIG, so adding an agent there fails HERE.
  it('models exactly 22 agents across 6 tiers with unique ids', () => {
    expect(AGENT_TIERS).toHaveLength(6);
    const ids = AGENT_TIERS.flatMap((t) => t.agents.map((a) => a.id));
    expect(ids).toHaveLength(22);
    expect(new Set(ids).size).toBe(22);
    expect(ids).toContain('cohort_profiler');
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

describe('refutation gate content', () => {
  it('has six tests with unique ids, exactly two critical, each with a default and a pass rule', () => {
    expect(REFUTATION_TESTS).toHaveLength(6);
    expect(new Set(REFUTATION_TESTS.map((t) => t.id)).size).toBe(6);
    expect(REFUTATION_TESTS.filter((t) => t.critical).map((t) => t.id)).toEqual([
      'placebo_treatment',
      'random_common_cause',
    ]);
    for (const t of REFUTATION_TESTS) {
      expect(t.defaults.length).toBeGreaterThan(0);
      expect(t.passRule.length).toBeGreaterThan(0);
      expect(t.mustHold.length).toBeGreaterThan(0);
    }
  });

  it('scores random common cause in SE units, never as a percentage of the effect (#2005)', () => {
    const rcc = REFUTATION_TESTS.find((t) => t.id === 'random_common_cause');
    expect(rcc).toBeDefined();
    expect(rcc!.critical).toBe(true);
    expect(rcc!.passRule).toMatch(/\bSE\b/);
    // Guard sanity: the same literal must reject the old copy.
    expect('effect moves by < 20 %').toMatch(/%/);
    expect(rcc!.passRule).not.toMatch(/%/);
  });

  it('describes the three gate bands in order', () => {
    expect(GATE_BANDS.map((b) => b.decision)).toEqual(['proceed', 'review', 'block']);
  });

  it('lists the quality gate section right after causal impact in the nav', () => {
    const ids = DOC_SECTIONS.map((s) => s.id);
    expect(ids.indexOf('refutation-gate')).toBe(ids.indexOf('causal-impact') + 1);
  });
});

// Backend SSOT: src/api/schemas/causal.py CohortName (4) and
// src/digital_twin/effect/provider.py INTERVENTION_CATALOG (8). The Python-side
// pin tests/unit/test_docs/test_documentation_cohorts_channels_ssot.py compares
// these lists against those sources, so a backend change fails HERE.
describe('purpose: predictive cohorts and intervention channels', () => {
  it('names the four cohorts in CohortName order, each with who is scored and which outcome', () => {
    expect(PREDICTIVE_COHORTS.map((c) => c.id)).toEqual([
      'initiation',
      'persistence',
      'discontinuation',
      'hcp_adoption',
    ]);
    for (const c of PREDICTIVE_COHORTS) {
      expect(c.name.length).toBeGreaterThan(0);
      expect(['patients', 'prescribers (HCPs)']).toContain(c.entity);
      expect(c.outcome.length).toBeGreaterThan(0);
      expect(c.labelColumn).toMatch(/^[a-z_]+\.[a-z_0-9]+$/);
    }
  });

  it('names the eight channels in catalog order: six HCP-level, two program-level', () => {
    expect(INTERVENTION_CHANNELS.map((c) => c.id)).toEqual([
      'email_campaign',
      'call_frequency_increase',
      'speaker_program_invitation',
      'sample_distribution',
      'peer_influence_activation',
      'digital_engagement',
      'patient_support_program',
      'rep_training_quality',
    ]);
    expect(INTERVENTION_CHANNELS.filter((c) => c.kind === 'hcp')).toHaveLength(6);
    expect(INTERVENTION_CHANNELS.filter((c) => c.kind === 'program')).toHaveLength(2);
    for (const c of INTERVENTION_CHANNELS) expect(c.lever.length).toBeGreaterThan(0);
  });

  it('derives the cohort and channel stat chips from the lists (never a transcribed digit)', () => {
    const chip = (label: string) => STAT_CHIPS.find((c) => c.label === label)?.value;
    expect(chip('predictive cohorts')).toBe(String(PREDICTIVE_COHORTS.length));
    expect(chip('intervention channels')).toBe(String(INTERVENTION_CHANNELS.length));
  });
});

// Lane D′ (2026-09-10, #1991): the E-value sensitivity test became a non-critical
// reading benchmarked per run against the confounding the adjustment removed.
// Only placebo and random_common_cause block on their own.
describe('refutation documentation content (2026-09-10 sensitivity reading)', () => {
  it('lists sensitivity as a non-critical reading with the benchmark rule', () => {
    const sens = REFUTATION_TESTS.find((t) => t.id === 'sensitivity_e_value');
    expect(sens).toBeDefined();
    expect(sens!.critical).toBe(false);
    expect(sens!.passRule).toMatch(/measured confounding/i);
    // Guard sanity: the same literal must reject the old copy.
    expect('E-value ≥ 2.0').toMatch(/2\.0/);
    expect(sens!.passRule).not.toMatch(/2\.0/);
    // All three warning readings are pinned: sensitive, null finding, not benchmarked
    // (both sub-cases), plus the reading order and the minimum cell size.
    expect(sens!.failSign).toMatch(/could account for the whole effect/i);
    expect(sens!.failSign).toMatch(/measured-confounding benchmark/i);
    expect(sens!.failSign).not.toMatch(/measured set/i);
    expect(sens!.failSign).toMatch(/null finding/i);
    expect(sens!.failSign).toMatch(/not benchmarked|unbenchmarked/i);
    expect(sens!.failSign).toMatch(/no measured confounders/i);
    expect(sens!.failSign).toMatch(/collinear|could not be scored/i);
    expect(sens!.passRule).toMatch(/randomized/i);
    expect(sens!.passRule).toMatch(/null finding/i);
    expect(sens!.passRule).toMatch(/benchmarked, non-null/i);
    expect(sens!.defaults).toMatch(/naive-vs-adjusted risk ratio/i);
    expect(sens!.defaults).toMatch(/fewer than 5 rows/i);
  });

  it('says two tests are critical', () => {
    expect(REFUTATION_INTRO).toMatch(/Two are critical/);
    expect(REFUTATION_TESTS.filter((t) => t.critical).map((t) => t.id).sort()).toEqual([
      'placebo_treatment',
      'random_common_cause',
    ]);
  });
});

// Lane G (2026-09-11, #2007): the negative-control-outcome test — the only
// refuter that can DETECT confounding (the others perturb or resample the same
// fit). A weight-0 READING for the first live period: non-critical, never
// blocks, never moves the confidence score.
describe('refutation documentation content (2026-09-11 negative-control reading)', () => {
  it('lists the negative-control outcome as a sixth, non-critical reading', () => {
    // Compile-time pin: the id is a member of the RefutationTestId union.
    const id: RefutationTestId = 'negative_control_outcome';
    const nc = REFUTATION_TESTS.find((t) => t.id === id);
    expect(nc).toBeDefined();
    expect(nc!.critical).toBe(false);
    expect(nc!.name).toBe('Negative-Control Outcome');
    expect(nc!.mustHold).toMatch(/must stay null/i);
    // The rule is an interval rule on the control's CI, three-way.
    expect(nc!.passRule).toMatch(/includes 0/);
    expect(nc!.passRule).toMatch(/warns/);
    expect(nc!.passRule).toMatch(/fails/);
    // Weight 0 — a reading, declared per treatment.
    expect(nc!.defaults).toMatch(/weight 0/);
    expect(nc!.defaults).toMatch(/per treatment/i);
    // The failing sign names the mechanism: the adjustment leaks confounding.
    expect(nc!.failSign).toMatch(/leaking confounding/i);
    expect(REFUTATION_TESTS.map((t) => t.id).indexOf(id)).toBe(5);
  });

  it('counts six tests in the intro and says the negative control is the one that can detect confounding', () => {
    expect(REFUTATION_INTRO).toMatch(/six refutation tests/);
    expect(REFUTATION_INTRO).not.toMatch(/five refutation tests/i);
    expect(REFUTATION_INTRO).toMatch(/detect confounding/i);
    expect(REFUTATION_INTRO).toMatch(/reading/i);
  });
});
