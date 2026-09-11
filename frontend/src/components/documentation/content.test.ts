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
  it('has five tests with unique ids, exactly two critical, each with a default and a pass rule', () => {
    expect(REFUTATION_TESTS).toHaveLength(5);
    expect(new Set(REFUTATION_TESTS.map((t) => t.id)).size).toBe(5);
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
    // Both warning readings are pinned: the caveat and the null finding.
    expect(sens!.failSign).toMatch(/could account for the whole effect/i);
    expect(sens!.failSign).toMatch(/null finding/i);
  });

  it('says two tests are critical', () => {
    expect(REFUTATION_INTRO).toMatch(/Two are critical/);
    expect(REFUTATION_TESTS.filter((t) => t.critical).map((t) => t.id).sort()).toEqual([
      'placebo_treatment',
      'random_common_cause',
    ]);
  });
});
