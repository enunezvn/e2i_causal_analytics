/**
 * Tests for the rule-based feature-importance strategic interpreter.
 * ==================================================================
 *
 * Pure, deterministic, NO LLM. Mirrors the model-performance interpret.ts
 * pattern (PR #1061): derive strategic statements from the REAL grouped SHAP
 * importance, return an honest `available:false` (never fabricated) when there
 * is no usable signal.
 */

import { describe, it, expect } from 'vitest';
import type { CovariateGroup } from '@/lib/shap-covariates';
import type { FeatureContribution } from '@/types/explain';
import { interpretGlobalImportance } from './interpret';

// Minimal CovariateGroup factory (categories are irrelevant to interpretation).
function grp(
  covariate: string,
  importance: number,
  signed: number,
  rank: number
): CovariateGroup<FeatureContribution> {
  return {
    covariate,
    importance,
    signed,
    direction: signed >= 0 ? 'positive' : 'negative',
    rank,
    categories: [],
    isGrouped: false,
  };
}

const OPTS = {
  modelLabel: 'Initiation',
  brand: 'Remibrutinib',
  sampleSize: 25,
  grain: 'patient' as const,
};

describe('interpretGlobalImportance — availability', () => {
  it('returns available:false (honest, not fabricated) for an empty ranking', () => {
    const out = interpretGlobalImportance([], OPTS);
    expect(out.available).toBe(false);
    expect(out.dominant).toBeNull();
    expect(out.concentration).toBe('n/a');
    expect(out.statements).toHaveLength(0);
    expect(out.headline).toMatch(/not enough|no .*signal|unavailable/i);
  });

  it('returns available:false when every covariate has zero importance', () => {
    const out = interpretGlobalImportance([grp('disease_severity', 0, 0, 1), grp('academic_hcp', 0, 0, 2)], OPTS);
    expect(out.available).toBe(false);
  });
});

describe('interpretGlobalImportance — dominant driver + share', () => {
  const groups = [
    grp('disease_severity', 0.8, 0.8, 1),
    grp('geographic_region', 0.2, -0.2, 2),
  ];

  it('identifies the dominant driver and its share of total importance', () => {
    const out = interpretGlobalImportance(groups, OPTS);
    expect(out.available).toBe(true);
    expect(out.dominant?.covariate).toBe('disease_severity');
    expect(out.dominant?.share).toBeCloseTo(0.8, 5);
    // human-friendly label (underscores → spaces)
    expect(out.dominant?.label).toBe('disease severity');
  });

  it('reports the dominant driver direction (raises) from the signed effect', () => {
    const out = interpretGlobalImportance(groups, OPTS);
    expect(out.dominant?.direction).toBe('positive');
    expect(out.statements.join(' ')).toMatch(/rais|increas|higher/i);
  });

  it('reports a lowering direction when the dominant net effect is negative', () => {
    const out = interpretGlobalImportance(
      [grp('disease_severity', 0.8, -0.8, 1), grp('geographic_region', 0.2, -0.2, 2)],
      OPTS
    );
    expect(out.dominant?.direction).toBe('negative');
    expect(out.statements.join(' ')).toMatch(/lower|decreas|reduc/i);
  });

  it('phrases the outcome from the model label (initiation)', () => {
    const out = interpretGlobalImportance(groups, OPTS);
    expect(out.headline.toLowerCase()).toContain('initiation');
  });
});

describe('interpretGlobalImportance — concentration', () => {
  it('flags a concentrated ranking when the top driver dominates (share ≥ 0.6)', () => {
    const out = interpretGlobalImportance(
      [grp('disease_severity', 0.85, 0.85, 1), grp('academic_hcp', 0.15, 0.15, 2)],
      OPTS
    );
    expect(out.concentration).toBe('concentrated');
    expect(out.statements.join(' ')).toMatch(/concentrat|dominat/i);
  });

  it('flags a balanced ranking when no single driver dominates', () => {
    const out = interpretGlobalImportance(
      [
        grp('a', 0.3, 0.3, 1),
        grp('b', 0.28, 0.28, 2),
        grp('c', 0.25, -0.25, 3),
        grp('d', 0.17, 0.17, 4),
      ],
      OPTS
    );
    expect(out.concentration).toBe('balanced');
    expect(out.statements.join(' ')).toMatch(/spread|balanc|no single/i);
  });
});

describe('interpretGlobalImportance — negligible drivers', () => {
  it('flags covariates contributing a negligible share (honest "what does not matter")', () => {
    const out = interpretGlobalImportance(
      [
        grp('disease_severity', 0.95, 0.95, 1),
        grp('academic_hcp', 0.04, 0.04, 2),
        grp('geographic_region', 0.01, -0.01, 3),
      ],
      OPTS
    );
    // 0.01 / 1.0 = 1% < 2% → negligible; 4% is NOT negligible.
    expect(out.negligible).toContain('geographic region');
    expect(out.negligible).not.toContain('academic hcp');
    expect(out.statements.join(' ')).toMatch(/negligibl|little|minimal/i);
  });
});

describe('interpretGlobalImportance — single covariate + caveats', () => {
  it('handles a single-covariate ranking (share = 1, concentrated)', () => {
    const out = interpretGlobalImportance([grp('disease_severity', 0.5, 0.5, 1)], OPTS);
    expect(out.available).toBe(true);
    expect(out.dominant?.share).toBeCloseTo(1, 5);
    expect(out.concentration).toBe('concentrated');
    expect(out.negligible).toHaveLength(0);
  });

  it('always emits honest caveats (sample size + association-not-causation)', () => {
    const out = interpretGlobalImportance([grp('disease_severity', 0.8, 0.8, 1)], OPTS);
    expect(out.caveats.join(' ')).toMatch(/25/); // sample size surfaced
    expect(out.caveats.join(' ')).toMatch(/sample/i);
    expect(out.caveats.join(' ')).toMatch(/causal|association/i);
  });
});
