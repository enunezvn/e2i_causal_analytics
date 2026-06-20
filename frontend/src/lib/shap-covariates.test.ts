/**
 * Tests for SHAP encoded-feature -> raw-covariate grouping.
 * ========================================================
 *
 * SHAP runs over the ENCODED vector: a numeric covariate `X` becomes `X` + an
 * `X__isna` missingness twin; a categorical `X` becomes one-hot `X_<value>`
 * columns (incl `X_nan`). Surfaced raw, a single covariate reads as many
 * duplicate rows ("geographic region" x5, "X" + "X isna"). groupByCovariate()
 * folds those encoded columns back to their parent covariate (importance = sum
 * of children) so the ranking shows one row per real covariate, expandable to
 * the per-category detail.
 */

import { describe, it, expect } from 'vitest';
import {
  parentCovariate,
  groupByCovariate,
  type Groupable,
} from './shap-covariates';

interface F {
  name: string;
  abs: number;
  signed: number;
}
const acc = (f: F): Groupable => ({ name: f.name, abs: f.abs, signed: f.signed });

// Real shape from the live initiation/Kisqali /explain/global response.
const KEEP = ['disease_severity', 'academic_hcp', 'geographic_region'];
const INITIATION: F[] = [
  { name: 'disease_severity', abs: 0.7698, signed: 0.7698 },
  { name: 'geographic_region_west', abs: 0.0955, signed: -0.0955 },
  { name: 'geographic_region_northeast', abs: 0.0727, signed: -0.0727 },
  { name: 'academic_hcp', abs: 0.0527, signed: 0.0527 },
  { name: 'geographic_region_south', abs: 0.0351, signed: -0.0351 },
  { name: 'academic_hcp__isna', abs: 0, signed: 0 },
  { name: 'disease_severity__isna', abs: 0, signed: 0 },
  { name: 'geographic_region_midwest', abs: 0, signed: 0 },
  { name: 'geographic_region_nan', abs: 0, signed: 0 },
];

describe('parentCovariate', () => {
  it('matches a bare numeric column to itself', () => {
    expect(parentCovariate('disease_severity', KEEP)).toBe('disease_severity');
  });
  it('matches a numeric __isna twin to its parent', () => {
    expect(parentCovariate('disease_severity__isna', KEEP)).toBe('disease_severity');
  });
  it('matches a one-hot category to its parent', () => {
    expect(parentCovariate('geographic_region_west', KEEP)).toBe('geographic_region');
    expect(parentCovariate('geographic_region_nan', KEEP)).toBe('geographic_region');
  });
  it('returns null for an unmatched encoded name', () => {
    expect(parentCovariate('totally_unrelated', KEEP)).toBeNull();
  });
  it('disambiguates by LONGEST matching covariate prefix', () => {
    const keep = ['region', 'region_code'];
    expect(parentCovariate('region_code_x', keep)).toBe('region_code');
    expect(parentCovariate('region_west', keep)).toBe('region');
  });
});

describe('groupByCovariate', () => {
  it('folds encoded columns into one row per raw covariate', () => {
    const groups = groupByCovariate(INITIATION, KEEP, acc);
    expect(groups.map((g) => g.covariate)).toEqual([
      'disease_severity',
      'geographic_region',
      'academic_hcp',
    ]);
  });

  it('sums children importance and signed effect per covariate', () => {
    const groups = groupByCovariate(INITIATION, KEEP, acc);
    const region = groups.find((g) => g.covariate === 'geographic_region')!;
    // importance = sum of |child| across all 5 region one-hots
    expect(region.importance).toBeCloseTo(0.0955 + 0.0727 + 0.0351, 6);
    // net signed effect (all region one-hots are negative here)
    expect(region.signed).toBeCloseTo(-(0.0955 + 0.0727 + 0.0351), 6);
    expect(region.direction).toBe('negative');
    expect(region.categories).toHaveLength(5);
  });

  it('ranks groups by aggregated importance desc with 1-based rank', () => {
    const groups = groupByCovariate(INITIATION, KEEP, acc);
    expect(groups.map((g) => g.rank)).toEqual([1, 2, 3]);
    expect(groups[0].covariate).toBe('disease_severity'); // 0.77 >> region 0.20 > academic 0.05
    expect(groups[0].importance).toBeCloseTo(0.7698, 6);
  });

  it('sorts categories within a group by |contribution| desc', () => {
    const groups = groupByCovariate(INITIATION, KEEP, acc);
    const region = groups.find((g) => g.covariate === 'geographic_region')!;
    expect((region.categories as F[]).map((c) => c.name)).toEqual([
      'geographic_region_west',
      'geographic_region_northeast',
      'geographic_region_south',
      'geographic_region_midwest',
      'geographic_region_nan',
    ]);
  });

  it('marks a multi-category / relabeled group as grouped (expandable)', () => {
    const groups = groupByCovariate(INITIATION, KEEP, acc);
    expect(groups.find((g) => g.covariate === 'geographic_region')!.isGrouped).toBe(true);
  });

  it('keeps an unmatched encoded feature as its own ungrouped row', () => {
    const feats: F[] = [
      { name: 'disease_severity', abs: 0.5, signed: 0.5 },
      { name: 'mystery_feature', abs: 0.3, signed: -0.3 },
    ];
    const groups = groupByCovariate(feats, KEEP, acc);
    const mystery = groups.find((g) => g.covariate === 'mystery_feature')!;
    expect(mystery).toBeDefined();
    expect(mystery.isGrouped).toBe(false);
    expect(mystery.categories).toHaveLength(1);
  });

  it('passes through (flat, one group per feature) when keepColumns is empty/undefined', () => {
    for (const keep of [undefined, null, []] as const) {
      const groups = groupByCovariate(INITIATION, keep, acc);
      expect(groups).toHaveLength(INITIATION.length);
      expect(groups.every((g) => g.categories.length === 1 && !g.isGrouped)).toBe(true);
      // ranked by importance desc
      expect(groups[0].covariate).toBe('disease_severity');
    }
  });
});
