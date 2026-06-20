/**
 * Feature-importance strategic interpreter.
 * =========================================
 *
 * Pure, deterministic, NO LLM. Turns the grouped (raw-covariate) SHAP importance
 * the page already computes into a small strategic read: which covariate drives
 * the prediction, by how much, in which direction, whether importance is
 * concentrated or spread, and what contributes negligibly — plus honest caveats.
 *
 * Design (mirrors model-performance/interpret.ts, PR #1061):
 *  - Operates on the SAME `CovariateGroup[]` the Feature Rankings render, so the
 *    interpretation never disagrees with what's on screen.
 *  - Returns `available:false` (honest, never a fabricated insight) when there is
 *    no usable signal (no covariates, or all-zero importance).
 *  - Adds INTERPRETATION beyond restating the ranking (share, direction,
 *    concentration, what's negligible, an implication) so it is not a tautology.
 *  - Says "association under the model, not causal effect" — SHAP importance is
 *    not a causal estimate (those live on the causal-analysis pages).
 *
 * @module lib/feature-importance/interpret
 */

import type { CovariateGroup } from '@/lib/shap-covariates';
import type { FeatureContribution } from '@/types/explain';

/** A covariate ≥ this share of total importance makes the ranking "concentrated". */
const CONCENTRATED_SHARE = 0.6;
/** A covariate < this share of total importance is "negligible". */
const NEGLIGIBLE_SHARE = 0.02;

export interface DriverInsight {
  /** Raw covariate name (e.g. `disease_severity`). */
  covariate: string;
  /** Human-friendly label (underscores → spaces). */
  label: string;
  /** Fraction of total importance in [0, 1]. */
  share: number;
  /** Net signed direction of the covariate's effect on the prediction. */
  direction: 'positive' | 'negative' | 'neutral';
  /** Summed |SHAP| importance (the ranking magnitude). */
  magnitude: number;
}

export interface ImportanceInsight {
  /** False when there is no usable signal — the UI shows an honest empty state. */
  available: boolean;
  /** One-line strategic summary. */
  headline: string;
  /** The top driver, or null when unavailable. */
  dominant: DriverInsight | null;
  /** All drivers, ranked by importance desc. */
  drivers: DriverInsight[];
  /** Whether importance concentrates in one driver or spreads across several. */
  concentration: 'concentrated' | 'balanced' | 'n/a';
  /** Human labels of covariates contributing a negligible share. */
  negligible: string[];
  /** Strategic bullet statements (empty when unavailable). */
  statements: string[];
  /** Honest caveats (sample size, association-not-causation). */
  caveats: string[];
}

export interface InterpretOpts {
  /** Friendly cohort label, e.g. "Initiation" / "HCP Adoption". */
  modelLabel: string;
  /** Per-brand model the importance was computed for. */
  brand: string;
  /** Entities the mean |SHAP| was averaged over (the honest sample n). */
  sampleSize: number;
  /** Cohort grain — drives the "patients" vs "HCPs" wording. */
  grain: 'patient' | 'hcp';
}

/** Underscores → spaces (matches the ranking's display transform). */
function humanize(name: string): string {
  return name.replace(/_/g, ' ');
}

/** Outcome phrase for prose, derived from the cohort label (no hardcoded magic). */
function outcomePhrase(modelLabel: string): string {
  const key = modelLabel.toLowerCase();
  if (key.includes('initiation')) return 'treatment initiation';
  if (key.includes('persistence')) return 'treatment persistence';
  if (key.includes('discontinuation')) return 'discontinuation';
  if (key.includes('adoption')) return 'HCP adoption';
  return `the ${key} outcome`;
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

/** Join labels into a readable list ("a", "a and b", "a, b and c"). */
function joinList(items: string[]): string {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/**
 * Interpret the grouped (raw-covariate) global SHAP importance.
 *
 * `groups` is expected ranked by importance desc (as `groupByCovariate`
 * produces); we do not re-sort, so the displayed #1 row is the dominant driver.
 */
export function interpretGlobalImportance(
  groups: CovariateGroup<FeatureContribution>[],
  opts: InterpretOpts,
): ImportanceInsight {
  const grainNoun = opts.grain === 'hcp' ? 'HCPs' : 'patients';
  const total = groups.reduce((acc, g) => acc + Math.max(0, g.importance), 0);

  if (groups.length === 0 || total <= 0) {
    return {
      available: false,
      headline: 'Not enough signal to interpret feature importance.',
      dominant: null,
      drivers: [],
      concentration: 'n/a',
      negligible: [],
      statements: [],
      caveats: [],
    };
  }

  const drivers: DriverInsight[] = groups.map((g) => ({
    covariate: g.covariate,
    label: humanize(g.covariate),
    share: g.importance / total,
    direction: g.importance === 0 ? 'neutral' : g.signed > 0 ? 'positive' : g.signed < 0 ? 'negative' : 'neutral',
    magnitude: g.importance,
  }));

  const dominant = drivers[0];
  const concentration: 'concentrated' | 'balanced' =
    dominant.share >= CONCENTRATED_SHARE ? 'concentrated' : 'balanced';
  const negligible = drivers.filter((d) => d.share < NEGLIGIBLE_SHARE).map((d) => d.label);

  const outcome = outcomePhrase(opts.modelLabel);
  const headline = `${dominant.label} is the dominant driver of ${outcome} (${pct(
    dominant.share,
  )} of total importance).`;

  const statements: string[] = [];

  // Direction of the dominant driver — adds the "which way" the ranking's color
  // hints at, in plain language. Phrased as an ASSOCIATION (net signed effect
  // across the sample) rather than "higher X", so it stays valid for a
  // categorical covariate (e.g. geographic_region) that has no ordinal "higher".
  if (dominant.direction === 'positive') {
    statements.push(
      `On average, ${dominant.label} is associated with a higher predicted likelihood of ${outcome}.`,
    );
  } else if (dominant.direction === 'negative') {
    statements.push(
      `On average, ${dominant.label} is associated with a lower predicted likelihood of ${outcome}.`,
    );
  }

  // Concentration → an implication, not just a label.
  const meaningful = drivers.filter((d) => d.share >= NEGLIGIBLE_SHARE).length;
  if (concentration === 'concentrated') {
    statements.push(
      `Importance is concentrated in ${dominant.label} — a small number of signals dominate, so targeting can lean on it.`,
    );
  } else {
    statements.push(
      `Importance is spread across ${meaningful} drivers — no single covariate dominates, so a multi-factor view is warranted.`,
    );
  }

  // What does NOT matter — directly answers the "is X redundant?" worry, honestly.
  if (negligible.length > 0) {
    statements.push(
      `${joinList(negligible)} ${negligible.length === 1 ? 'contributes' : 'contribute'} negligibly — unlikely to move this prediction.`,
    );
  }

  const caveats: string[] = [
    `Computed as mean |SHAP| over a ${opts.sampleSize}-${grainNoun.slice(0, -1)} sample of the cohort (not the cohort size).`,
    'SHAP shows associations under the deployed model, not causal effects — use the causal analysis pages for causal estimates.',
  ];

  return {
    available: true,
    headline,
    dominant,
    drivers,
    concentration,
    negligible,
    statements,
    caveats,
  };
}
