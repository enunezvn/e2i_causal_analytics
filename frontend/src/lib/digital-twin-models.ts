/**
 * Digital Twin model honesty helpers (#2206)
 * ==========================================
 *
 * The backend's `/digital-twin/models` rows carry data-derived honesty fields:
 * a fit fingerprint (equal across rows = one shared fit under several brand
 * labels), what the R² was scored against, and an explicit fidelity status
 * where a NULL score is `unvalidated` — never a pass. These helpers turn that
 * census into the one-line summary and tooltip the page renders. Nothing here
 * is inferred from brand names or counts alone; every claim reads a field.
 */

import type { TwinModelSummary } from '@/types/digital-twin';

/** Human label for the explicit fidelity state. Unknown values are shown as such. */
export function fidelityStatusLabel(status: string | null | undefined): string {
  switch (status) {
    case 'validated':
      return 'Validated';
    case 'below_threshold':
      return 'Below threshold';
    case 'unvalidated':
      return 'Unvalidated';
    default:
      return 'Unknown';
  }
}

export interface ModelCensus {
  /** Rows in the listing (one per brand label). */
  labels: number;
  /** Distinct fit fingerprints among them. */
  fits: number;
  /** Every row's R² is scored on a synthetic self-generated target. */
  allSynthetic: boolean;
  /** Rows whose fidelity_score is NULL. */
  unvalidated: number;
  /** Brands whose rows share a fingerprint with at least one other row. */
  sharedBrands: string[];
}

export function summarizeModelCensus(models: readonly TwinModelSummary[]): ModelCensus | null {
  if (models.length === 0) return null;
  const byFingerprint = new Map<string, TwinModelSummary[]>();
  for (const m of models) {
    const group = byFingerprint.get(m.training_fingerprint) ?? [];
    group.push(m);
    byFingerprint.set(m.training_fingerprint, group);
  }
  const sharedBrands = [...byFingerprint.values()]
    .filter((group) => group.length > 1)
    .flatMap((group) => group.map((m) => m.brand));
  return {
    labels: models.length,
    fits: byFingerprint.size,
    allSynthetic: models.every((m) => m.r2_score_basis === 'synthetic_target'),
    unvalidated: models.filter((m) => m.fidelity_status === 'unvalidated').length,
    sharedBrands,
  };
}

/** Short card subtext, e.g. "3 brand labels over 1 shared synthetic fit · unvalidated". */
export function describeModelCensus(models: readonly TwinModelSummary[]): string | null {
  const c = summarizeModelCensus(models);
  if (!c) return null;
  const labels = `${c.labels} brand label${c.labels === 1 ? '' : 's'}`;
  const shared = c.fits < c.labels ? 'shared ' : '';
  const synthetic = c.allSynthetic ? 'synthetic ' : '';
  const fits = `${c.fits} ${shared}${synthetic}fit${c.fits === 1 ? '' : 's'}`;
  const fidelity =
    c.unvalidated === c.labels
      ? 'unvalidated'
      : c.unvalidated === 0
        ? 'validated'
        : `${c.labels - c.unvalidated}/${c.labels} validated`;
  return `${labels} over ${fits} · ${fidelity}`;
}

/** Full explanation for the card tooltip. */
export function explainModelCensus(models: readonly TwinModelSummary[]): string | null {
  const c = summarizeModelCensus(models);
  if (!c) return null;
  const parts: string[] = [];
  if (c.fits < c.labels) {
    parts.push(
      `Brand is routing metadata: ${c.sharedBrands.join(', ')} share one identical fit ` +
        '(same training config, features and metrics); brand is not a model feature.'
    );
  }
  if (c.allSynthetic) {
    parts.push(
      'R² is scored against the synthetic training frame’s self-generated target, ' +
        'not real-world outcomes.'
    );
  }
  if (c.unvalidated > 0) {
    parts.push(
      c.unvalidated === c.labels
        ? 'Fidelity is unvalidated: no experiment outcome has been compared against these models yet.'
        : `${c.unvalidated} of ${c.labels} models have no experiment outcome compared against them yet.`
    );
  }
  return parts.length > 0 ? parts.join(' ') : null;
}
