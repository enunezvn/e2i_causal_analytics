/**
 * Digital Twin model honesty helpers (#2206)
 * ==========================================
 *
 * The backend's `/digital-twin/models` rows carry data-derived honesty fields:
 * a fit fingerprint (equal across rows = one shared fit under several brand
 * labels), `shared_fit_model_count` (a census over ALL brands, so a brand-scoped
 * viewer still learns their fit is shared), `brand_is_feature`, what the R² was
 * scored against, and an explicit fidelity status where a NULL score is
 * `unvalidated` — never a pass. These helpers turn that census into the one-line
 * summary and tooltip the page renders. Every claim reads a field; nothing is
 * inferred from brand names or from counting the visible rows alone.
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
  /** Rows the caller can see (one per brand label in their grant). */
  visible: number;
  /** Brand labels that exist over these fits, INCLUDING ones hidden by the caller's
   *  brand grant — the backend's `shared_fit_model_count` is a census over all brands. */
  labels: number;
  /** Distinct fit fingerprints among the visible rows. */
  fits: number;
  /** Every visible row's R² is scored on a synthetic self-generated target. */
  allSynthetic: boolean;
  /** Visible rows whose fidelity_score is NULL. */
  unvalidated: number;
  /** Visible rows measured at or above the engine's gate. */
  validated: number;
  /** Visible rows measured below the engine's gate — a failed gate, never "validated". */
  belowThreshold: number;
  /** Visible brands whose fit is shared with at least one other brand model. */
  sharedBrands: string[];
  /** Brand labels sharing a fit that the caller cannot see (scoped grant). */
  hiddenSharedLabels: number;
  /** Whether any visible model uses brand as a feature (false: routing metadata). */
  brandIsFeature: boolean;
}

/**
 * Derive the census from the backend's data-derived fields, not from the visible
 * rows alone: a brand-scoped viewer sees ONE row, and its `shared_fit_model_count`
 * (3) and `brand_is_feature` (false) are what say the fit is shared (codex r1 #3).
 */
export function summarizeModelCensus(models: readonly TwinModelSummary[]): ModelCensus | null {
  if (models.length === 0) return null;
  const byFingerprint = new Map<string, TwinModelSummary[]>();
  for (const m of models) {
    const group = byFingerprint.get(m.training_fingerprint) ?? [];
    group.push(m);
    byFingerprint.set(m.training_fingerprint, group);
  }
  let labels = 0;
  let hiddenSharedLabels = 0;
  const sharedBrands: string[] = [];
  for (const group of byFingerprint.values()) {
    // The group's true size is the backend census (all brands), never fewer than shown.
    const size = Math.max(group.length, ...group.map((m) => m.shared_fit_model_count));
    labels += size;
    hiddenSharedLabels += size - group.length;
    if (size > 1) sharedBrands.push(...group.map((m) => m.brand));
  }
  return {
    visible: models.length,
    labels,
    fits: byFingerprint.size,
    allSynthetic: models.every((m) => m.r2_score_basis === 'synthetic_target'),
    unvalidated: models.filter((m) => m.fidelity_status === 'unvalidated').length,
    validated: models.filter((m) => m.fidelity_status === 'validated').length,
    belowThreshold: models.filter((m) => m.fidelity_status === 'below_threshold').length,
    sharedBrands,
    hiddenSharedLabels,
    brandIsFeature: models.some((m) => m.brand_is_feature),
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
  // The fidelity clause speaks only for the rows the caller can see, and names
  // all three states: a below-threshold model is a failed gate, never "validated".
  const who = c.visible < c.labels && c.visible === 1 ? `${models[0].brand} ` : '';
  let fidelity: string;
  if (c.unvalidated === c.visible) fidelity = `${who}unvalidated`;
  else if (c.validated === c.visible) fidelity = `${who}validated`;
  else if (c.belowThreshold === c.visible) fidelity = `${who}below threshold`;
  else
    fidelity = [
      c.validated > 0 ? `${c.validated} validated` : null,
      c.belowThreshold > 0 ? `${c.belowThreshold} below threshold` : null,
      c.unvalidated > 0 ? `${c.unvalidated} unvalidated` : null,
    ]
      .filter((part): part is string => part !== null)
      .join(' · ');
  return `${labels} over ${fits} · ${fidelity}`;
}

/** Full explanation for the card tooltip. */
export function explainModelCensus(models: readonly TwinModelSummary[]): string | null {
  const c = summarizeModelCensus(models);
  if (!c) return null;
  const parts: string[] = [];
  if (c.fits < c.labels) {
    const hidden =
      c.hiddenSharedLabels > 0
        ? ` (${c.hiddenSharedLabels} other brand model${c.hiddenSharedLabels === 1 ? '' : 's'} outside your brand grant)`
        : '';
    // "Routing metadata" is a conclusion the backend's brand_is_feature supports
    // only when brand is NOT a feature; otherwise state the shared fit alone.
    const lead = c.brandIsFeature ? '' : 'Brand is routing metadata: ';
    const feature = c.brandIsFeature ? 'brand is a model feature.' : 'brand is not a model feature.';
    parts.push(
      `${lead}${c.sharedBrands.join(', ')}${hidden} share one identical fit ` +
        `(same training config, features and metrics); ${feature}`
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
      c.unvalidated === c.visible
        ? 'Fidelity is unvalidated: no experiment outcome has been compared against these models yet.'
        : `${c.unvalidated} of ${c.visible} models have no experiment outcome compared against them yet.`
    );
  }
  if (c.belowThreshold > 0) {
    parts.push(
      `${c.belowThreshold} of ${c.visible} model${c.visible === 1 ? ' is' : 's are'} below the fidelity threshold (measured prediction accuracy failed the 0.70 gate).`
    );
  }
  return parts.length > 0 ? parts.join(' ') : null;
}

/** True when every trained model the caller can see has never been validated. */
export function allModelsUnvalidated(models: readonly TwinModelSummary[]): boolean {
  return models.length > 0 && models.every((m) => m.fidelity_status === 'unvalidated');
}
