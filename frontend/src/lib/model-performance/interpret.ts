/**
 * Model-performance interpretation helpers.
 *
 * Pure functions that turn the confusion-matrix / ROC API responses plus the
 * model name into human-readable, model-specific interpretation. No network,
 * no fabrication: a metric with a zero denominator reads "n/a" rather than a
 * fake 0%/100%.
 *
 * @module lib/model-performance/interpret
 */

import type { ConfusionMatrixResponse } from '@/types/monitoring';

export interface ModelMeaning {
  /** Singular subject noun, e.g. "patient" | "HCP". */
  subject: string;
  /** Plural subject noun, e.g. "patients" | "HCPs". */
  subjectPlural: string;
  /** Positive-class verb phrase, e.g. "initiated treatment". */
  positiveEvent: string;
  /** False when the cohort could not be identified (generic fallback used). */
  known: boolean;
}

/**
 * Derive the real-world meaning of the positive class from a gold-standard
 * model name. Case-insensitive substring match; hcp_adoption is checked before
 * the patient cohorts. Unknown names get a generic, never-throwing fallback.
 */
export function describeModel(modelName: string): ModelMeaning {
  const n = (modelName || '').toLowerCase();
  if (n.includes('hcp_adoption')) {
    return { subject: 'HCP', subjectPlural: 'HCPs', positiveEvent: 'adopted the brand', known: true };
  }
  if (n.includes('initiation')) {
    return { subject: 'patient', subjectPlural: 'patients', positiveEvent: 'initiated treatment', known: true };
  }
  if (n.includes('persistence')) {
    return { subject: 'patient', subjectPlural: 'patients', positiveEvent: 'persisted ≥180 days', known: true };
  }
  if (n.includes('discontinuation')) {
    return {
      subject: 'patient',
      subjectPlural: 'patients',
      positiveEvent: 'discontinued within 180 days',
      known: true,
    };
  }
  return { subject: 'case', subjectPlural: 'cases', positiveEvent: 'were in the positive class', known: false };
}

export interface Metric {
  /** Fraction in [0,1], or null when undefined (zero denominator). */
  value: number | null;
  /** Display string: "59%" or "n/a". */
  pct: string;
}

function metric(numerator: number, denominator: number): Metric {
  if (denominator <= 0) return { value: null, pct: 'n/a' };
  const v = numerator / denominator;
  return { value: v, pct: `${Math.round(v * 100)}%` };
}

export interface ConfusionInterpretation {
  precision: Metric;
  recall: Metric;
  specificity: Metric;
  accuracy: Metric;
  f1: Metric;
  verdict: string;
}

function positivesPhrase(meaning: ModelMeaning): string {
  return meaning.known
    ? `${meaning.subjectPlural} who actually ${meaning.positiveEvent}`
    : 'positive cases';
}

function buildVerdict(
  tp: number,
  fn: number,
  precision: Metric,
  recall: Metric,
  specificity: Metric,
  meaning: ModelMeaning
): string {
  const R = recall.value;
  const P = precision.value;
  const S = specificity.value;

  if (R === null && P === null) {
    return 'Not enough holdout outcomes to characterize this model’s behavior.';
  }

  const actualPos = tp + fn;
  const caught =
    `catches ${tp.toLocaleString('en-US')} of ${actualPos.toLocaleString('en-US')} ` +
    `${positivesPhrase(meaning)} (recall ${recall.pct})`;
  const right = P !== null ? `, and is right ${precision.pct} of the time when it predicts so (precision)` : '';
  const spec = S !== null ? `, with specificity ${specificity.pct}` : '';

  // Only reached when at least one of R/P is non-null. If exactly one is null
  // (e.g. tp+fn==0 but fp>0), the sentence still renders with an "(... n/a)"
  // clause — acceptable: a trained model on real holdout has tp>=1.
  let archetype: string;
  if (R !== null && S !== null && R < 0.5 && S >= 0.7) {
    archetype = ' — conservative: it under-calls and misses most true cases.';
  } else if (R !== null && P !== null && R >= 0.7 && P < 0.5) {
    archetype = ' — aggressive: it over-calls, trading false alarms for coverage.';
  } else if (R !== null && P !== null && R >= 0.6 && P >= 0.6) {
    archetype = ' — a balanced classifier.';
  } else {
    archetype = ' — limited discrimination at this threshold; read predictions with caution.';
  }

  return `This model ${caught}${right}${spec}${archetype}`;
}

/**
 * Derive precision/recall/specificity/accuracy/F1 (each "n/a" when its
 * denominator is zero) plus a rule-based, domain-framed verdict from a binary
 * confusion matrix.
 */
export function interpretConfusion(
  data: ConfusionMatrixResponse,
  meaning: ModelMeaning
): ConfusionInterpretation {
  const { tn, fp, fn, tp } = data;
  const n = tn + fp + fn + tp;
  const precision = metric(tp, tp + fp);
  const recall = metric(tp, tp + fn);
  const specificity = metric(tn, tn + fp);
  const accuracy = metric(tp + tn, n);

  let f1: Metric = { value: null, pct: 'n/a' };
  if (precision.value !== null && recall.value !== null && precision.value + recall.value > 0) {
    const v = (2 * precision.value * recall.value) / (precision.value + recall.value);
    f1 = { value: v, pct: `${Math.round(v * 100)}%` };
  }

  const verdict = buildVerdict(tp, fn, precision, recall, specificity, meaning);
  return { precision, recall, specificity, accuracy, f1, verdict };
}

export interface RocInterpretation {
  /** Quality band label. */
  band: string;
  /** Full plain-language sentence. */
  text: string;
}

/**
 * Non-inflated AUC quality band (0.5 = chance). Assumes `auc` in [0.5, 1.0]
 * (holdout ROC AUC; the 12 gold-standard models are all > 0.5; sub-0.5
 * anti-predictive AUC is out of scope).
 */
export function aucBand(auc: number): string {
  if (auc < 0.6) return 'near-random';
  if (auc < 0.7) return 'weak';
  if (auc < 0.8) return 'acceptable';
  if (auc < 0.9) return 'good';
  return 'excellent';
}

/**
 * Interpret an ROC AUC: quality band + the ranking-probability sentence vs the
 * 0.5 chance baseline, framed in the model's cohort domain.
 */
export function interpretRoc(auc: number, meaning: ModelMeaning): RocInterpretation {
  const band = aucBand(auc);
  const pct = Math.round(auc * 100);
  const rank = meaning.known
    ? `a random ${meaning.subject} who ${meaning.positiveEvent} above a random one who did not`
    : 'a random positive case above a random negative case';

  let compare: string;
  if (auc < 0.6) compare = 'barely above the 0.50 coin-flip baseline';
  else if (auc < 0.7) compare = 'modestly better than the 0.50 coin-flip baseline';
  else if (auc <= 0.85) compare = 'clearly better than chance (0.50)';
  else compare = 'strong separation, well above chance (0.50)';

  const text = `AUC ${auc.toFixed(3)} (${band}). The model ranks ${rank} ${pct}% of the time — ${compare}.`;
  return { band, text };
}
