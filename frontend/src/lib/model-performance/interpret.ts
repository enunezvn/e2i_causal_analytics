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
