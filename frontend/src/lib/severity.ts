/**
 * Severity Mapper
 * ===============
 *
 * Single source of truth for collapsing the backend's several severity
 * vocabularies into ONE three-level UI vocabulary (`critical` / `warning` /
 * `info`), plus a display label and a `Badge` variant token.
 *
 * Why this exists (#26):
 *   The backend emits severity under at least three distinct enums, verified
 *   by grepping the Python API and the generated OpenAPI types:
 *     - AlertSeverity   (src/api/routes/experiments.py): "critical"|"warning"|"info"
 *     - DriftSeverity   (src/api/routes/monitoring.py):   "none"|"low"|"medium"|"high"|"critical"
 *     - PatternSeverity (src/api/routes/feedback.py):     "low"|"medium"|"high"|"critical"
 *   `AlertItem.severity` is typed as a bare `string` on the wire, so a drift
 *   "high" can flow into a component that only knows critical/warning/info and
 *   silently degrade to "info". Centralizing the mapping removes that latent
 *   under-mapping and the ad-hoc copies scattered across consumers.
 *
 * Mapping rule (backend -> UI):
 *   critical            -> critical
 *   high                -> warning   (a "high" alert is NOT informational)
 *   warning | medium    -> warning
 *   info | low | none   -> info
 *   unknown / empty     -> info      (safe default; this function never throws)
 *
 * Note: this maps the *alert/drift severity* vocabulary. It is intentionally
 * NOT applied to presentation primitives that own broader status vocabularies
 * (e.g. StatusBadge's `healthy|loading|paused|...`, KPICard's `neutral`), which
 * are not backend-severity mappings and would be wrong to force through here.
 */

/** The three-level UI severity vocabulary. */
export type UiSeverity = 'critical' | 'warning' | 'info';

/** A `Badge` component variant token (see src/components/ui/badge.tsx). */
export type SeverityVariant = 'destructive' | 'warning' | 'secondary';

/** Full descriptor for rendering a severity consistently across the app. */
export interface SeverityDescriptor {
  /** Collapsed UI severity. */
  severity: UiSeverity;
  /** Human-readable label. */
  label: string;
  /** Badge variant token to pass straight to `<Badge variant={...} />`. */
  variant: SeverityVariant;
}

const UI_SEVERITY_DESCRIPTOR: Record<UiSeverity, SeverityDescriptor> = {
  critical: { severity: 'critical', label: 'Critical', variant: 'destructive' },
  warning: { severity: 'warning', label: 'Warning', variant: 'warning' },
  info: { severity: 'info', label: 'Info', variant: 'secondary' },
};

/**
 * Backend severity string -> UI severity. Covers the union of every backend
 * severity vocabulary. Keys are the exact lowercase enum values the API emits.
 */
const BACKEND_TO_UI: Record<string, UiSeverity> = {
  // AlertSeverity (already UI vocabulary)
  critical: 'critical',
  warning: 'warning',
  info: 'info',
  // DriftSeverity / PatternSeverity
  high: 'warning',
  medium: 'warning',
  low: 'info',
  none: 'info',
};

/**
 * Map any backend severity string to the three-level UI severity.
 *
 * Case-insensitive (backend enums sometimes arrive upper-cased). Unknown,
 * empty, null, or undefined values fall back to `'info'` — the safe default —
 * and never throw.
 */
export function toUiSeverity(
  backendSeverity: string | null | undefined,
): UiSeverity {
  if (!backendSeverity) {
    return 'info';
  }
  return BACKEND_TO_UI[backendSeverity.trim().toLowerCase()] ?? 'info';
}

/**
 * Map any backend severity string to a full {@link SeverityDescriptor}
 * (UI severity + label + Badge variant). Same fallback semantics as
 * {@link toUiSeverity}.
 */
export function mapSeverity(
  backendSeverity: string | null | undefined,
): SeverityDescriptor {
  return UI_SEVERITY_DESCRIPTOR[toUiSeverity(backendSeverity)];
}

export default mapSeverity;
