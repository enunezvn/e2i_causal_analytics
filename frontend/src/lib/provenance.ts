/**
 * Provenance trust rule
 * =====================
 *
 * The health-score backend tags every response wrapper with `data_provenance`
 * (measured | partial | unknown | placeholder) and fails CLOSED: the field
 * defaults to an untrusted value so a path that forgets to tag real data
 * degrades to an honest empty state instead of presenting sample data as live.
 *
 * This is the single frontend definition of that trust decision — every
 * consumer of a health-score payload must read through it so no card can
 * accidentally surface a placeholder payload as real (codex PR-4 rounds 2–4).
 *
 * @module lib/provenance
 */

/**
 * A response's items are surfaced as real data only when the backend tagged
 * the wrapper with a trusted provenance. "measured" = live probe; "partial" =
 * some sub-fields unmeasured but the measured signals are real. "placeholder" /
 * "unknown" / absent are NOT trustworthy.
 */
export function isTrustedProvenance(provenance: string | undefined): boolean {
  return provenance === 'measured' || provenance === 'partial';
}
