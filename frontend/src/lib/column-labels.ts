/**
 * Display label for a gold-standard column (treatment / outcome).
 *
 * The backend owns the label SSOT (`causal._COLUMN_LABELS`) and serves it as a
 * `labels` map on `GET /causal/variables` and `GET /segments/datasets`. Every
 * page that prints a column name goes through this helper so the same column
 * reads the same everywhere — 2026-09-05: /segment-analysis rendered
 * "Product samples provided (rep sample drop)" (#1893) while /causal-analysis
 * still printed `sample_dropped`, because it never consumed the served map.
 *
 * Fallback (map absent or column not curated) mirrors the backend auto-label
 * byte-for-byte — underscores to spaces, then Python `str.capitalize()`: first
 * character upper-cased, the REST lower-cased — so an uncurated column never
 * reads differently on two pages, nor in the leaderboard cell beside the
 * backend's own summary prose.
 */
export function columnLabel(
  labels: Record<string, string> | null | undefined,
  col: string
): string {
  // A response missing the column name (partial payload) renders nothing, as
  // the raw interpolation did — a label helper must never crash a page.
  if (!col) return '';
  const curated = labels?.[col];
  if (curated) return curated;
  const spaced = col.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1).toLowerCase();
}
