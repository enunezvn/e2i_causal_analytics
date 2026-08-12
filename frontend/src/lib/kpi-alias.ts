/**
 * KPI id / brand aliasing for chat generative UI
 * ==============================================
 *
 * The KPI substrate (`kpi_history`, served by /api/kpis/{id}/history) keys on
 * registry codes (WS3-BI-005, BR-001, …), while the chat model naturally
 * passes the friendly ids the `renderKpiTrend` action description teaches
 * (trx, nrx, …). This module is the translation hop between the two — it
 * exists so the chat action can speak friendly ids without the /api/kpis/*
 * routes growing an alias layer (every other consumer already passes codes).
 *
 * Per-brand coverage (as seeded): TRx/NRx/ROI have global ('' brand) and
 * per-brand series; NBRx and TRx Share are per-brand ONLY, so those charts
 * need a brand or they come back honest-empty.
 *
 * The alias table is derived from the generated KPI catalog so that ALL 44
 * registry KPIs resolve, not just the Rx-volume handful the chat action
 * originally advertised.
 */

import { KPI_CATALOG, REGION_ALIAS_MAP } from './kpi-catalog.generated';

/**
 * Colloquial aliases the registry itself cannot yield — spoken names with no
 * counterpart in a KPI's id, yaml key, or display name. Everything derivable
 * from the registry comes from KPI_CATALOG instead (see kpi-catalog.generated),
 * so this map stays small and each entry is a deliberate vocabulary decision.
 */
const MANUAL_ALIASES: Record<string, string> = {
  market_share: 'WS3-BI-008', // "TRx Share" is never said out loud
  scripts: 'WS3-BI-005',
  prescriptions: 'WS3-BI-005',
  ate: 'CM-001',
  cate: 'CM-002',
  mau: 'WS3-BI-001',
  wau: 'WS3-BI-002',
  psi: 'WS1-MP-009',
  auc: 'WS1-MP-001',
};

/**
 * Every alias the registry yields (id, yaml key, display name, parenthesised
 * abbreviation) folded into one lookup, with the colloquial map layered on top.
 * Built once at module load from the generated catalog — all 44 KPIs, not the
 * 6 the hand-written table used to cover.
 */
const KPI_ALIASES: Record<string, string> = (() => {
  const table: Record<string, string> = {};
  for (const entry of KPI_CATALOG) {
    for (const alias of entry.aliases) table[alias] = entry.id;
  }
  return { ...table, ...MANUAL_ALIASES };
})();

/**
 * Registry codes pass through as-is. Covers every family in the registry:
 * workstream-scoped (WS3-BI-005, WS1-MP-009, WS2-TR-004) and bare-prefix
 * (BR-001, CM-003). The previous pattern hardcoded `br-` as the only
 * bare-prefix family, so the whole CM-* causal-metric family failed to
 * normalize — 'cm-001' reached the API lowercased and missed.
 */
const REGISTRY_CODE = /^([a-z]+\d+-[a-z]+-\d+|[a-z]+-\d+)$/i;

/** Brand values are stored canonical-cased and matched exactly by the API. */
const BRAND_ALIASES: Record<string, string> = {
  remibrutinib: 'Remibrutinib',
  remi: 'Remibrutinib',
  fabhalta: 'Fabhalta',
  kisqali: 'Kisqali',
};

/**
 * Resolve a model-supplied KPI identifier to the registry code the substrate
 * speaks. Unknown ids pass through unchanged — the API then returns an
 * honest-empty series rather than this hop guessing.
 */
export function resolveKpiId(kpiId: string): string {
  const trimmed = kpiId.trim();
  if (REGISTRY_CODE.test(trimmed)) return trimmed.toUpperCase();
  return KPI_ALIASES[normalizeAlias(trimmed)] ?? trimmed;
}

/**
 * Normalize a free-text KPI reference to the alias key form: lowercase, with
 * runs of space / hyphen / underscore / slash collapsed to a single underscore.
 *
 * MUST stay in lockstep with `alias_forms` in scripts/gen_kpi_catalog.py —
 * the generator normalizes the catalog's alias keys with the same rule, so a
 * divergence here silently stops those keys from ever matching.
 * `kpi-catalog.test.ts` asserts the parity.
 */
export function normalizeAlias(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s\-_/]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/**
 * Canonicalize a model-supplied brand name to the exact casing stored in
 * kpi_history. Blank/missing brand → undefined (global series).
 */
export function resolveBrand(brand: string | undefined): string | undefined {
  const trimmed = brand?.trim();
  if (!trimmed) return undefined;
  return BRAND_ALIASES[trimmed.toLowerCase()] ?? trimmed;
}

/**
 * Canonicalize a model-supplied region ('Northeast', 'North East', 'NE',
 * 'Pacific', …) to its region_type enum label via the platform synonym table
 * (#1538 — REGION_ALIAS_MAP is generated from src/services/enum_labels.py, the
 * SSOT every backend surface shares). Three outcomes, three shapes:
 *
 * - blank / missing        -> `undefined` (no region scope)
 * - label or known synonym -> the enum label ('northeast' | 'south' | …)
 * - anything else          -> `null` — the caller must REFUSE with the known
 *   labels rather than pass it through: region can never match a row, so a
 *   passthrough would put a 0-value figure under a junk region caption (the
 *   backend chat tool fails fast on the same input for the same reason).
 */
export function resolveRegion(region: string | undefined): string | null | undefined {
  const trimmed = region?.trim();
  if (!trimmed) return undefined;
  // Mirror enum_labels.fold_region_key: casefold + remove separators — the
  // labels are single concatenated words, so 'North East' folds to 'northeast'.
  const folded = trimmed.toLowerCase().replace(/[\s_-]+/g, '');
  const exact = REGION_ALIAS_MAP[folded];
  if (exact !== undefined) return exact;
  // #1565: mirror resolve_region_label's LOOKUP-time noise strip — a leading
  // "the" and trailing "region"/"area" tokens never change WHICH region a
  // phrase names ('the Northeast region' -> northeast). Tried only after the
  // exact fold misses, so no existing resolution changes. Both patterns
  // require a separator boundary: bare noise words ('the', 'region') strip
  // to nothing and stay null. 'coast' is deliberately NOT stripped —
  // 'central coast' (California) must never resolve to 'central' -> midwest.
  const lower = trimmed.toLowerCase();
  const stripped = lower
    .replace(/^the[\s_-]+/, '')
    .replace(/(?:[\s_-]+(?:region|area))+$/, '');
  if (stripped !== lower) {
    return REGION_ALIAS_MAP[stripped.replace(/[\s_-]+/g, '')] ?? null;
  }
  return null;
}

/** Severity aliases → canonical patient_journeys.segment_assignment values. */
const SEGMENT_ALIASES: Record<string, string> = {
  low: 'low_severity',
  low_severity: 'low_severity',
  mild: 'low_severity',
  medium: 'medium_severity',
  medium_severity: 'medium_severity',
  moderate: 'medium_severity',
  high: 'high_severity',
  high_severity: 'high_severity',
  severe: 'high_severity',
};

/**
 * Canonicalize a model-supplied severity tier ('medium', 'High severity', …)
 * to the segment_assignment value the API speaks. Unknown values pass
 * through — the API then 422s honestly rather than this hop guessing.
 */
export function resolveSegment(segment: string | undefined): string | undefined {
  const trimmed = segment?.trim();
  if (!trimmed) return undefined;
  const normalized = trimmed.toLowerCase().replace(/[\s-]+/g, '_');
  return SEGMENT_ALIASES[normalized] ?? trimmed;
}

/**
 * Canonicalize a model-supplied line-of-therapy ('2', 'LOT 2', 'line 2') to
 * the '0'-'3' bucket keys the API speaks (prior_therapy_lines).
 */
export function resolveTherapyLine(line: string | number | undefined): string | undefined {
  if (line === undefined || line === null) return undefined;
  const trimmed = String(line).trim();
  if (!trimmed) return undefined;
  const match = trimmed.match(/(\d+)/);
  return match ? match[1] : trimmed;
}

/**
 * Resolve a model-supplied comparison axis ('severity', 'LOT', …) to the
 * canonical axis the segmented-history API speaks. Unknown → undefined.
 */
export function resolveCompareAxis(
  compareBy: string | undefined
): 'segment' | 'therapy_line' | undefined {
  const normalized = compareBy?.trim().toLowerCase().replace(/[\s-]+/g, '_');
  if (!normalized) return undefined;
  if (
    ['segment', 'segments', 'severity', 'severity_tier', 'severity_segment', 'patient_segment'].includes(
      normalized
    )
  ) {
    return 'segment';
  }
  if (
    ['therapy_line', 'therapy_lines', 'lot', 'line', 'lines', 'line_of_therapy'].includes(normalized)
  ) {
    return 'therapy_line';
  }
  return undefined;
}
