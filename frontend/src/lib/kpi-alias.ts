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
 */

/** Friendly id → KPI registry code. Keys are normalized: lowercase, [ -]→_ */
const KPI_ALIASES: Record<string, string> = {
  trx: 'WS3-BI-005',
  nrx: 'WS3-BI-006',
  nbrx: 'WS3-BI-007',
  trx_share: 'WS3-BI-008',
  market_share: 'WS3-BI-008',
  conversion_rate: 'WS3-BI-009',
  roi: 'WS3-BI-010',
};

/** Registry codes (WS3-BI-005, WS2-TR-004, BR-001, …) pass through as-is. */
const REGISTRY_CODE = /^(ws\d+-[a-z]+-\d+|br-\d+)$/i;

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
  const normalized = trimmed.toLowerCase().replace(/[\s-]+/g, '_');
  return KPI_ALIASES[normalized] ?? trimmed;
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
