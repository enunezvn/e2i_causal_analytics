/**
 * Friendly label for the backend `data_provenance` marker — honest about the effect
 * basis. Both known values are synthetic data (the SYNTHETIC badge stays on the page);
 * the cohort one is a brand/intervention-ESTIMATED effect, the other a flat uniform.
 * One mapping for every surface that names a twin estimate's source (the Digital Twin
 * page and the proposed-experiments feed, #2206): a proposal must never read as
 * real-world evidence.
 *
 * @module lib/digital-twin-provenance
 */
export function provenanceLabel(provenance: string | null | undefined): string {
  switch (provenance) {
    case 'synthetic_uplift_v1':
      return 'synthetic uplift model (v1 — uniform, not brand-specific)';
    case 'cohort_estimated_synthetic_gold_v1':
      return 'brand cohort–estimated (synthetic-gold; not real-world data)';
    case null:
    case undefined:
    case '':
      return 'provenance not recorded';
    default:
      return provenance;
  }
}
