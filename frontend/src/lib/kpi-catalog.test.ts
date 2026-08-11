/**
 * kpi-catalog tests — 2026-07-30
 * ==============================
 *
 * Why this exists: the chat's chart action can only plot a KPI it can first
 * resolve, and the hand-written alias table covered 6 of the registry's 44.
 * Worse, the registry-code regex matched `ws\d+-…` and `br-…` only, so the
 * whole CM-* causal-metric family fell through un-normalized — 'cm-001' went to
 * the API lowercased and missed. These tests hold the generated catalog to the
 * YAML registry and assert every KPI in it resolves.
 *
 * The drift guard matters because the catalog is generated: an edit to
 * config/kpi_definitions.yaml without re-running scripts/gen_kpi_catalog.py
 * would otherwise leave the chat silently unable to chart the new KPI.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import { KPI_CATALOG, REGION_ALIAS_MAP, REGION_LABELS } from './kpi-catalog.generated';
import { normalizeAlias, resolveKpiId } from './kpi-alias';

/** Registry ids scraped straight from the YAML, independent of the generator. */
function yamlKpiIds(): string[] {
  const yaml = readFileSync(
    resolve(__dirname, '../../../config/kpi_definitions.yaml'),
    'utf8'
  );
  // `helper_views` entries carry no id, so an id-line scrape is exactly the set
  // of real KPIs without needing a YAML parser in the frontend test env.
  const ids = [...yaml.matchAll(/^\s+id:\s*"([^"]+)"/gm)].map((m) => m[1]);
  return [...new Set(ids)];
}

describe('KPI catalog', () => {
  it('covers every KPI id in config/kpi_definitions.yaml', () => {
    const fromYaml = yamlKpiIds().sort();
    const fromCatalog = KPI_CATALOG.map((e) => e.id).sort();
    // Fails when the YAML gains a KPI and the catalog was not regenerated.
    expect(fromCatalog).toEqual(fromYaml);
  });

  it('covers all six workstreams', () => {
    expect(new Set(KPI_CATALOG.map((e) => e.workstream))).toEqual(
      new Set([
        'ws1_data_quality',
        'ws1_model_performance',
        'ws2_triggers',
        'ws3_business',
        'brand_specific',
        'causal_metrics',
      ])
    );
  });

  it('maps no alias to two different KPIs', () => {
    const owner = new Map<string, string>();
    for (const entry of KPI_CATALOG) {
      for (const alias of entry.aliases) {
        const existing = owner.get(alias);
        expect(existing === undefined || existing === entry.id).toBe(true);
        owner.set(alias, entry.id);
      }
    }
  });

  it('stores aliases already in normalized form', () => {
    // The generator and normalizeAlias must agree, or catalog keys can never
    // be hit by a lookup. This is the parity assertion both sides point at.
    for (const entry of KPI_CATALOG) {
      for (const alias of entry.aliases) {
        expect(normalizeAlias(alias)).toBe(alias);
      }
    }
  });
});

describe('resolveKpiId covers the whole registry', () => {
  it('resolves every KPI by its registry code, in any case', () => {
    for (const entry of KPI_CATALOG) {
      expect(resolveKpiId(entry.id)).toBe(entry.id);
      expect(resolveKpiId(entry.id.toLowerCase())).toBe(entry.id);
    }
  });

  it('resolves every KPI by its yaml key', () => {
    for (const entry of KPI_CATALOG) {
      expect(resolveKpiId(entry.key)).toBe(entry.id);
    }
  });

  it('resolves every KPI by its display name', () => {
    for (const entry of KPI_CATALOG) {
      expect(resolveKpiId(entry.name)).toBe(entry.id);
    }
  });

  it('resolves the causal-metric family the old regex missed', () => {
    // The specific regression: bare-prefix codes outside the br- family.
    expect(resolveKpiId('cm-001')).toBe('CM-001');
    expect(resolveKpiId('CM-003')).toBe('CM-003');
    expect(resolveKpiId('Average Treatment Effect (ATE)')).toBe('CM-001');
    expect(resolveKpiId('ate')).toBe('CM-001');
  });

  it('resolves names people actually type', () => {
    expect(resolveKpiId('ROC-AUC')).toBe('WS1-MP-001');
    expect(resolveKpiId('roc auc')).toBe('WS1-MP-001');
    expect(resolveKpiId('trigger precision')).toBe('WS2-TR-001');
    expect(resolveKpiId('Cross-source Match Rate')).toBe('WS1-DQ-003');
    expect(resolveKpiId('MAU')).toBe('WS3-BI-001');
  });

  it('still passes unknown ids through untouched', () => {
    // Guessing a KPI is worse than an honest empty chart.
    expect(resolveKpiId('bogus_metric')).toBe('bogus_metric');
  });
});

describe('semantic types', () => {
  it('never types a model-quality score as a percentage', () => {
    // An ROC-AUC axis labelled "85%" is wrong: it is a unitless score.
    for (const id of ['WS1-MP-001', 'WS1-MP-002', 'WS1-MP-003', 'WS1-MP-005']) {
      expect(KPI_CATALOG.find((e) => e.id === id)?.semanticType).toBe('Score');
    }
  });

  it('types Rx volumes and user counts as counts', () => {
    for (const id of ['WS3-BI-005', 'WS3-BI-006', 'WS3-BI-007', 'WS3-BI-001']) {
      expect(KPI_CATALOG.find((e) => e.id === id)?.semanticType).toBe('Count');
    }
  });

  it('types causal effect sizes as signed numbers', () => {
    // A zero-based percentage axis would clip a negative effect out of view,
    // and whether the interval crosses zero is the reason to draw the chart.
    for (const id of ['CM-001', 'CM-002', 'CM-003', 'CM-004', 'CM-005']) {
      expect(KPI_CATALOG.find((e) => e.id === id)?.semanticType).toBe('Number');
    }
  });
});

describe('region vocabulary (#1538)', () => {
  // The map is GENERATED from src/services/enum_labels.py (REGION_ALIASES) —
  // the platform's one region synonym table — so the chat chart tools speak
  // the same vocabulary as the backend chat tool and cohort resolution.
  it('exports the four region_type enum labels', () => {
    expect([...REGION_LABELS].sort()).toEqual(['midwest', 'northeast', 'south', 'west']);
  });

  it('maps every label to itself and every alias to a real label', () => {
    for (const label of REGION_LABELS) {
      expect(REGION_ALIAS_MAP[label]).toBe(label);
    }
    for (const target of Object.values(REGION_ALIAS_MAP)) {
      expect(REGION_LABELS).toContain(target);
    }
  });

  it('keys the map in folded form (casefolded, separators removed)', () => {
    // Mirrors enum_labels.fold_region_key: the labels are single concatenated
    // words, so lookups fold "North East"/"mid-west" down to them.
    for (const key of Object.keys(REGION_ALIAS_MAP)) {
      expect(key).toBe(key.toLowerCase());
      expect(key).not.toMatch(/[\s_-]/);
    }
    expect(REGION_ALIAS_MAP['northeast']).toBe('northeast');
    expect(REGION_ALIAS_MAP['newengland']).toBe('northeast');
    expect(REGION_ALIAS_MAP['pacific']).toBe('west');
  });
});
