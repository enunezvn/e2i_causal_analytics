/**
 * Causal API Client Tests
 * =======================
 *
 * Focused tests for the opt-in response validation wired into the causal
 * client (C31). Causal endpoints have no default MSW handlers, so each test
 * registers its own handler via `server.use`.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import {
  getCausalHealth,
  listEstimators,
  getCausalEstimationData,
  getCausalVariables,
  getClinicalContext,
} from './causal';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Causal API Client', () => {
  describe('response validation (C31)', () => {
    it('getCausalHealth passes a valid response through (schema wired)', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            libraries_available: { dowhy: true, econml: true, causalml: false },
            estimators_loaded: 12,
            pipeline_orchestrator_ready: true,
            hierarchical_analyzer_ready: true,
            last_analysis: new Date().toISOString(),
            analysis_count_24h: 7,
            average_latency_ms: 1234,
          })
        )
      );

      const result = await getCausalHealth();
      expect(result.status).toBe('healthy');
      expect(result.estimators_loaded).toBe(12);
    });

    it('getCausalHealth throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            // libraries_available must be Record<string, boolean>
            libraries_available: { dowhy: 'yes' },
            estimators_loaded: 1,
            pipeline_orchestrator_ready: true,
            hierarchical_analyzer_ready: true,
            analysis_count_24h: 0,
          })
        )
      );

      await expect(getCausalHealth()).rejects.toBeInstanceOf(ApiValidationError);
    });

    it('listEstimators passes a valid response through (schema wired)', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/estimators`, () =>
          HttpResponse.json({
            estimators: [
              {
                name: 'causal_forest',
                library: 'econml',
                estimator_type: 'CATE',
                description: 'Causal forest estimator',
                best_for: ['heterogeneous effects'],
                parameters: ['n_estimators'],
                supports_confidence_intervals: true,
                supports_heterogeneous_effects: true,
              },
            ],
            total: 1,
            by_library: { econml: ['causal_forest'] },
          })
        )
      );

      const result = await listEstimators();
      expect(result.total).toBe(1);
      expect(result.estimators[0]?.name).toBe('causal_forest');
    });

    it('listEstimators throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/estimators`, () =>
          HttpResponse.json({
            estimators: [{ name: 'broken' }], // missing required estimator fields
            total: 1,
            by_library: {},
          })
        )
      );

      await expect(listEstimators()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });

  // Regression: the page's "Run parallel pipeline" prefetch 422'd live because
  // these two GETs passed `{ params: {...} }` to get(), which takes a FLAT
  // params object — so axios serialized `params[treatment_var]=...` and the
  // backend's REQUIRED treatment_var/outcome_var arrived missing (422).
  describe('query param serialization (flat, not params[...] wrapped)', () => {
    it('getCausalEstimationData sends flat treatment_var/outcome_var/covariates', async () => {
      let captured: URLSearchParams | null = null;
      server.use(
        http.get(`${env.apiUrl}/causal/estimation-data`, ({ request }) => {
          captured = new URL(request.url).searchParams;
          return HttpResponse.json({
            dataset: 'patient_journeys',
            columns: ['treatment_arm', 'persistent_180d'],
            n_rows: 1,
            estimation_data_records: [{ treatment_arm: 1, persistent_180d: 0 }],
          });
        })
      );

      await getCausalEstimationData({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
        covariates: ['disease_severity', 'engagement_score'],
        limit: 4000,
      });

      const params = captured as unknown as URLSearchParams;
      expect(params.get('treatment_var')).toBe('treatment_arm');
      expect(params.get('outcome_var')).toBe('persistent_180d');
      expect(params.get('covariates')).toBe('disease_severity,engagement_score');
      expect(params.get('dataset')).toBe('patient_journeys');
      expect(params.get('limit')).toBe('4000');
      // The double-wrap bug would have produced these instead:
      expect(params.get('params[treatment_var]')).toBeNull();
      expect(params.has('params[outcome_var]')).toBe(false);
    });

    it('getCausalVariables sends a flat dataset param', async () => {
      let captured: URLSearchParams | null = null;
      server.use(
        http.get(`${env.apiUrl}/causal/variables`, ({ request }) => {
          captured = new URL(request.url).searchParams;
          return HttpResponse.json({
            dataset: 'patient_journeys',
            treatment_candidates: ['treatment_arm'],
            outcome_candidates: ['persistent_180d'],
            covariate_candidates: ['disease_severity'],
            columns: ['treatment_arm', 'persistent_180d', 'disease_severity'],
          });
        })
      );

      await getCausalVariables('patient_journeys');

      const params = captured as unknown as URLSearchParams;
      expect(params.get('dataset')).toBe('patient_journeys');
      expect(params.get('params[dataset]')).toBeNull();
    });

    it('getCausalVariables sends the brand as a flat param and omits it when null', async () => {
      const seen: Array<string | null> = [];
      server.use(
        http.get(`${env.apiUrl}/causal/variables`, ({ request }) => {
          seen.push(new URL(request.url).searchParams.get('brand'));
          return HttpResponse.json({
            dataset: 'patient_journeys',
            treatment_candidates: [],
            outcome_candidates: [],
            covariate_candidates: [],
            columns: [],
            clinical_biomarkers: [],
          });
        })
      );

      await getCausalVariables('patient_journeys', 'Fabhalta');
      await getCausalVariables('patient_journeys', null);
      await getCausalVariables('patient_journeys');

      // Brand rides as a flat param when set; all-brands sends NO brand param
      // (the backend's brand=None keeps the universals only).
      expect(seen).toEqual(['Fabhalta', null, null]);
    });
  });
});

// #1763: the clinical context must follow the analysis, so the treatment column has
// to reach the backend. Nothing pinned the (brand, outcome)-only signature before.
describe('getClinicalContext (#1763)', () => {
  const CONTEXT = {
    brand: 'Kisqali',
    drug_name: 'ribociclib',
    disease: 'Malignant neoplasm of breast',
    our_outcome: 'persistent_180d',
    our_treatment: 'copay_support',
    mapped_endpoint: 'Treatment persistence / duration of therapy',
    treatment_context: {
      column: 'copay_support',
      label: 'Copay support',
      framing: 'receiving copay assistance',
      kind: 'commercial',
      source: 'curated',
    },
    analysis_framing:
      'This analysis estimates the effect of receiving copay assistance on 180-day treatment persistence for ribociclib in Malignant neoplasm of breast.',
    mechanism: { mechanism_of_action: 'CDK4/6 inhibitor', source: 'chembl' },
    pivotal_endpoints: { endpoints: [], source: 'clinicaltrials.gov' },
    real_world_evidence: null,
    honesty_label: 'estimate = synthetic; context = real',
  };

  it('sends the treatment as a query param when the analysis has one', async () => {
    let captured: URL | null = null;
    server.use(
      http.get(`${env.apiUrl}/causal/clinical-context`, ({ request }) => {
        captured = new URL(request.url);
        return HttpResponse.json(CONTEXT);
      })
    );

    const context = await getClinicalContext('Kisqali', 'persistent_180d', 'copay_support');

    expect(captured!.searchParams.get('brand')).toBe('Kisqali');
    expect(captured!.searchParams.get('outcome')).toBe('persistent_180d');
    expect(captured!.searchParams.get('treatment')).toBe('copay_support');
    // The response's analysis fields survive the un-validated get<T> path.
    expect(context.our_treatment).toBe('copay_support');
    expect(context.treatment_context?.kind).toBe('commercial');
    expect(context.analysis_framing).toContain('copay assistance');
  });

  it('omits the treatment param entirely on the brand-level call', async () => {
    let captured: URL | null = null;
    server.use(
      http.get(`${env.apiUrl}/causal/clinical-context`, ({ request }) => {
        captured = new URL(request.url);
        return HttpResponse.json({ ...CONTEXT, our_treatment: null, treatment_context: null });
      })
    );

    await getClinicalContext('Kisqali', 'persistent_180d');

    expect(captured!.searchParams.has('treatment')).toBe(false);
  });
});
