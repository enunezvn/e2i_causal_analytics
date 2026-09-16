/**
 * Drift check: the hand-written expert-review mirror vs the generated contract (#2092).
 *
 * `./expert-review.ts` mirrors the expert-review schemas in `./generated/api.ts`
 * by hand, and the API client imports the mirror, so nothing else ever compares
 * the two. Every assertion here is TYPE-LEVEL: `expectTypeOf` is a no-op at
 * runtime and the check is enforced by `tsc` (`npm run typecheck`, which covers
 * `src/**` including this file), not by vitest's runner.
 *
 * Per schema:
 *  1. the KEY SETS are equal — a field added, renamed or removed on either side
 *     fails (index signatures are ignored: `DagStructureSnapshot` is open-ended
 *     on the backend);
 *  2. every other property is compared in BOTH directions — generated → mirror
 *     (what the backend sends satisfies what the UI reads) and mirror →
 *     generated (the mirror did not silently widen or go stale).
 *
 * A relaxation is a key excluded from a direction, listed by name with its
 * reason beside the assertion it narrows; each one still pins what it can (the
 * key, the non-optional type, the nullability). Containers compare their own
 * scalars and assert each nested field points at the pair pinned on its own.
 * Every relaxation listed was measured to be load-bearing: removing it fails
 * `tsc` on the current contract. Not covered:
 * `AgentAssessment` / `AssessmentItem` / `AssessmentVerdict`, which have no
 * generated counterpart (the backend types the assessment as an opaque dict).
 */
import { describe, it, expectTypeOf } from 'vitest';
import type { components } from './generated/api';
import type {
  AgentAssessmentResponse,
  DagChanges,
  DagStructure,
  DiscoveryGate,
  ExpertReviewDetailResponse,
  PendingReviewItem,
  PendingReviewsResponse,
  ResolveReviewRequest,
  ResolveReviewResponse,
  ReviewRecord,
  ReviewSummaryResponse,
  ReviewVersion,
} from './expert-review';

type G = components['schemas'];

/** The declared properties only — drops `[key: string]: unknown` index signatures. */
type Declared<T> = {
  [K in keyof T as string extends K ? never : number extends K ? never : K]: T[K];
};

/**
 * Relaxations shared by the pending row and the full record (ReviewRecord extends
 * PendingReviewItem):
 * - `version_count`: optional in the mirror, required-with-server-default (1) in
 *   the generated type, so a partial row literal stays valid. Excluded from
 *   mirror → generated only; generated → mirror still checks it.
 * - `agent_assessment_json`: the backend types it `Dict[str, Any]`; the mirror
 *   narrows it to `AgentAssessment`. Neither direction is assignable, so it is
 *   excluded from both — its key and its nullability are still pinned.
 * - `dag_structure_json`: `DagStructure` is an interface without the snapshot's
 *   open index signature, so mirror → generated cannot hold. Excluded from that
 *   direction only; the snapshot pair is pinned on its own below.
 */
type RowMirrorToGeneratedRelaxed = 'version_count' | 'agent_assessment_json' | 'dag_structure_json';
type RowGeneratedToMirrorRelaxed = 'agent_assessment_json';

describe('expert-review mirror types match generated/api.ts (#2092)', () => {
  it('DagStructure ↔ DagStructureSnapshot', () => {
    expectTypeOf<keyof DagStructure>().toEqualTypeOf<keyof Declared<G['DagStructureSnapshot']>>();
    expectTypeOf<G['DagStructureSnapshot']>().toExtend<DagStructure>();
    expectTypeOf<Declared<G['DagStructureSnapshot']>>().toEqualTypeOf<DagStructure>();
    expectTypeOf<DiscoveryGate>().toEqualTypeOf<
      NonNullable<G['DagStructureSnapshot']['discovery_gate_decision']>
    >();
  });

  it('DagChanges', () => {
    expectTypeOf<keyof DagChanges>().toEqualTypeOf<keyof G['DagChanges']>();
    expectTypeOf<G['DagChanges']>().toEqualTypeOf<DagChanges>();
  });

  it('ReviewVersion', () => {
    expectTypeOf<keyof ReviewVersion>().toEqualTypeOf<keyof G['ReviewVersion']>();
    expectTypeOf<G['ReviewVersion']>().toExtend<ReviewVersion>();
    expectTypeOf<Omit<ReviewVersion, 'dag_structure_json'>>().toEqualTypeOf<
      Omit<G['ReviewVersion'], 'dag_structure_json'>
    >();
  });

  it('PendingReviewItem (version_count? relaxation)', () => {
    expectTypeOf<keyof PendingReviewItem>().toEqualTypeOf<keyof G['PendingReviewItem']>();
    expectTypeOf<Omit<G['PendingReviewItem'], RowGeneratedToMirrorRelaxed>>().toExtend<
      Omit<PendingReviewItem, RowGeneratedToMirrorRelaxed>
    >();
    expectTypeOf<Omit<PendingReviewItem, RowMirrorToGeneratedRelaxed>>().toExtend<
      Omit<G['PendingReviewItem'], RowMirrorToGeneratedRelaxed>
    >();
    // The relaxation is exactly "optional", never a different type.
    expectTypeOf<Required<PendingReviewItem>['version_count']>().toEqualTypeOf<
      G['PendingReviewItem']['version_count']
    >();
    expectTypeOf<Extract<PendingReviewItem['agent_assessment_json'], null>>().toEqualTypeOf<
      Extract<G['PendingReviewItem']['agent_assessment_json'], null>
    >();
  });

  it('ReviewRecord', () => {
    expectTypeOf<keyof ReviewRecord>().toEqualTypeOf<keyof G['ReviewRecord']>();
    expectTypeOf<Omit<G['ReviewRecord'], RowGeneratedToMirrorRelaxed>>().toExtend<
      Omit<ReviewRecord, RowGeneratedToMirrorRelaxed>
    >();
    expectTypeOf<Omit<ReviewRecord, RowMirrorToGeneratedRelaxed>>().toExtend<
      Omit<G['ReviewRecord'], RowMirrorToGeneratedRelaxed>
    >();
    expectTypeOf<Required<ReviewRecord>['version_count']>().toEqualTypeOf<
      G['ReviewRecord']['version_count']
    >();
  });

  it('PendingReviewsResponse and ExpertReviewDetailResponse', () => {
    // Containers: the scalar fields compare both ways; each nested field must
    // point at the pair already pinned above on BOTH sides (comparing the nested
    // rows again here would re-run them without their relaxations).
    expectTypeOf<keyof PendingReviewsResponse>().toEqualTypeOf<keyof G['PendingReviewsResponse']>();
    expectTypeOf<PendingReviewsResponse['total']>().toEqualTypeOf<G['PendingReviewsResponse']['total']>();
    expectTypeOf<PendingReviewsResponse['reviews']>().toEqualTypeOf<PendingReviewItem[]>();
    expectTypeOf<G['PendingReviewsResponse']['reviews']>().toEqualTypeOf<G['PendingReviewItem'][]>();

    expectTypeOf<keyof ExpertReviewDetailResponse>().toEqualTypeOf<
      keyof G['ExpertReviewDetailResponse']
    >();
    expectTypeOf<ExpertReviewDetailResponse['review']>().toEqualTypeOf<ReviewRecord>();
    expectTypeOf<G['ExpertReviewDetailResponse']['review']>().toEqualTypeOf<G['ReviewRecord']>();
    expectTypeOf<ExpertReviewDetailResponse['history']>().toEqualTypeOf<ReviewRecord[]>();
    expectTypeOf<G['ExpertReviewDetailResponse']['history']>().toEqualTypeOf<G['ReviewRecord'][]>();
    expectTypeOf<ExpertReviewDetailResponse['versions']>().toEqualTypeOf<ReviewVersion[]>();
    expectTypeOf<G['ExpertReviewDetailResponse']['versions']>().toEqualTypeOf<G['ReviewVersion'][]>();
    expectTypeOf<Pick<ExpertReviewDetailResponse, 'current_version_id'>>().toEqualTypeOf<
      Pick<G['ExpertReviewDetailResponse'], 'current_version_id'>
    >();
  });

  it('ResolveReviewRequest (a REQUEST: mirror → generated is the contract)', () => {
    expectTypeOf<keyof ResolveReviewRequest>().toEqualTypeOf<keyof G['ResolveReviewRequest']>();
    // - `validity_days`: required-with-server-default (90) in the generated type;
    //   the UI may omit it. Excluded from mirror → generated only.
    expectTypeOf<Omit<ResolveReviewRequest, 'validity_days'>>().toExtend<
      Omit<G['ResolveReviewRequest'], 'validity_days'>
    >();
    expectTypeOf<Required<ResolveReviewRequest>['validity_days']>().toEqualTypeOf<
      G['ResolveReviewRequest']['validity_days']
    >();
    // - `checklist`: optional with a server default ({}) in the generated type;
    //   the mirror REQUIRES it because the resolve form always sends it.
    //   Excluded from generated → mirror only.
    expectTypeOf<Omit<G['ResolveReviewRequest'], 'checklist'>>().toExtend<
      Omit<ResolveReviewRequest, 'checklist'>
    >();
    expectTypeOf<ResolveReviewRequest['checklist']>().toEqualTypeOf<
      Required<G['ResolveReviewRequest']>['checklist']
    >();
  });

  it('ResolveReviewResponse, ReviewSummaryResponse, AgentAssessmentResponse', () => {
    expectTypeOf<G['ResolveReviewResponse']>().toEqualTypeOf<ResolveReviewResponse>();
    expectTypeOf<G['ReviewSummaryResponse']>().toEqualTypeOf<ReviewSummaryResponse>();
    expectTypeOf<keyof AgentAssessmentResponse>().toEqualTypeOf<
      keyof G['AgentAssessmentResponse']
    >();
    // - `assessment`: an opaque dict on the backend, narrowed to AgentAssessment
    //   by the mirror (same reason as agent_assessment_json). Excluded both ways.
    expectTypeOf<Omit<AgentAssessmentResponse, 'assessment'>>().toEqualTypeOf<
      Omit<G['AgentAssessmentResponse'], 'assessment'>
    >();
  });
});
