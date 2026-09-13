/**
 * The tool-composer observability response survives Zod parsing intact.
 *
 * Zod strips unknown keys by default, so a schema that forgets a field silently deletes it on
 * the way to the component — the failure looks like a backend bug and is not one. The fixture
 * is a response captured from a REAL database run (Task 14's real-DB test), so this asserts a
 * deep-equal round trip against what the API actually returns, not against an idealised shape.
 */
import { describe, it, expect } from 'vitest';

import { ToolComposerObservabilityResponseSchema } from './api-schemas';
import PAYLOAD from '@/components/admin/__fixtures__/tool-composer-observability.json';

describe('ToolComposerObservabilityResponseSchema', () => {
  it('parses the captured response and strips nothing', () => {
    const parsed = ToolComposerObservabilityResponseSchema.parse(PAYLOAD);

    expect(parsed).toEqual(PAYLOAD);
  });

  it('keeps a null measured latency null rather than defaulting it to a number', () => {
    const parsed = ToolComposerObservabilityResponseSchema.parse(PAYLOAD);

    const unmeasured = parsed.tools.find((tool) => tool.verdict === 'too_few_runs');
    expect(unmeasured).toBeDefined();
    expect(unmeasured?.p50_latency_ms).toBeNull();
    expect(unmeasured?.declared_latency_ms).toEqual(expect.any(Number));
  });

  it('keeps every step class of a recent failure', () => {
    const parsed = ToolComposerObservabilityResponseSchema.parse(PAYLOAD);

    const failure = parsed.recent_failures.find((f) => f.composition_id === 'comp_failed');
    expect(failure?.step_classes).toHaveLength(2);
    expect(failure?.step_classes.map((s) => s.outcome_class)).toEqual([
      'error',
      'dependency_unmet',
    ]);
  });

  it('rejects a response whose verdict is not one the rule can produce', () => {
    const payload = PAYLOAD as unknown as { tools: Record<string, unknown>[] };
    const broken = {
      ...(PAYLOAD as unknown as Record<string, unknown>),
      tools: [{ ...payload.tools[0], verdict: 'probably_fine' }],
    };

    expect(() => ToolComposerObservabilityResponseSchema.parse(broken)).toThrow();
  });

  // #2021: contract pins, not a live-path guard. `getToolComposerObservability` passes no
  // `schema` to `get()`, so this response is never parsed by this schema at request time. These
  // tests pin the wire contract itself, independent of whether this endpoint is wired to it.
  it('keeps a tool row refusal-reason fields when present', () => {
    const payload = PAYLOAD as unknown as { tools: Record<string, unknown>[] };
    const withReason = {
      ...(PAYLOAD as unknown as Record<string, unknown>),
      tools: [
        {
          ...payload.tools[0],
          most_common_refusal_reason: 'non_binary_treatment',
          most_common_refusal_sentence: 'the treatment column is not a binary 0/1 indicator',
          n_refused_coded: 8,
          n_most_common_refusal_reason: 3,
        },
        ...payload.tools.slice(1),
      ],
    };

    const parsed = ToolComposerObservabilityResponseSchema.parse(withReason);

    expect(parsed.tools[0]).toMatchObject({
      most_common_refusal_reason: 'non_binary_treatment',
      most_common_refusal_sentence: 'the treatment column is not a binary 0/1 indicator',
      n_refused_coded: 8,
      n_most_common_refusal_reason: 3,
    });
  });

  it('keeps a step class reason_code and reason when present', () => {
    const payload = PAYLOAD as unknown as {
      recent_failures: { step_classes: Record<string, unknown>[] }[];
    };
    const stepClasses = payload.recent_failures[0].step_classes;
    const withReason = {
      ...(PAYLOAD as unknown as Record<string, unknown>),
      recent_failures: [
        {
          ...(PAYLOAD as unknown as { recent_failures: Record<string, unknown>[] })
            .recent_failures[0],
          step_classes: [
            {
              ...stepClasses[0],
              reason_code: 'non_binary_treatment',
              reason: 'the treatment column is not a binary 0/1 indicator',
            },
            ...stepClasses.slice(1),
          ],
        },
        ...(PAYLOAD as unknown as { recent_failures: Record<string, unknown>[] })
          .recent_failures.slice(1),
      ],
    };

    const parsed = ToolComposerObservabilityResponseSchema.parse(withReason);

    expect(parsed.recent_failures[0].step_classes[0]).toMatchObject({
      reason_code: 'non_binary_treatment',
      reason: 'the treatment column is not a binary 0/1 indicator',
    });
  });

  it('keeps a step class reason_details, numbers as numbers and booleans as booleans', () => {
    const payload = PAYLOAD as unknown as { recent_failures: Record<string, unknown>[] };
    const failure = payload.recent_failures[0] as { step_classes: Record<string, unknown>[] };
    const details = { n_no_contrast: 3, n_non_finite: 0, share_kept: 0.25, is_scoped: true };
    const withDetails = {
      ...(PAYLOAD as unknown as Record<string, unknown>),
      recent_failures: [
        {
          ...failure,
          step_classes: [
            { ...failure.step_classes[0], reason_details: details },
            ...failure.step_classes.slice(1),
          ],
        },
        ...payload.recent_failures.slice(1),
      ],
    };

    const parsed = ToolComposerObservabilityResponseSchema.parse(withDetails);

    expect(parsed.recent_failures[0].step_classes[0].reason_details).toStrictEqual(details);
  });

  it('parses a payload with no refusal-reason fields at all (pre-ml/043 backend)', () => {
    const parsed = ToolComposerObservabilityResponseSchema.parse(PAYLOAD);

    expect(parsed.tools[0].most_common_refusal_reason).toBeUndefined();
    expect(parsed.recent_failures[0].step_classes[0].reason_code).toBeUndefined();
    expect(parsed.recent_failures[0].step_classes[0].reason_details).toBeUndefined();
  });
});
