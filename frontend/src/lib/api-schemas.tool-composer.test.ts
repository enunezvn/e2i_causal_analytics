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
});
