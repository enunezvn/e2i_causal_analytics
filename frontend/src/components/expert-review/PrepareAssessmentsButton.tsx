/**
 * Walk the visible pending rows that have no cached assessment and generate
 * one each, ONE AT A TIME (each call is cached server-side). Shows k / n,
 * is cancellable, stops on the first error and shows it. No new endpoint.
 */
import { useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Sparkles, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { generateReviewAssessment } from '@/api/expert-review';
import { queryKeys } from '@/lib/query-client';
import type { PendingReviewItem } from '@/types/expert-review';

interface RunState {
  running: boolean;
  done: number;
  total: number;
  error: string | null;
}

export function PrepareAssessmentsButton({
  reviews,
  autoAssessGuard,
}: {
  reviews: PendingReviewItem[];
  /** The page's once-per-review-id guard; marked before each request so a form
   *  expanded meanwhile does not start a second generation for the same review. */
  autoAssessGuard?: MutableRefObject<Set<string>>;
}) {
  const queryClient = useQueryClient();
  const cancelRef = useRef(false);
  const [state, setState] = useState<RunState>({ running: false, done: 0, total: 0, error: null });
  const missing = reviews.filter((r) => !r.agent_assessment_json);

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: [...queryKeys.expertReviews.all(), 'pending'] });

  const run = async () => {
    cancelRef.current = false;
    // Skip reviews whose generation a form already started (shared guard); a
    // second request for the same id would only race the first.
    const todo = missing.filter((r) => !autoAssessGuard?.current.has(r.review_id));
    setState({ running: true, done: 0, total: todo.length, error: null });
    for (const review of todo) {
      if (cancelRef.current) break;
      // Re-check per iteration: a form expanded while an earlier request was
      // in flight may have started this one meanwhile.
      if (autoAssessGuard?.current.has(review.review_id)) {
        setState((s) => ({ ...s, done: s.done + 1 }));
        continue;
      }
      try {
        autoAssessGuard?.current.add(review.review_id);
        await generateReviewAssessment(review.review_id);
        setState((s) => ({ ...s, done: s.done + 1 }));
      } catch (e) {
        // Release the id so a later "Prepare" (or the form's button) can retry it.
        autoAssessGuard?.current.delete(review.review_id);
        setState((s) => ({
          ...s,
          running: false,
          error: e instanceof Error ? e.message : 'Assessment generation failed.',
        }));
        await invalidate();
        return;
      }
    }
    setState((s) => ({ ...s, running: false }));
    await invalidate();
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        size="sm"
        variant="outline"
        onClick={run}
        disabled={state.running || missing.length === 0}
        aria-label="Prepare assessments"
      >
        <Sparkles className="mr-1 h-4 w-4" />
        {state.running
          ? `Preparing ${state.done} / ${state.total}…`
          : `Prepare assessments (${missing.length} missing)`}
      </Button>
      {state.running && (
        <Button size="sm" variant="ghost" onClick={() => (cancelRef.current = true)}>
          <Square className="mr-1 h-3.5 w-3.5" />
          Stop
        </Button>
      )}
      {state.error && (
        <WarningBanner
          title={`Stopped after ${state.done} of ${state.total}`}
          messages={[state.error]}
          className="basis-full"
        />
      )}
    </div>
  );
}
