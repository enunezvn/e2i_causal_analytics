/**
 * Approve / reject one pending review with the 010 checklist and an advisory
 * agent assessment. The assessment is generated automatically the first time
 * the form mounts for a row without a cached one (spec §4.5) — once per
 * review id, StrictMode-safe — and can be regenerated on demand. It never
 * pre-fills the human checklist.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import { CheckCircle2, RefreshCw, Sparkles, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { useResolveReview, useReviewAssessment } from '@/hooks/api/use-expert-review';
import type { AgentAssessment, PendingReviewItem, ReviewApprovalStatus } from '@/types/expert-review';
import { CHECKLIST_ITEMS, VERDICT_VARIANT } from './checklist';

export interface ResolveFormProps {
  review: PendingReviewItem;
  onClose: () => void;
  /**
   * Page-level once-per-review-id guard for the auto-generated assessment. The
   * linked-review card and the queue row can both mount a form for the SAME
   * review; sharing one Set keeps the backend from building the assessment
   * twice (pre-execution review 2026-09-08, codex MED). Optional so the form
   * still guards itself when rendered alone.
   */
  autoAssessGuard?: MutableRefObject<Set<string>>;
}

export function ResolveForm({ review, onClose, autoAssessGuard }: ResolveFormProps) {
  // Unique per FORM INSTANCE: the linked card and the queue row can render the
  // same review, and duplicate element ids would let a label operate the other
  // form (pre-execution review iter-2, codex MED).
  const uid = useId();
  const [checklist, setChecklist] = useState<Record<string, boolean>>({});
  const [comments, setComments] = useState('');
  const resolve = useResolveReview();

  // Once-per-review-id guard for the auto-generated assessment (spec §4.5): a
  // Set shared by every form on the page when the page provides one, so a second
  // form for the same id (linked card + queue row) and StrictMode's double-run
  // effect both hit it. Defined BEFORE the mutation hook so onError can use it.
  const localGuard = useRef<Set<string>>(new Set());
  const guard = autoAssessGuard ?? localGuard;
  const assessmentMutation = useReviewAssessment({
    // Hook-level, not per-call: the Mutation itself runs this, so it fires even
    // after the form has unmounted (a collapsed row), whereas TanStack skips the
    // per-call mutate callbacks once the observer is gone. Releasing the id lets
    // a later expand, or the page's Prepare button, retry the FAILED generation
    // once (the effect's deps do not change on error, so there is no loop).
    // OWNERSHIP: only the AUTOMATIC request's failure releases the id. A manual
    // Generate from a second form for the same review (linked card + queue row)
    // fails independently while the automatic request may still be in flight;
    // releasing then would let a remount or Prepare start a duplicate request.
    onError: (_error, variables) => {
      if (variables.auto) guard.current.delete(variables.reviewId);
    },
  });
  const { mutate: generateAssessment } = assessmentMutation;

  // Prefer the freshly generated assessment; fall back to the row's cache.
  const assessment: AgentAssessment | null =
    assessmentMutation.data?.assessment ?? review.agent_assessment_json ?? null;
  const assessmentById = new Map((assessment?.items ?? []).map((item) => [item.id, item]));

  useEffect(() => {
    if (assessment) return;
    if (guard.current.has(review.review_id)) return;
    // Deferred past StrictMode's synchronous effect cleanup + re-run. A mutate
    // issued in the FIRST pass is orphaned: query-core's MutationObserver
    // detaches from the in-flight mutation on unsubscribe and never re-attaches
    // (mutationObserver.js onUnsubscribe), so the form stayed pending after the
    // request had completed (measured under <StrictMode>, which main.tsx uses).
    // The guard is marked when the timer fires, so a cancelled pass marks nothing.
    const timer = setTimeout(() => {
      if (guard.current.has(review.review_id)) return;
      guard.current.add(review.review_id);
      generateAssessment({ reviewId: review.review_id, auto: true });
    }, 0);
    return () => clearTimeout(timer);
  }, [assessment, generateAssessment, guard, review.review_id]);

  const submit = useCallback(
    (approval_status: ReviewApprovalStatus) => {
      resolve.mutate(
        {
          reviewId: review.review_id,
          body: {
            approval_status,
            checklist,
            comments: comments ? { note: comments } : undefined,
          },
        },
        { onSuccess: onClose }
      );
    },
    [resolve, review.review_id, checklist, comments, onClose]
  );

  return (
    <div className="space-y-4 rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/20 p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
          <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
          Agent assessment (advisory — the checklist answers are yours)
          {assessment?.is_fallback && ' · deterministic, no LLM'}
        </span>
        <Button
          size="sm"
          variant="outline"
          onClick={() => generateAssessment({ reviewId: review.review_id, force: !!assessment })}
          disabled={assessmentMutation.isPending}
        >
          <RefreshCw
            className={`mr-1 h-3.5 w-3.5 ${assessmentMutation.isPending ? 'animate-spin' : ''}`}
          />
          {assessment ? 'Regenerate agent assessment' : 'Generate agent assessment'}
        </Button>
      </div>

      {assessmentMutation.isError && (
        <WarningBanner
          title="Failed to generate agent assessment"
          messages={[assessmentMutation.error?.message ?? 'An unexpected error occurred.']}
        />
      )}

      <div className="space-y-2">
        {CHECKLIST_ITEMS.map((item) => {
          const graded = assessmentById.get(item.id);
          return (
            <div key={item.id} className="space-y-0.5">
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`${uid}-${item.id}`}
                  checked={!!checklist[item.id]}
                  onCheckedChange={(v) =>
                    setChecklist((prev) => ({ ...prev, [item.id]: v === true }))
                  }
                />
                <Label htmlFor={`${uid}-${item.id}`} className="text-sm">
                  {item.question}
                </Label>
                {graded && (
                  <Badge variant={VERDICT_VARIANT[graded.verdict] ?? 'outline'}>{graded.verdict}</Badge>
                )}
              </div>
              {graded && (
                <p className="pl-6 text-xs text-[var(--color-muted-foreground)]">{graded.rationale}</p>
              )}
            </div>
          );
        })}
      </div>

      <div className="space-y-1">
        <Label htmlFor={`${uid}-comments`} className="text-sm">
          Comments
        </Label>
        <textarea
          id={`${uid}-comments`}
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          rows={3}
          className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 text-sm"
          placeholder="Reviewer notes (optional)"
        />
      </div>

      {resolve.isError && (
        <WarningBanner
          title="Failed to submit review"
          messages={[resolve.error?.message ?? 'An unexpected error occurred.']}
        />
      )}

      <div className="flex items-center gap-2">
        <Button size="sm" onClick={() => submit('approved')} disabled={resolve.isPending}>
          <CheckCircle2 className="mr-1 h-4 w-4" />
          Approve
        </Button>
        <Button size="sm" variant="destructive" onClick={() => submit('rejected')} disabled={resolve.isPending}>
          <XCircle className="mr-1 h-4 w-4" />
          Reject
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose} disabled={resolve.isPending}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
