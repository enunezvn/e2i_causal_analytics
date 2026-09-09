/**
 * The review the causal drill-down linked to (`/expert-reviews?review=<id>`):
 * one row in ANY status plus every review of the same DAG structure. A pending
 * linked review resolves in place; a resolved one shows who decided what.
 */
import type { MutableRefObject } from 'react';
import { RefreshCw } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { useExpertReview } from '@/hooks/api/use-expert-review';
import { DagPanel } from './DagPanel';
import { ResolveForm } from './ResolveForm';
import { shortHash, statusVariant } from './checklist';

function fmtDate(value?: string | null): string {
  if (!value) return '—';
  return value.slice(0, 10);
}

/**
 * Status-specific resolved copy matching the gate's precedence
 * (src/causal_engine/expert_review_gate.py check_approval, ~:272-350): the
 * ACTIVE approval governs unless a NEWER rejection supersedes it; a newer
 * pending row reopens a REJECTED structure but does not displace an approval.
 */
function resolvedCopy(status?: string | null): string {
  if (status === 'approved') {
    return 'This review is resolved. Its approval applies until it expires or a newer review rejects the structure.';
  }
  if (status === 'rejected') {
    return 'This review is resolved. The rejection holds until a newer pending review of the same structure reopens it.';
  }
  return 'This review is resolved.';
}

export function LinkedReviewCard({
  reviewId,
  autoAssessGuard,
}: {
  reviewId: string;
  autoAssessGuard?: MutableRefObject<Set<string>>;
}) {
  const q = useExpertReview(reviewId);
  // The history INCLUDES the linked review itself (GET /expert-reviews/{id});
  // hoisted so the row marker survives TS narrowing inside the map callback.
  const currentId = q.data?.review.review_id;
  // Branch on the HTTP status (ApiError.status); the message text is only a
  // fallback for an error that carries no status at all.
  const notFound =
    q.error?.status === 404 ||
    (q.error?.status === undefined && /not found/i.test(q.error?.message ?? ''));

  return (
    <Card data-testid="linked-review">
      <CardHeader>
        <CardTitle>Linked review</CardTitle>
        <CardDescription>
          Opened from a causal analysis. Review <span className="font-mono">{shortHash(reviewId)}</span>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {q.isLoading && (
          <div className="flex items-center justify-center py-6">
            <RefreshCw className="h-5 w-5 animate-spin text-[var(--color-muted-foreground)]" />
          </div>
        )}
        {q.isError && (
          <WarningBanner
            title={notFound ? 'This review no longer exists' : 'Failed to load the linked review'}
            messages={[q.error?.message ?? 'An unexpected error occurred.']}
          />
        )}
        {q.data && (
          <>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant={statusVariant(q.data.review.approval_status)}>
                {q.data.review.approval_status ?? 'unknown'}
              </Badge>
              <span>{q.data.review.brand ?? 'no brand'}</span>
              <span>·</span>
              <span>
                {q.data.review.treatment_variable ?? '—'} → {q.data.review.outcome_variable ?? '—'}
              </span>
              <span>·</span>
              <span>created {fmtDate(q.data.review.created_at)}</span>
            </div>
            {q.data.review.approval_status !== 'pending' && (
              <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
                <dt className="text-[var(--color-muted-foreground)]">Reviewer</dt>
                <dd>{q.data.review.reviewer_name ?? q.data.review.reviewer_id ?? '—'}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Decided</dt>
                <dd>{fmtDate(q.data.review.approved_at ?? q.data.review.created_at)}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Valid until</dt>
                <dd>{q.data.review.valid_until ? fmtDate(q.data.review.valid_until) : 'no expiry recorded'}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Concerns</dt>
                <dd>{q.data.review.concerns_raised?.length ? q.data.review.concerns_raised.join('; ') : '—'}</dd>
                <dt className="text-[var(--color-muted-foreground)]">Conditions</dt>
                <dd>{q.data.review.conditions ?? '—'}</dd>
              </dl>
            )}
            <div className="grid gap-4 xl:grid-cols-2">
              <DagPanel structure={q.data.review.dag_structure_json} />
              {q.data.review.approval_status === 'pending' ? (
                <ResolveForm
                  review={q.data.review}
                  onClose={() => undefined}
                  autoAssessGuard={autoAssessGuard}
                />
              ) : (
                <div className="text-sm text-[var(--color-muted-foreground)]">
                  {resolvedCopy(q.data.review.approval_status)}
                </div>
              )}
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Same structure, all reviews</h4>
              {q.data.history.length === 0 ? (
                <p className="text-xs text-[var(--color-muted-foreground)]">No other reviews share this DAG hash.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Review</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Reviewer</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {q.data.history.map((h) => {
                      const isCurrent = h.review_id === currentId;
                      return (
                        <TableRow key={h.review_id} data-current={isCurrent ? 'true' : undefined}>
                          <TableCell className="font-mono text-xs">
                            {shortHash(h.review_id)}
                            {isCurrent && (
                              <span className="ml-1 text-[var(--color-muted-foreground)]">(this review)</span>
                            )}
                          </TableCell>
                          <TableCell>
                            <Badge variant={statusVariant(h.approval_status)}>{h.approval_status ?? '—'}</Badge>
                          </TableCell>
                          <TableCell>{fmtDate(h.created_at)}</TableCell>
                          <TableCell>{h.reviewer_name ?? '—'}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
