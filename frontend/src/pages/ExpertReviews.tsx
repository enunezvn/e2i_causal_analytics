/**
 * Expert Reviews Page (R6-F2 Phase B4; DAG snapshot + advisory assessment 097;
 * lane 1: linked review, brand filter, honest summary, assessment prefetch)
 * ============================================================================
 *
 * Admin review-queue UI for the causal-DAG human-in-the-loop loop.
 *
 * A REVIEW- or BLOCK-band causal estimate creates a `pending` expert_reviews
 * row; an operator sees it here and resolves it (approve/reject) with the 010
 * checklist items + comments. The expanded row renders the DAG under review
 * from its stored snapshot and an ADVISORY agent assessment that never
 * pre-fills the human checklist.
 *
 * Lane 1 (spec §4.5):
 * - `?review=<id>` opens a linked-review card (any status + same-DAG history),
 *   the destination of the causal drill-down's "Open review" link.
 * - The queue and the counts follow the GLOBAL brand filter (the same SSOT the
 *   Causal Analysis page reads, #1752). "All" is the only way to see the rows
 *   that carry no brand.
 * - A summary read failure renders a banner instead of silently dropping the
 *   counts.
 * - "Prepare assessments" generates the missing advisory assessments one row
 *   at a time; expanding a row generates its own if none is cached.
 *
 * Honest states: loading spinner, error banner, and an EmptyState (no hardcoded
 * SAMPLE_ data) when the live queue is empty.
 *
 * @module pages/ExpertReviews
 */

import { Fragment, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ClipboardCheck, Inbox, RefreshCw } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/EmptyState';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { DagPanel } from '@/components/expert-review/DagPanel';
import { LinkedReviewCard } from '@/components/expert-review/LinkedReviewCard';
import { PrepareAssessmentsButton } from '@/components/expert-review/PrepareAssessmentsButton';
import { ResolveForm } from '@/components/expert-review/ResolveForm';
import { shortHash } from '@/components/expert-review/checklist';
import { usePendingReviews, useReviewSummary } from '@/hooks/api/use-expert-review';
import { useE2IFilters } from '@/hooks/use-e2i-filters';

export default function ExpertReviews() {
  const [searchParams] = useSearchParams();
  const linkedReviewId = searchParams.get('review')?.trim() || null;

  const { filters } = useE2IFilters();
  const brand = filters.brand === 'All' ? undefined : (filters.brand as string);
  const params = brand ? { brand } : undefined;

  const { data, isLoading, isError, error, refetch, isFetching } = usePendingReviews(params);
  const summary = useReviewSummary(params);
  const [openRow, setOpenRow] = useState<string | null>(null);
  // One auto-generated assessment per review id across the page (the linked
  // card and a queue row can show the same review).
  const autoAssessGuard = useRef<Set<string>>(new Set());

  const reviews = data?.reviews ?? [];

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <ClipboardCheck className="h-6 w-6" />
            Expert Reviews
          </h1>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Human-in-the-loop validation queue for causal DAGs awaiting expert sign-off.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw className={`mr-1 h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {summary.isError && (
        <WarningBanner
          title="Review counts unavailable"
          messages={[summary.error?.message ?? 'An unexpected error occurred.']}
        />
      )}
      {summary.data && (
        <div className="flex flex-wrap gap-2">
          {/* pending/approved/rejected/expired partition the rows; expiring_soon
              is a SUBSET of approved (#1972), so it is labelled and styled as a
              qualifier rather than a fourth peer count that could be added in. */}
          <Badge variant="secondary">Pending: {summary.data.pending}</Badge>
          <Badge variant="secondary">Approved: {summary.data.approved}</Badge>
          <Badge variant="secondary">Rejected: {summary.data.rejected}</Badge>
          <Badge variant="secondary">Expired: {summary.data.expired}</Badge>
          <Badge variant="outline">of which expiring soon: {summary.data.expiring_soon}</Badge>
        </div>
      )}

      {linkedReviewId && (
        <LinkedReviewCard reviewId={linkedReviewId} autoAssessGuard={autoAssessGuard} />
      )}

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <CardTitle>Pending Queue</CardTitle>
              <CardDescription>
                {brand
                  ? `Oldest reviews first · brand: ${brand}. Reviews with no brand are listed under All.`
                  : 'Oldest reviews first · all brands, including reviews with no brand.'}
              </CardDescription>
            </div>
            {reviews.length > 0 && (
              <PrepareAssessmentsButton reviews={reviews} autoAssessGuard={autoAssessGuard} />
            )}
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-6 w-6 animate-spin text-[var(--color-muted-foreground)]" />
            </div>
          ) : isError ? (
            <WarningBanner
              title="Failed to load pending reviews"
              messages={[error?.message ?? 'An unexpected error occurred.']}
            />
          ) : reviews.length === 0 ? (
            <EmptyState
              icon={<Inbox className="h-8 w-8" aria-hidden="true" />}
              title="No pending reviews"
              description="REVIEW-band causal estimates will appear here for expert sign-off."
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Brand</TableHead>
                  <TableHead>Treatment</TableHead>
                  <TableHead>Outcome</TableHead>
                  <TableHead>DAG hash</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Age (days)</TableHead>
                  <TableHead className="text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {reviews.map((review) => (
                  <Fragment key={review.review_id}>
                    <TableRow>
                      <TableCell>{review.brand ?? '—'}</TableCell>
                      <TableCell>{review.treatment_variable ?? '—'}</TableCell>
                      <TableCell>{review.outcome_variable ?? '—'}</TableCell>
                      <TableCell className="font-mono text-xs">{shortHash(review.dag_version_hash)}</TableCell>
                      <TableCell>{review.review_type ?? '—'}</TableCell>
                      <TableCell>
                        {review.days_pending != null ? Math.round(review.days_pending) : '—'}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            setOpenRow((prev) => (prev === review.review_id ? null : review.review_id))
                          }
                        >
                          {openRow === review.review_id ? 'Close' : 'Review'}
                        </Button>
                      </TableCell>
                    </TableRow>
                    {openRow === review.review_id && (
                      <TableRow>
                        <TableCell colSpan={7}>
                          <div className="grid gap-4 xl:grid-cols-2">
                            <DagPanel structure={review.dag_structure_json} />
                            <ResolveForm
                              review={review}
                              onClose={() => setOpenRow(null)}
                              autoAssessGuard={autoAssessGuard}
                            />
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
