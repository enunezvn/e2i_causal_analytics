/**
 * Expert Reviews Page (R6-F2 Phase B4; DAG snapshot + advisory assessment 097)
 * ============================================================================
 *
 * Admin review-queue UI for the causal-DAG human-in-the-loop loop.
 *
 * A REVIEW-band causal estimate creates a `pending` expert_reviews row; an
 * operator sees it here and resolves it (approve/reject) with the 010 checklist
 * items + comments. The expanded row renders the DAG under review from its
 * stored snapshot (`dag_structure_json`) — rows created before snapshot capture
 * show an honest "not captured" fallback — and an ADVISORY agent assessment of
 * the checklist questions (`agent_assessment_json`, generated on demand). The
 * assessment never pre-fills the human checklist.
 *
 * Honest states: loading spinner, error banner, and an EmptyState (no hardcoded
 * SAMPLE_ data) when the live queue is empty.
 *
 * @module pages/ExpertReviews
 */

import { Fragment, useState, useCallback } from 'react';
import {
  ClipboardCheck,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Inbox,
  Sparkles,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { EmptyState } from '@/components/ui/EmptyState';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { CausalDAG } from '@/components/visualizations/causal/CausalDAG';
import type {
  CausalNode,
  CausalEdge,
} from '@/components/visualizations/causal/CausalDAG';
import {
  usePendingReviews,
  useResolveReview,
  useReviewAssessment,
  useReviewSummary,
} from '@/hooks/api/use-expert-review';
import type {
  AgentAssessment,
  AssessmentVerdict,
  DagStructure,
  PendingReviewItem,
  ReviewApprovalStatus,
} from '@/types/expert-review';

// The minimal reviewer checklist (the 010 SYSTEM_TEMPLATE required items).
const CHECKLIST_ITEMS: { id: string; question: string }[] = [
  { id: 'conf_complete', question: 'Are all known confounders included?' },
  { id: 'edge_plausible', question: 'Do causal arrows reflect domain knowledge?' },
  { id: 'no_forbidden', question: 'Are there no forbidden edges (future→past)?' },
  { id: 'mediators_correct', question: 'Are intermediate variables correctly positioned?' },
  { id: 'sutva_plausible', question: 'Is the no-interference assumption reasonable?' },
  { id: 'positivity', question: 'Is there sufficient overlap in treatment groups?' },
];

function shortHash(hash?: string | null): string {
  if (!hash) return '—';
  return hash.length > 12 ? `${hash.slice(0, 12)}…` : hash;
}

// Verdict chip styling: concern is the only destructive signal; supports and
// the two "human judgment required" verdicts stay visually calm.
const VERDICT_VARIANT: Record<AssessmentVerdict, 'secondary' | 'destructive' | 'outline'> =
  {
    supports: 'secondary',
    concern: 'destructive',
    unclear: 'outline',
    no_evidence: 'outline',
  };

/** Render the stored DAG snapshot, or an honest fallback for pre-097 rows. */
function DagPanel({ structure }: { structure?: DagStructure | null }) {
  if (!structure?.nodes?.length) {
    return (
      <div className="rounded-md border border-dashed border-[var(--color-border)] p-4 text-sm text-[var(--color-muted-foreground)]">
        DAG structure not captured for this review (created before snapshot
        capture was added). The DAG hash identifies the structure but cannot be
        rendered from it.
      </div>
    );
  }

  const treatments = new Set(structure.treatment_nodes ?? []);
  const outcomes = new Set(structure.outcome_nodes ?? []);
  const augmented = new Set(
    (structure.augmented_edges ?? []).map(([s, t]) => `${s}->${t}`)
  );

  const nodes: CausalNode[] = structure.nodes.map((id) => ({
    id,
    label: id,
    type: treatments.has(id) ? 'treatment' : outcomes.has(id) ? 'outcome' : 'variable',
  }));
  const edges: CausalEdge[] = (structure.edges ?? []).map(([source, target]) => ({
    id: `${source}->${target}`,
    source,
    target,
    // Discovery-augmented edges are visually distinct: they were added by the
    // discovery gate, not the curated domain DAG.
    type: augmented.has(`${source}->${target}`) ? 'association' : 'causal',
  }));

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">DAG under review</h4>
      <CausalDAG
        nodes={nodes}
        edges={edges}
        minHeight={320}
        ariaLabel="Causal DAG under review"
      />
      {structure.augmented_edges && structure.augmented_edges.length > 0 && (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Dashed/association edges were discovery-augmented (gate=
          {structure.discovery_gate_decision ?? 'unknown'}).
        </p>
      )}
    </div>
  );
}

interface ResolveFormProps {
  review: PendingReviewItem;
  onClose: () => void;
}

function ResolveForm({ review, onClose }: ResolveFormProps) {
  const [checklist, setChecklist] = useState<Record<string, boolean>>({});
  const [comments, setComments] = useState('');
  const resolve = useResolveReview();
  const assessmentMutation = useReviewAssessment();

  // Prefer the freshly generated assessment; fall back to the row's cache.
  const assessment: AgentAssessment | null =
    assessmentMutation.data?.assessment ?? review.agent_assessment_json ?? null;
  const assessmentById = new Map(
    (assessment?.items ?? []).map((item) => [item.id, item])
  );

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
          onClick={() =>
            assessmentMutation.mutate({
              reviewId: review.review_id,
              force: !!assessment,
            })
          }
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
          messages={[
            assessmentMutation.error?.message ?? 'An unexpected error occurred.',
          ]}
        />
      )}

      <div className="space-y-2">
        {CHECKLIST_ITEMS.map((item) => {
          const graded = assessmentById.get(item.id);
          return (
            <div key={item.id} className="space-y-0.5">
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`${review.review_id}-${item.id}`}
                  checked={!!checklist[item.id]}
                  onCheckedChange={(v) =>
                    setChecklist((prev) => ({ ...prev, [item.id]: v === true }))
                  }
                />
                <Label htmlFor={`${review.review_id}-${item.id}`} className="text-sm">
                  {item.question}
                </Label>
                {graded && (
                  <Badge variant={VERDICT_VARIANT[graded.verdict] ?? 'outline'}>
                    {graded.verdict}
                  </Badge>
                )}
              </div>
              {graded && (
                <p className="pl-6 text-xs text-[var(--color-muted-foreground)]">
                  {graded.rationale}
                </p>
              )}
            </div>
          );
        })}
      </div>

      <div className="space-y-1">
        <Label htmlFor={`${review.review_id}-comments`} className="text-sm">
          Comments
        </Label>
        <textarea
          id={`${review.review_id}-comments`}
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
        <Button
          size="sm"
          onClick={() => submit('approved')}
          disabled={resolve.isPending}
        >
          <CheckCircle2 className="mr-1 h-4 w-4" />
          Approve
        </Button>
        <Button
          size="sm"
          variant="destructive"
          onClick={() => submit('rejected')}
          disabled={resolve.isPending}
        >
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

export default function ExpertReviews() {
  const { data, isLoading, isError, error, refetch, isFetching } = usePendingReviews();
  const { data: summary } = useReviewSummary();
  const [openRow, setOpenRow] = useState<string | null>(null);

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

      {summary && (
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">Pending: {summary.pending}</Badge>
          <Badge variant="secondary">Approved: {summary.approved}</Badge>
          <Badge variant="secondary">Rejected: {summary.rejected}</Badge>
          <Badge variant="secondary">Expiring soon: {summary.expiring_soon}</Badge>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Pending Queue</CardTitle>
          <CardDescription>Oldest reviews first.</CardDescription>
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
                      <TableCell className="font-mono text-xs">
                        {shortHash(review.dag_version_hash)}
                      </TableCell>
                      <TableCell>{review.review_type ?? '—'}</TableCell>
                      <TableCell>
                        {review.days_pending != null ? Math.round(review.days_pending) : '—'}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            setOpenRow((prev) =>
                              prev === review.review_id ? null : review.review_id
                            )
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
