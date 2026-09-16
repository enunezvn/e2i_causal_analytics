/**
 * Render a review's stored DAG snapshot, or an honest fallback for pre-097 rows.
 *
 * Under the graph it renders the delta of the version the review is CURRENTLY
 * on (migration 141), so an operator sees exactly what moved since the last
 * sign-off of the structure they are being asked to approve.
 *
 * The caller NAMES that version (`currentVersionId`, from the detail
 * response); the panel never picks one itself. The timeline is a set of facts
 * and may end on a row the review is not on — a run that recorded its version
 * and then lost the compare-and-set advance leaves an orphan after the winner —
 * so the last entry's delta can describe a losing structure. With no name, or
 * a name that is not in the timeline, the panel shows the graph alone: silence
 * is the honest answer, even with a long timeline visibly on offer.
 */
import { CausalDAG } from '@/components/visualizations/causal/CausalDAG';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type { DagStructure, ReviewVersion } from '@/types/expert-review';
import { DagDiff } from './DagDiff';

export function DagPanel({
  structure,
  versions,
  currentVersionId,
}: {
  structure?: DagStructure | null;
  versions?: ReviewVersion[] | null;
  /**
   * `ExpertReviewDetailResponse.current_version_id` — which timeline entry the
   * review is on. Undefined (not yet loaded) or null (no entry carries the
   * review's version pair) both mean NO delta.
   */
  currentVersionId?: string | null;
}) {
  // Only the named entry's delta. Its `changes` is null on the FIRST version
  // (nothing to diff against) and on a review minted before the versions table.
  //
  // `is_changed` false is near-unreachable by construction — the engine appends
  // a version only when the structure or the covariate set moved — but the flag
  // is the contract, so an unchanged delta renders NOTHING rather than a
  // heading promising a change the panel cannot show. (DagDiff's own
  // "no structural change" copy is for a caller that asks for a named delta;
  // here the honest answer is silence.)
  const current = currentVersionId
    ? versions?.find((v) => v.version_id === currentVersionId)
    : undefined;
  const currentChanges = current?.changes?.is_changed ? current.changes : null;

  if (!structure?.nodes?.length) {
    return (
      <div className="rounded-md border border-dashed border-[var(--color-border)] p-4 text-sm text-[var(--color-muted-foreground)]">
        DAG structure not captured for this review (created before snapshot capture was
        added). The DAG hash identifies the structure but cannot be rendered from it.
      </div>
    );
  }

  const treatments = new Set(structure.treatment_nodes ?? []);
  const outcomes = new Set(structure.outcome_nodes ?? []);
  const augmented = new Set((structure.augmented_edges ?? []).map(([s, t]) => `${s}->${t}`));

  const nodes: CausalNode[] = structure.nodes.map((id) => ({
    id,
    label: id,
    type: treatments.has(id) ? 'treatment' : outcomes.has(id) ? 'outcome' : 'variable',
  }));
  const edges: CausalEdge[] = (structure.edges ?? []).map(([source, target]) => ({
    id: `${source}->${target}`,
    source,
    target,
    // Discovery-augmented edges are visually distinct: the discovery gate added them.
    type: augmented.has(`${source}->${target}`) ? 'association' : 'causal',
  }));

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">DAG under review</h4>
      <CausalDAG nodes={nodes} edges={edges} minHeight={320} ariaLabel="Causal DAG under review" />
      {structure.augmented_edges && structure.augmented_edges.length > 0 && (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Dashed/association edges were discovery-augmented (gate=
          {structure.discovery_gate_decision ?? 'unknown'}).
        </p>
      )}
      {currentChanges && (
        <div className="space-y-1">
          <h4 className="text-sm font-medium">Changed since the previous version</h4>
          <DagDiff changes={currentChanges} />
        </div>
      )}
    </div>
  );
}
