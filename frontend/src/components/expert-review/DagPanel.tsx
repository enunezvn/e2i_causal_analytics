/**
 * Render a review's stored DAG snapshot, or an honest fallback for pre-097 rows.
 *
 * When the estimand has more than one structure version (migration 141), the
 * delta of the NEWEST version against the one before it is rendered under the
 * graph, so an operator sees what moved since the last sign-off. A single
 * version has nothing to diff against and renders the graph alone.
 */
import { CausalDAG } from '@/components/visualizations/causal/CausalDAG';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type { DagStructure, ReviewVersion } from '@/types/expert-review';
import { DagDiff } from './DagDiff';

export function DagPanel({
  structure,
  versions,
}: {
  structure?: DagStructure | null;
  versions?: ReviewVersion[] | null;
}) {
  // `versions` is OLDEST first; the last entry carries the newest delta. Its
  // `changes` is null on a review minted before the versions table.
  const latest = versions && versions.length > 1 ? versions[versions.length - 1] : null;
  const latestChanges = latest?.changes ?? null;

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
      {latestChanges && (
        <div className="space-y-1">
          <h4 className="text-sm font-medium">Changed since the previous version</h4>
          <DagDiff changes={latestChanges} />
        </div>
      )}
    </div>
  );
}
