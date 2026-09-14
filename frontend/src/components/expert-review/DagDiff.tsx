/**
 * The structural delta between two versions of one estimand (#1991 debt 3).
 *
 * Renders WHAT moved, not merely that the hash changed: added/removed nodes,
 * edges and adjustment sets. An unchanged delta says so explicitly rather than
 * rendering nothing — a silent blank would read as "not computed". A group with
 * no entries is omitted entirely so an empty heading never implies a change.
 */
import type { DagChanges } from '@/types/expert-review';

/**
 * Edges and adjustment sets arrive as `string[][]` (backend List[List[str]]).
 *
 * The rendered glyphs are NON-ASCII and a test matching on them must use the
 * same code points: the edge arrow is U+2192 RIGHTWARDS ARROW (→, not "->")
 * and a removal is marked with U+2212 MINUS SIGN (−, not the ASCII hyphen).
 */
const edge = ([s, t]: string[]) => `${s} → ${t}`;
const set = (a: string[]) => `{${a.join(', ')}}`;

export function DagDiff({ changes }: { changes: DagChanges }) {
  if (!changes.is_changed) {
    return (
      <p className="text-xs text-[var(--color-muted-foreground)]">
        No structural change from the previous version.
      </p>
    );
  }

  const rows: Array<[string, string[]]> = [
    [
      'Nodes',
      [
        ...changes.nodes_added.map((n) => `+ ${n}`),
        ...changes.nodes_removed.map((n) => `− ${n}`),
      ],
    ],
    [
      'Edges',
      [
        ...changes.edges_added.map((e) => `+ ${edge(e)}`),
        ...changes.edges_removed.map((e) => `− ${edge(e)}`),
      ],
    ],
    [
      'Adjustment sets',
      [
        ...changes.adjustment_sets_added.map((a) => `+ ${set(a)}`),
        ...changes.adjustment_sets_removed.map((a) => `− ${set(a)}`),
      ],
    ],
  ];

  return (
    <dl className="grid gap-1 text-xs">
      {rows
        .filter(([, items]) => items.length > 0)
        .map(([label, items]) => (
          <div key={label}>
            <dt className="font-medium">{label}</dt>
            {items.map((item) => (
              <dd key={item} className="ml-2 font-mono">
                {item}
              </dd>
            ))}
          </div>
        ))}
    </dl>
  );
}
