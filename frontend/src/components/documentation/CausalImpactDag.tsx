/**
 * CausalImpactDag — the CausalImpactAgent's "Acceptance → Action → Multi-path
 * Revenue Impact" DAG, ported from docs/Archive/CausalImpactAgent.html.
 * Nodes are color-coded by variable role (shared key: CAUSAL_VARIABLE_TYPES);
 * a path toggle spotlights one causal pathway in red with bold arrowheads and
 * dims the rest. Edge labels are the explainer's hand-authored figures, so the
 * figure carries the page's "illustrative example" badge — nothing here is a
 * measured effect.
 */
import { useState } from 'react';
import {
  CAUSAL_VARIABLE_TYPES,
  DAG_EDGES,
  DAG_NODES,
  DAG_PATHS,
  DAG_TITLE,
  type DagEdge,
  type DagNode,
  type DagPathGroup,
} from './content';

const NODE_R = 26;
const EDGE_GAP = NODE_R + 3;
const HIGHLIGHT = '#ef4444';
const COLOR_BY_TYPE = Object.fromEntries(
  CAUSAL_VARIABLE_TYPES.map((v) => [v.type, v.color])
) as Record<DagNode['type'], string>;
const NODE_BY_ID = new Map(DAG_NODES.map((n) => [n.id, n]));

// The reinforcing loop (revenue → feedback) is the one edge that cannot be a
// straight line — it would cross the bottom row of nodes. It is routed along
// the lower margin, leaving revenue on its right side and entering feedback
// from its lower-left, clear of every node circle and label.
const LOOP_PATH = 'M 966 160 C 1075 185 1075 372 900 372 C 720 376 400 392 515.6 289';
const LOOP_LABEL = { x: 740, y: 362 };

interface EdgeGeometry {
  d: string;
  lx: number;
  ly: number;
  anchor: 'middle' | 'start';
}

function edgeGeometry(edge: DagEdge): EdgeGeometry {
  const a = NODE_BY_ID.get(edge.from);
  const b = NODE_BY_ID.get(edge.to);
  if (!a || !b) throw new Error(`DAG edge references an unknown node: ${edge.from} → ${edge.to}`);
  if (edge.group === 'loop') {
    return { d: LOOP_PATH, lx: LOOP_LABEL.x, ly: LOOP_LABEL.y, anchor: 'middle' };
  }
  // Trim both ends to the node boundary so the arrowhead is visible instead of
  // buried under the target node.
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const x1 = a.x + ux * EDGE_GAP;
  const y1 = a.y + uy * EDGE_GAP;
  const x2 = b.x - ux * EDGE_GAP;
  const y2 = b.y - uy * EDGE_GAP;
  const d = `M ${x1.toFixed(1)} ${y1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  const t = edge.labelT ?? 0.5;
  const mx = a.x + dx * t;
  const my = a.y + dy * t;
  // A caption centred on a vertical edge would sit on the line; hang it off
  // the right side instead. Diagonal/horizontal captions float just above.
  if (Math.abs(dx) < 1) return { d, lx: mx + 10, ly: my + 4, anchor: 'start' };
  return { d, lx: mx, ly: my - 10, anchor: 'middle' };
}

const EDGE_GEOMETRY = DAG_EDGES.map(edgeGeometry);

function NodeShape({ node, selected }: { node: DagNode; selected: boolean }) {
  const stroke = selected ? HIGHLIGHT : COLOR_BY_TYPE[node.type];
  const strokeWidth = selected ? 4 : 3;
  return (
    <g data-node={node.id} data-selected={selected ? 'true' : 'false'}>
      {node.type === 'confounder' ? (
        <polygon
          points={`${node.x},${node.y - NODE_R} ${node.x + NODE_R},${node.y} ${node.x},${node.y + NODE_R} ${node.x - NODE_R},${node.y}`}
          className="fill-[var(--color-card)]"
          stroke={stroke}
          strokeWidth={strokeWidth}
        />
      ) : (
        <circle
          cx={node.x}
          cy={node.y}
          r={NODE_R}
          className="fill-[var(--color-card)]"
          stroke={stroke}
          strokeWidth={strokeWidth}
        />
      )}
      <text
        x={node.x}
        y={node.labelAbove ? node.y - 34 : node.y + 44}
        textAnchor="middle"
        fontSize="12"
        fontWeight="500"
        className="fill-[var(--color-foreground)] [paint-order:stroke] [stroke:var(--color-card)] [stroke-width:3px]"
      >
        {node.label}
      </text>
    </g>
  );
}

function PathButton({
  label,
  pressed,
  onClick,
}: {
  label: string;
  pressed: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors motion-reduce:transition-none ${
        pressed
          ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
          : 'border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-foreground)] hover:border-[var(--color-primary)]/50'
      }`}
    >
      {label}
    </button>
  );
}

export function CausalImpactDag() {
  const [highlight, setHighlight] = useState<DagPathGroup | null>(null);
  const activePath = highlight ? DAG_PATHS.find((p) => p.group === highlight) : undefined;

  const selectedNodes = new Set<string>();
  if (highlight) {
    for (const e of DAG_EDGES) {
      if (e.group === highlight) {
        selectedNodes.add(e.from);
        selectedNodes.add(e.to);
      }
    }
  }

  return (
    <figure
      aria-labelledby="causal-impact-dag-title"
      className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4"
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3
          id="causal-impact-dag-title"
          className="text-sm font-semibold text-[var(--color-foreground)]"
        >
          {DAG_TITLE}
        </h3>
        <span className="rounded-full border border-amber-500/50 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-600 dark:text-amber-400">
          Illustrative example
        </span>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--color-muted-foreground)]">
        {CAUSAL_VARIABLE_TYPES.map((v) => (
          <span key={v.type} className="inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className={`inline-block h-2.5 w-2.5 ${v.shape === 'diamond' ? 'rotate-45' : 'rounded-full'}`}
              style={{ backgroundColor: v.color }}
            />
            {v.label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: HIGHLIGHT }}
          />
          Highlighted path
        </span>
        <span>Bold arrowheads indicate direction.</span>
      </div>

      <div role="group" aria-label="Spotlight a causal pathway" className="mb-3 flex flex-wrap gap-2">
        <PathButton label="All paths" pressed={highlight === null} onClick={() => setHighlight(null)} />
        {DAG_PATHS.map((p) => (
          <PathButton
            key={p.group}
            label={p.label}
            pressed={highlight === p.group}
            onClick={() => setHighlight((cur) => (cur === p.group ? null : p.group))}
          />
        ))}
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox="0 0 1100 400"
          role="img"
          aria-label={`Illustrative causal DAG: ${DAG_TITLE}. ${
            activePath ? `Highlighted: ${activePath.label}.` : 'All paths shown.'
          }`}
          className="h-auto w-full min-w-[720px]"
        >
          <defs>
            <marker
              id="causal-dag-arrow"
              viewBox="0 0 24 24"
              refX="24"
              refY="12"
              markerWidth="12"
              markerHeight="12"
              markerUnits="userSpaceOnUse"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L24,12 L0,24 z" className="fill-[var(--color-foreground)]" />
            </marker>
            <marker
              id="causal-dag-arrow-highlight"
              viewBox="0 0 24 24"
              refX="24"
              refY="12"
              markerWidth="14"
              markerHeight="14"
              markerUnits="userSpaceOnUse"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L24,12 L0,24 z" fill={HIGHLIGHT} />
            </marker>
          </defs>

          {DAG_EDGES.map((edge, i) => {
            const geo = EDGE_GEOMETRY[i];
            const selected = highlight !== null && edge.group === highlight;
            const dimmed = highlight !== null && !selected;
            return (
              <g
                key={`${edge.from}-${edge.to}`}
                data-edge={`${edge.from}|${edge.to}`}
                data-group={edge.group}
                data-selected={selected ? 'true' : 'false'}
                className={dimmed ? 'opacity-30' : undefined}
              >
                <path
                  d={geo.d}
                  fill="none"
                  stroke={selected ? HIGHLIGHT : undefined}
                  className={selected ? undefined : 'stroke-[var(--color-foreground)]'}
                  strokeWidth={selected ? 3 : 2}
                  strokeDasharray={edge.dashed ? '6,6' : undefined}
                  markerEnd={selected ? 'url(#causal-dag-arrow-highlight)' : 'url(#causal-dag-arrow)'}
                />
                {edge.label ? (
                  <text
                    x={geo.lx}
                    y={geo.ly}
                    textAnchor={geo.anchor}
                    fontSize="11"
                    fill={selected ? HIGHLIGHT : undefined}
                    className={`[paint-order:stroke] [stroke:var(--color-card)] [stroke-width:3px] ${
                      selected ? 'font-semibold' : 'fill-[var(--color-muted-foreground)]'
                    }`}
                  >
                    {edge.label}
                  </text>
                ) : null}
              </g>
            );
          })}

          {DAG_NODES.map((node) => (
            <NodeShape key={node.id} node={node} selected={selectedNodes.has(node.id)} />
          ))}
        </svg>
      </div>

      <figcaption className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        {activePath
          ? activePath.description
          : 'Tip: toggle a path to see red, bold arrows emphasizing directionality along that lifeline.'}
      </figcaption>
    </figure>
  );
}
