/**
 * Knowledge Graph Page
 * ====================
 *
 * Main page component for the Knowledge Graph visualization.
 * Displays the interactive graph with nodes and relationships
 * from the E2I Causal Analytics system.
 *
 * Uses TanStack Query hooks to fetch data from the graph API
 * with automatic caching, loading states, and error handling.
 *
 * @module pages/KnowledgeGraph
 */

import { useState, useMemo, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { KnowledgeGraph as KnowledgeGraphViz } from '@/components/visualizations/KnowledgeGraph';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useNodes, useRelationships } from '@/hooks/api/use-graph';
import type { GraphNode, GraphRelationship } from '@/types/graph';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * The pharma brands whose synthetic gold-standard causal graphs are loaded.
 * Each ``(:Variable)-[:CAUSES {brand}]->(:Variable)`` edge is stamped with its
 * brand by ``scripts/sync_causal_paths_to_falkordb.py``, so the page renders
 * exactly ONE brand's causal chains at a time (selected via the dropdown).
 * Matching is case-insensitive — the graph carries both ``Kisqali`` and a
 * legacy lowercase ``kisqali`` (and ``Remibrutinib``/``remibrutinib``).
 */
const BRANDS = ['Kisqali', 'Fabhalta', 'Remibrutinib'] as const;
type Brand = (typeof BRANDS)[number];

/**
 * Pull the full (scoped) graph in one window so connected-component detection
 * and the rendered stats are computed over the complete data, not an arbitrary
 * page. The backend caps these at 2000; today's graph is ~640 nodes / ~610 edges.
 */
const NODE_FETCH_LIMIT = 2000;
const REL_FETCH_LIMIT = 2000;

/**
 * Minimum connected-component size to render WITHIN a brand's causal graph. A
 * brand's gold standard can include a tiny off-chain causal pair; isolated
 * singletons and 2-node pairs carry little signal and read as a stray
 * "disconnected secondary graph", so they are pruned. K = 3 keeps every genuine
 * multi-node chain. Tunable.
 */
const MIN_COMPONENT_SIZE = 3;

/**
 * Derive the selected brand's causal subgraph: keep the ``CAUSES`` edges tagged
 * with that brand (case-insensitive) and the ``Variable`` nodes they connect,
 * THEN drop within-brand singletons and 2-node pairs (connected components
 * smaller than ``MIN_COMPONENT_SIZE``) so the canvas shows the brand's connected
 * causal chain(s), not stray off-chain fragments. This IS the brand's synthetic
 * gold-standard causal graph — no other entity types, no cross-brand edges.
 */
function causalSubgraphForBrand(
  nodes: GraphNode[],
  relationships: GraphRelationship[],
  brand: string
): { nodes: GraphNode[]; relationships: GraphRelationship[] } {
  const target = brand.toLowerCase();
  const brandRels = relationships.filter((r) => {
    const b = r.properties?.brand;
    return r.type === 'CAUSES' && typeof b === 'string' && b.toLowerCase() === target;
  });

  // Undirected adjacency over the brand's causal Variables.
  const adjacency = new Map<string, string[]>();
  const touch = (id: string) => {
    if (!adjacency.has(id)) adjacency.set(id, []);
  };
  for (const r of brandRels) {
    touch(r.source_id);
    touch(r.target_id);
    adjacency.get(r.source_id)!.push(r.target_id);
    adjacency.get(r.target_id)!.push(r.source_id);
  }

  // BFS each connected component; keep only those with >= MIN_COMPONENT_SIZE nodes.
  const seen = new Set<string>();
  const keep = new Set<string>();
  for (const start of adjacency.keys()) {
    if (seen.has(start)) continue;
    seen.add(start);
    const component = [start];
    const queue = [start];
    while (queue.length > 0) {
      const current = queue.pop()!;
      for (const neighbor of adjacency.get(current)!) {
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          component.push(neighbor);
          queue.push(neighbor);
        }
      }
    }
    if (component.length >= MIN_COMPONENT_SIZE) {
      for (const id of component) keep.add(id);
    }
  }

  return {
    nodes: nodes.filter((n) => keep.has(n.id)),
    relationships: brandRels.filter((r) => keep.has(r.source_id) && keep.has(r.target_id)),
  };
}


// =============================================================================
// PAGE COMPONENT
// =============================================================================

function KnowledgeGraphPage() {
  // State for selected elements
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphRelationship | null>(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');

  // Selected brand — the page loads exactly this brand's synthetic gold-standard
  // causal graph (its brand-tagged CAUSES chains). Defaults to the first brand.
  const [selectedBrand, setSelectedBrand] = useState<Brand>('Kisqali');

  // Fetch the causal layer's nodes: Variable entities only. The full window is
  // pulled so every brand's chains are present and switching brands needs no
  // refetch (the brand subgraph is derived client-side below).
  const {
    data: nodesData,
    isLoading: isLoadingNodes,
    error: nodesError,
    refetch: refetchNodes,
  } = useNodes({
    entity_types: 'Variable',
    limit: NODE_FETCH_LIMIT,
  });

  // Fetch the brand-tagged CAUSES edges (the synthetic gold-standard causal
  // chains). The selected brand's subgraph is derived from these below.
  const {
    data: relationshipsData,
    isLoading: isLoadingRelationships,
    error: relationshipsError,
    refetch: refetchRelationships,
  } = useRelationships({ relationship_types: 'CAUSES', limit: REL_FETCH_LIMIT });

  // Combined loading state
  const isLoading = isLoadingNodes || isLoadingRelationships;

  // Combined error state (prioritize nodes error, then relationships error)
  const error = nodesError || relationshipsError;

  // Retry handler for error state
  const handleRetry = useCallback(() => {
    if (nodesError) {
      void refetchNodes();
    }
    if (relationshipsError) {
      void refetchRelationships();
    }
  }, [nodesError, relationshipsError, refetchNodes, refetchRelationships]);

  // Extract nodes and relationships from API response (memoized to prevent unnecessary re-renders)
  const allNodes = useMemo(() => nodesData?.nodes ?? [], [nodesData?.nodes]);
  const allRelationships = useMemo(
    () => relationshipsData?.relationships ?? [],
    [relationshipsData?.relationships]
  );

  // Base rendered graph: the selected brand's synthetic gold-standard causal
  // subgraph (brand-tagged CAUSES chains + their Variables), independent of the
  // search query. Switching brands re-derives without a refetch.
  const brandGraph = useMemo(
    () => causalSubgraphForBrand(allNodes, allRelationships, selectedBrand),
    [allNodes, allRelationships, selectedBrand]
  );

  // Filter the brand graph by the search query (matches name or type).
  const filteredNodes = useMemo(() => {
    if (!searchQuery.trim()) return brandGraph.nodes;
    const query = searchQuery.toLowerCase();
    return brandGraph.nodes.filter(
      (node) =>
        node.name.toLowerCase().includes(query) ||
        node.type.toLowerCase().includes(query)
    );
  }, [brandGraph.nodes, searchQuery]);

  // Keep only edges whose endpoints are both in the (possibly searched) node set.
  const filteredRelationships = useMemo(() => {
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    return brandGraph.relationships.filter(
      (rel) => nodeIds.has(rel.source_id) && nodeIds.has(rel.target_id)
    );
  }, [brandGraph.relationships, filteredNodes]);

  // Use filtered data for display
  const nodes = filteredNodes;
  const relationships = filteredRelationships;

  // Stats reflect EXACTLY what is rendered (after scope + connectivity + search),
  // computed from the loaded data. The global /graph/stats endpoint is NOT used
  // here: it sums only the legacy enum types (a severe undercount) and would not
  // match the scoped, connected view shown on the canvas.
  const stats = useMemo(() => {
    const nodesByType = nodes.reduce(
      (acc, node) => {
        acc[node.type] = (acc[node.type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    return {
      totalNodes: nodes.length,
      totalRelationships: relationships.length,
      nodesByType,
    };
  }, [nodes, relationships]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Knowledge Graph</h1>
        <p className="text-[var(--color-muted-foreground)]">
          Explore the knowledge graph visualization with interactive nodes and edges.
          Click on nodes to see details, drag to pan, and scroll to zoom.
        </p>
      </div>

      {/* Search and Legend Row */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        {/* Search Input */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--color-muted-foreground)]" />
          <Input
            placeholder="Search nodes by name or type..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-10"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="sm"
              className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7 p-0"
              onClick={() => setSearchQuery('')}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Search Results Info */}
        {searchQuery && (
          <div className="flex items-center text-sm text-[var(--color-muted-foreground)]">
            Found {filteredNodes.length} nodes, {filteredRelationships.length} relationships
          </div>
        )}

        {/* Brand selector: load this brand's synthetic gold-standard causal graph */}
        <div className="flex items-center gap-2 md:ml-auto">
          <label htmlFor="kg-brand" className="text-sm text-[var(--color-muted-foreground)]">
            Brand
          </label>
          <select
            id="kg-brand"
            aria-label="Brand"
            value={selectedBrand}
            onChange={(e) => setSelectedBrand(e.target.value as Brand)}
            className="h-9 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 text-sm"
          >
            {BRANDS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Scope hint */}
      <p className="text-xs text-[var(--color-muted-foreground)] mb-4">
        Showing{' '}
        <span className="font-medium text-[var(--color-foreground)]">{selectedBrand}</span>
        &apos;s synthetic gold-standard causal graph — its variables and their
        brand-specific cause→effect chains. Select another brand to load its graph.
      </p>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Nodes</CardDescription>
            <CardTitle className="text-2xl">
              {isLoading ? (
                <span className="inline-block h-8 w-16 animate-pulse rounded bg-[var(--color-muted)]" />
              ) : (
                stats.totalNodes
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex gap-1">
                <span className="h-5 w-16 animate-pulse rounded bg-[var(--color-muted)]" />
                <span className="h-5 w-16 animate-pulse rounded bg-[var(--color-muted)]" />
              </div>
            ) : (
              <div className="flex flex-wrap gap-1">
                {Object.entries(stats.nodesByType).map(([type, count]) => (
                  <Badge key={type} variant="secondary" className="text-xs">
                    {type}: {count}
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Relationships</CardDescription>
            <CardTitle className="text-2xl">
              {isLoading ? (
                <span className="inline-block h-8 w-16 animate-pulse rounded bg-[var(--color-muted)]" />
              ) : (
                stats.totalRelationships
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Connections between entities
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Selected</CardDescription>
            <CardTitle className="text-lg truncate">
              {selectedNode?.name || selectedEdge?.type || 'None'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {selectedNode
                ? `Type: ${selectedNode.type}`
                : selectedEdge
                  ? `Confidence: ${((selectedEdge.confidence ?? 0) * 100).toFixed(0)}%`
                  : 'Click a node or edge'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Graph Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Graph Visualization</CardTitle>
          <CardDescription>
            Interactive knowledge graph showing entities and their relationships
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <KnowledgeGraphViz
            nodes={nodes}
            relationships={relationships}
            layout="cose"
            minHeight={500}
            isLoading={isLoading}
            error={error}
            onRetry={handleRetry}
            onNodeSelect={(node) => {
              setSelectedNode(node);
              setSelectedEdge(null);
            }}
            onEdgeSelect={(edge) => {
              setSelectedEdge(edge);
              setSelectedNode(null);
            }}
            className="rounded-b-lg"
          />
        </CardContent>
      </Card>

      {/* Selected Node/Edge Details */}
      {(selectedNode || selectedEdge) && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>
              {selectedNode ? 'Node Details' : 'Relationship Details'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedNode && (
              <dl className="grid grid-cols-2 gap-4">
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">ID</dt>
                  <dd className="text-sm">{selectedNode.id}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Name</dt>
                  <dd className="text-sm">{selectedNode.name}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Type</dt>
                  <dd className="text-sm">
                    <Badge variant="outline">{selectedNode.type}</Badge>
                  </dd>
                </div>
                {selectedNode.created_at && (
                  <div>
                    <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Created</dt>
                    <dd className="text-sm">
                      {new Date(selectedNode.created_at).toLocaleDateString()}
                    </dd>
                  </div>
                )}
                {Object.keys(selectedNode.properties).length > 0 && (
                  <div className="col-span-2">
                    <dt className="text-sm font-medium text-[var(--color-muted-foreground)] mb-1">
                      Properties
                    </dt>
                    <dd className="text-sm">
                      <pre className="bg-[var(--color-muted)]/20 p-2 rounded text-xs overflow-auto">
                        {JSON.stringify(selectedNode.properties, null, 2)}
                      </pre>
                    </dd>
                  </div>
                )}
              </dl>
            )}

            {selectedEdge && (
              <dl className="grid grid-cols-2 gap-4">
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">ID</dt>
                  <dd className="text-sm">{selectedEdge.id}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Type</dt>
                  <dd className="text-sm">
                    <Badge variant="outline">{selectedEdge.type}</Badge>
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Source</dt>
                  <dd className="text-sm">{selectedEdge.source_id}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Target</dt>
                  <dd className="text-sm">{selectedEdge.target_id}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Confidence</dt>
                  <dd className="text-sm">
                    {selectedEdge.confidence !== undefined
                      ? `${(selectedEdge.confidence * 100).toFixed(1)}%`
                      : 'N/A'}
                  </dd>
                </div>
                {selectedEdge.created_at && (
                  <div>
                    <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">Created</dt>
                    <dd className="text-sm">
                      {new Date(selectedEdge.created_at).toLocaleDateString()}
                    </dd>
                  </div>
                )}
              </dl>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default KnowledgeGraphPage;
