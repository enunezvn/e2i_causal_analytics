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
 * Node + relationship types that make up the synthetic gold-standard CAUSAL
 * graph. Variables carry the brand-tagged ``CAUSES`` chains; KPIs / CausalPaths /
 * Regions carry the cross-type causal structure (``CAUSES`` between KPIs,
 * ``EXPLAINS`` from a CausalPath, ``INFLUENCES`` a Region, ``AFFECTS``). We pull
 * these together so the page shows the WHOLE causal gold standard, not one
 * brand's variable slice. (Treatment exists but has no causal edge in the data,
 * so it drops out as isolated — shown only if it ever gains a causal link.)
 */
const CAUSAL_NODE_TYPES = 'Variable,KPI,CausalPath,Region,Treatment';
const CAUSAL_REL_TYPES = 'CAUSES,EXPLAINS,INFLUENCES,AFFECTS';

/**
 * Brand-tagged ``(:Variable)-[:CAUSES {brand}]->(:Variable)`` chains are stamped
 * by ``scripts/sync_causal_paths_to_falkordb.py``. The brand filter is OPTIONAL:
 * default ``All`` shows every brand's chains + the shared causal structure.
 * Matching is case-insensitive (the graph carries ``Kisqali`` and legacy
 * lowercase ``kisqali``, etc.).
 */
const BRANDS = ['Kisqali', 'Fabhalta', 'Remibrutinib'] as const;
const BRAND_FILTERS = ['All', ...BRANDS] as const;
type BrandFilter = (typeof BRAND_FILTERS)[number];

/**
 * Pull the full graph in one window so the rendered stats are computed over the
 * complete causal layer. The backend caps these at 2000.
 */
const NODE_FETCH_LIMIT = 2000;
const REL_FETCH_LIMIT = 2000;

/**
 * Derive the causal gold-standard graph to render. Keep:
 *  - every causal edge when ``brand === 'All'``;
 *  - otherwise the selected brand's tagged ``CAUSES`` edges PLUS the untagged,
 *    brand-agnostic structural edges (KPI↔KPI causes, CausalPath EXPLAINS KPI,
 *    KPI INFLUENCES Region, AFFECTS) so the brand's variables show in context.
 * Then keep only nodes touched by ≥1 kept edge (drop isolated singletons), and
 * only edges whose endpoints both survive. No cross-brand variable chains leak
 * into a single-brand view; ``All`` shows the complete causal structure.
 */
function causalGoldStandardGraph(
  nodes: GraphNode[],
  relationships: GraphRelationship[],
  brand: BrandFilter
): { nodes: GraphNode[]; relationships: GraphRelationship[] } {
  const target = brand.toLowerCase();
  const kept = relationships.filter((r) => {
    if (brand === 'All') return true;
    const b = r.properties?.brand;
    // Brand-tagged edge: keep only this brand's. Untagged structural edge: keep.
    return typeof b === 'string' ? b.toLowerCase() === target : true;
  });

  const touched = new Set<string>();
  for (const r of kept) {
    touched.add(r.source_id);
    touched.add(r.target_id);
  }
  const keptNodes = nodes.filter((n) => touched.has(n.id));
  const nodeIds = new Set(keptNodes.map((n) => n.id));
  return {
    nodes: keptNodes,
    relationships: kept.filter((r) => nodeIds.has(r.source_id) && nodeIds.has(r.target_id)),
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

  // Optional brand filter over the causal gold-standard graph. Default 'All' —
  // the page shows the WHOLE causal layer; a brand narrows to its variable chains
  // (+ the shared structure). Switching needs no refetch (derived client-side).
  const [selectedBrand, setSelectedBrand] = useState<BrandFilter>('All');

  // Fetch the causal layer's nodes (Variable + KPI + CausalPath + Region +
  // Treatment) in one full window so the whole gold-standard causal graph is
  // available and the brand filter is derived client-side.
  const {
    data: nodesData,
    isLoading: isLoadingNodes,
    error: nodesError,
    refetch: refetchNodes,
  } = useNodes({
    entity_types: CAUSAL_NODE_TYPES,
    limit: NODE_FETCH_LIMIT,
  });

  // Fetch the brand-tagged CAUSES edges (the synthetic gold-standard causal
  // chains). The selected brand's subgraph is derived from these below.
  const {
    data: relationshipsData,
    isLoading: isLoadingRelationships,
    error: relationshipsError,
    refetch: refetchRelationships,
  } = useRelationships({ relationship_types: CAUSAL_REL_TYPES, limit: REL_FETCH_LIMIT });

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

  // Base rendered graph: the causal gold-standard graph (all causal node types +
  // edges), optionally narrowed to one brand. Independent of the search query;
  // switching the brand filter re-derives without a refetch.
  const brandGraph = useMemo(
    () => causalGoldStandardGraph(allNodes, allRelationships, selectedBrand),
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
            onChange={(e) => setSelectedBrand(e.target.value as BrandFilter)}
            className="h-9 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 text-sm"
          >
            {BRAND_FILTERS.map((b) => (
              <option key={b} value={b}>
                {b === 'All' ? 'All brands' : b}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Scope hint */}
      <p className="text-xs text-[var(--color-muted-foreground)] mb-4">
        Showing the synthetic gold-standard{' '}
        <span className="font-medium text-[var(--color-foreground)]">causal graph</span> —{' '}
        {selectedBrand === 'All' ? (
          <>all brands&apos; variable cause→effect chains plus the shared KPI / causal-path structure.</>
        ) : (
          <>
            <span className="font-medium text-[var(--color-foreground)]">{selectedBrand}</span>
            &apos;s variable cause→effect chains plus the shared KPI / causal-path structure.
          </>
        )}{' '}
        Use the brand filter to narrow to one brand.
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
