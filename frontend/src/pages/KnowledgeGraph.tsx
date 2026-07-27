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

import { useState, useMemo, useCallback, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { KnowledgeGraph as KnowledgeGraphViz } from '@/components/visualizations/KnowledgeGraph';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useNodes, useRelationships } from '@/hooks/api/use-graph';
import { StrategicInsightCard } from '@/components/insights';
import { useKnowledgeGraphInsight } from '@/hooks/api';
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
 * brand's variable slice.
 *
 * ``Treatment`` is intentionally EXCLUDED from the fetch: brand product/regimen
 * nodes (Kisqali_Maintenance, Fabhalta_Therapy, …) carry NO causal edge
 * (CAUSES/EXPLAINS/INFLUENCES/AFFECTS), so they only ever rendered as isolated
 * singletons (and were dropped by the connectivity filter anyway). Not fetching
 * them keeps the payload and the rendered legend clean.
 */
const CAUSAL_NODE_TYPES = 'Variable,KPI,CausalPath,Region';
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
 * Collapse parallel edges into one logical edge.
 *
 * The synthetic gold standard stamps the SAME logical relationship once per
 * ``(brand × region)`` (``sync_causal_paths_to_falkordb.py`` MERGEs on
 * ``{brand, region}``), so a single ``a -CAUSES-> b`` can appear up to
 * 3 brands × 4 regions = 12 times. Rendered verbatim that is an unreadable
 * hairball. We key by ``source|type|target`` (so direction and edge type stay
 * distinct) and keep ONE representative — the highest-confidence physical copy —
 * surfacing the union of ``brands`` / ``regions`` and a ``parallel_edge_count``
 * on its properties so the detail panel stays honest about what was merged.
 */
function dedupeParallelEdges(relationships: GraphRelationship[]): GraphRelationship[] {
  const byKey = new Map<
    string,
    { edge: GraphRelationship; brands: Set<string>; regions: Set<string>; count: number }
  >();
  for (const r of relationships) {
    const key = `${r.source_id}\u0000${r.type}\u0000${r.target_id}`;
    let entry = byKey.get(key);
    if (!entry) {
      entry = { edge: r, brands: new Set<string>(), regions: new Set<string>(), count: 0 };
      byKey.set(key, entry);
    }
    entry.count += 1;
    const brand = r.properties?.brand;
    const region = r.properties?.region;
    if (typeof brand === 'string') entry.brands.add(brand);
    if (typeof region === 'string') entry.regions.add(region);
    // Keep the highest-confidence physical copy as the representative.
    if ((r.confidence ?? 0) > (entry.edge.confidence ?? 0)) entry.edge = r;
  }
  return [...byKey.values()].map(({ edge, brands, regions, count }) => {
    // Replace the per-instance brand/region with the merged aggregate.
    const properties: Record<string, unknown> = { ...edge.properties };
    delete properties.brand;
    delete properties.region;
    if (brands.size > 0) properties.brands = [...brands].sort();
    if (regions.size > 0) properties.regions = [...regions].sort();
    if (count > 1) properties.parallel_edge_count = count;
    return { ...edge, properties };
  });
}

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
  const survivingEdges = kept.filter(
    (r) => nodeIds.has(r.source_id) && nodeIds.has(r.target_id)
  );
  return {
    nodes: keptNodes,
    // Collapse parallel (brand×region) copies of each logical edge so the gold
    // standard renders as a readable DAG instead of a hairball (the sync stamps
    // one edge per brand×region — up to 3×4 = 12 copies of the same pair).
    relationships: dedupeParallelEdges(survivingEdges),
  };
}

/**
 * Narrow the (brand-scoped) graph to one variable's causal neighborhood: the
 * variable itself plus every ancestor and descendant along ``CAUSES`` edges —
 * i.e. the full chains through it — plus the structural context (EXPLAINS /
 * INFLUENCES / AFFECTS edges touching those nodes, and their far endpoints).
 * Derived client-side from the already-loaded graph, so switching variables
 * needs no refetch and the option list always matches the active brand scope.
 */
function variableNeighborhoodGraph(
  graph: { nodes: GraphNode[]; relationships: GraphRelationship[] },
  variableId: string
): { nodes: GraphNode[]; relationships: GraphRelationship[] } {
  const fwd = new Map<string, string[]>();
  const rev = new Map<string, string[]>();
  const push = (m: Map<string, string[]>, k: string, v: string) => {
    const arr = m.get(k);
    if (arr) arr.push(v);
    else m.set(k, [v]);
  };
  for (const r of graph.relationships) {
    if (r.type !== 'CAUSES') continue;
    push(fwd, r.source_id, r.target_id);
    push(rev, r.target_id, r.source_id);
  }
  const reach = (start: string, adj: Map<string, string[]>): Set<string> => {
    const seen = new Set<string>([start]);
    const queue = [start];
    while (queue.length > 0) {
      const cur = queue.pop() as string;
      for (const nxt of adj.get(cur) ?? []) {
        if (!seen.has(nxt)) {
          seen.add(nxt);
          queue.push(nxt);
        }
      }
    }
    return seen;
  };
  const core = new Set([...reach(variableId, fwd), ...reach(variableId, rev)]);
  // Structural context: non-CAUSES edges touching the causal core bring in
  // their far endpoints (attached KPI / CausalPath / Region nodes).
  const contextIds = new Set<string>();
  for (const r of graph.relationships) {
    if (r.type === 'CAUSES') continue;
    if (core.has(r.source_id)) contextIds.add(r.target_id);
    else if (core.has(r.target_id)) contextIds.add(r.source_id);
  }
  const keep = new Set([...core, ...contextIds]);
  return {
    nodes: graph.nodes.filter((n) => keep.has(n.id)),
    relationships: graph.relationships.filter((r) =>
      r.type === 'CAUSES'
        ? core.has(r.source_id) && core.has(r.target_id)
        : (core.has(r.source_id) || core.has(r.target_id)) &&
          keep.has(r.source_id) &&
          keep.has(r.target_id)
    ),
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

  // Optional variable filter within the brand scope: 'All' shows every chain,
  // a variable node id narrows to the chains through that variable. The option
  // list derives from the brand-scoped graph, so it adapts with the brand.
  const [selectedVariable, setSelectedVariable] = useState<string>('All');

  // Agentic strategic interpretation of the (brand-scoped) causal graph. Lazy:
  // the card renders a "Generate strategic insight" button; the mutation fires
  // on demand with the current brand + the curated (gold-standard) scope.
  const kgInsight = useKnowledgeGraphInsight();
  // The brand the interpretation was SUBMITTED for. The response carries no
  // scope, so we tag the submitted brand ourselves and only surface a result
  // while it still matches the active selection — a run fired for brand A that
  // resolves AFTER the user switches to brand B (reset below cleared the old
  // data, but the late mutation repopulates kgInsight.data) is suppressed
  // instead of rendered — and mislabeled — under B. Mirrors CausalAnalysis's
  // causalInsightScope pattern.
  const [kgInsightScope, setKgInsightScope] = useState<BrandFilter | null>(null);
  const resetKgInsight = kgInsight.reset;
  // On a brand switch, drop the previous brand's interpretation so it never
  // lingers under the new scope (the stale-interpretation bug).
  useEffect(() => {
    resetKgInsight();
    setKgInsightScope(null);
  }, [selectedBrand, resetKgInsight]);
  const kgInsightInScope = kgInsightScope === selectedBrand;
  const kgInsightData = kgInsightInScope ? kgInsight.data : undefined;

  // Fetch the causal layer's nodes (Variable + KPI + CausalPath + Region +
  // Treatment) in one full window so the whole gold-standard causal graph is
  // available and the brand filter is derived client-side.
  //
  // ``curated_only`` excludes agent-written runtime nodes: the causal_impact
  // agent persists its analysis treatment/outcome Variables (tagged with an
  // ``agent`` property) into the SAME graph, which otherwise rendered as stray
  // disconnected pairs alongside the gold standard. Seed/sync gold-standard
  // nodes carry no ``agent`` tag, so the curated view shows only them — and it
  // stays clean no matter how many analyses the agent runs.
  const {
    data: nodesData,
    isLoading: isLoadingNodes,
    error: nodesError,
    refetch: refetchNodes,
  } = useNodes({
    entity_types: CAUSAL_NODE_TYPES,
    limit: NODE_FETCH_LIMIT,
    curated_only: true,
  });

  // Fetch the brand-tagged CAUSES edges (the synthetic gold-standard causal
  // chains). The selected brand's subgraph is derived from these below.
  const {
    data: relationshipsData,
    isLoading: isLoadingRelationships,
    error: relationshipsError,
    refetch: refetchRelationships,
  } = useRelationships({
    relationship_types: CAUSAL_REL_TYPES,
    limit: REL_FETCH_LIMIT,
    curated_only: true,
  });

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

  // Variable options for the selector: the brand graph's Variable nodes. Derived
  // from brandGraph (NOT the narrowed scope) so picking one never shrinks the
  // list to its own neighborhood.
  const variableOptions = useMemo(
    () =>
      brandGraph.nodes
        .filter((n) => n.type === 'Variable')
        .sort((a, b) => a.name.localeCompare(b.name)),
    [brandGraph.nodes]
  );
  const selectedVariableName =
    selectedVariable === 'All'
      ? null
      : (variableOptions.find((v) => v.id === selectedVariable)?.name ?? null);

  // A selected variable deliberately SURVIVES a brand switch (comparing the same
  // variable's chains across brands is the point of the selector); it only
  // clamps back to 'All' if the new brand scope no longer contains it.
  useEffect(() => {
    if (isLoading || selectedVariable === 'All') return;
    if (!brandGraph.nodes.some((n) => n.id === selectedVariable)) {
      setSelectedVariable('All');
    }
  }, [isLoading, selectedVariable, brandGraph.nodes]);

  // Narrow to the selected variable's causal neighborhood (chains through it +
  // attached structural context). 'All' renders the whole brand scope.
  const scopedGraph = useMemo(
    () =>
      selectedVariable === 'All'
        ? brandGraph
        : variableNeighborhoodGraph(brandGraph, selectedVariable),
    [brandGraph, selectedVariable]
  );

  // Filter the scoped graph by the search query (matches name or type).
  const filteredNodes = useMemo(() => {
    if (!searchQuery.trim()) return scopedGraph.nodes;
    const query = searchQuery.toLowerCase();
    return scopedGraph.nodes.filter(
      (node) =>
        node.name.toLowerCase().includes(query) ||
        node.type.toLowerCase().includes(query)
    );
  }, [scopedGraph.nodes, searchQuery]);

  // Keep only edges whose endpoints are both in the (possibly searched) node set.
  const filteredRelationships = useMemo(() => {
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    return scopedGraph.relationships.filter(
      (rel) => nodeIds.has(rel.source_id) && nodeIds.has(rel.target_id)
    );
  }, [scopedGraph.relationships, filteredNodes]);

  // Use filtered data for display
  const nodes = filteredNodes;
  const relationships = filteredRelationships;

  // The selected edge's ATE, when it carries one (brand-tagged CAUSES chains).
  const rawAte = selectedEdge?.properties?.ate_estimate;
  const selectedEdgeAte = typeof rawAte === 'number' ? rawAte : undefined;

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

        {/* Brand + variable selectors: scope the gold-standard causal graph */}
        <div className="flex flex-wrap items-center gap-4 md:ml-auto">
          <div className="flex items-center gap-2">
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
          {/* Variable selector: options derive from the brand-scoped graph, so
              they update with the brand; picking one narrows the canvas to the
              causal chains through that variable. */}
          <div className="flex items-center gap-2">
            <label htmlFor="kg-variable" className="text-sm text-[var(--color-muted-foreground)]">
              Variable
            </label>
            <select
              id="kg-variable"
              aria-label="Variable"
              value={selectedVariable}
              onChange={(e) => setSelectedVariable(e.target.value)}
              className="h-9 max-w-56 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 text-sm"
            >
              <option value="All">All variables</option>
              {variableOptions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Scope hint */}
      <div className="mb-4 space-y-1">
      <p className="text-xs text-[var(--color-muted-foreground)]">
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
        {selectedVariableName ? (
          <>
            Narrowed to the chains through{' '}
            <span className="font-medium text-[var(--color-foreground)]">{selectedVariableName}</span>.
          </>
        ) : (
          <>Use the brand and variable filters to narrow the view.</>
        )}
      </p>
      {/* Effect-styling legend: brands share the chain topology by design — the
          brand-specific signal is each edge's effect size / confidence, encoded
          visually when a single brand is selected. */}
      {selectedBrand !== 'All' && (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Edges reflect {selectedBrand}&apos;s estimates: width ∝ |effect size (ATE)|,{' '}
          <span
            className="inline-block h-2 w-2 rounded-full align-baseline"
            style={{ backgroundColor: '#059669' }}
            aria-hidden="true"
          />{' '}
          positive /{' '}
          <span
            className="inline-block h-2 w-2 rounded-full align-baseline"
            style={{ backgroundColor: '#e11d48' }}
            aria-hidden="true"
          />{' '}
          negative effect, opacity = confidence.
        </p>
      )}
      </div>

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

      {/* Strategic Interpretation (agentic insight over the brand-scoped graph).
          Everything rendered is gated on kgInsightInScope so a brand switch
          returns the card to its Generate state instead of showing the previous
          brand's interpretation (see the scope-tagging comment above). */}
      <div className="mb-6">
        <StrategicInsightCard
          isLoading={kgInsightInScope && kgInsight.isPending}
          error={kgInsightInScope ? (kgInsight.error?.message ?? null) : null}
          insight={kgInsightData?.insight}
          keyTakeaways={kgInsightData?.key_takeaways}
          grounding={kgInsightData?.grounding}
          isFallback={kgInsightData?.is_fallback}
          provenance={kgInsightData?.provenance}
          generatedAt={kgInsightData?.generated_at}
          onGenerate={() => {
            setKgInsightScope(selectedBrand);
            kgInsight.mutate({ brand: selectedBrand, curated_only: true });
          }}
        />
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
            styleEdgesByEffect={selectedBrand !== 'All'}
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
                {selectedEdgeAte !== undefined && (
                  <div>
                    <dt className="text-sm font-medium text-[var(--color-muted-foreground)]">
                      Effect size (ATE)
                    </dt>
                    <dd className="text-sm">{selectedEdgeAte.toFixed(3)}</dd>
                  </div>
                )}
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
