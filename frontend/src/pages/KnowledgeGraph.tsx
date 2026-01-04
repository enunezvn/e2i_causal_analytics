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
import { useNodes, useRelationships, useGraphStats } from '@/hooks/api/use-graph';
import type { GraphNode, GraphRelationship } from '@/types/graph';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Entity type color mapping for the legend
 * Must match the colors in KnowledgeGraph visualization component
 */
const ENTITY_TYPE_COLORS: Record<string, string> = {
  Patient: '#3b82f6', // blue-500
  HCP: '#10b981', // emerald-500
  Brand: '#f59e0b', // amber-500
  Region: '#8b5cf6', // violet-500
  KPI: '#ef4444', // red-500
  CausalPath: '#06b6d4', // cyan-500
  Trigger: '#f97316', // orange-500
  Agent: '#ec4899', // pink-500
  Episode: '#6366f1', // indigo-500
  Community: '#14b8a6', // teal-500
  Treatment: '#84cc16', // lime-500
  Prediction: '#a855f7', // purple-500
  Experiment: '#22c55e', // green-500
  AgentActivity: '#64748b', // slate-500
};


// =============================================================================
// PAGE COMPONENT
// =============================================================================

function KnowledgeGraphPage() {
  // State for selected elements
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphRelationship | null>(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch nodes from API with pagination (get first 100 nodes)
  const {
    data: nodesData,
    isLoading: isLoadingNodes,
    error: nodesError,
    refetch: refetchNodes,
  } = useNodes({ limit: 100 });

  // Fetch relationships from API with pagination (get first 200 relationships)
  const {
    data: relationshipsData,
    isLoading: isLoadingRelationships,
    error: relationshipsError,
    refetch: refetchRelationships,
  } = useRelationships({ limit: 200 });

  // Fetch graph stats for the overview cards
  const { data: graphStats, isLoading: isLoadingStats } = useGraphStats();

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

  // Filter nodes based on search query
  const filteredNodes = useMemo(() => {
    if (!searchQuery.trim()) return allNodes;
    const query = searchQuery.toLowerCase();
    return allNodes.filter(
      (node) =>
        node.name.toLowerCase().includes(query) ||
        node.type.toLowerCase().includes(query)
    );
  }, [allNodes, searchQuery]);

  // Filter relationships to only include those connecting filtered nodes
  const filteredRelationships = useMemo(() => {
    if (!searchQuery.trim()) return allRelationships;
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    return allRelationships.filter(
      (rel) => nodeIds.has(rel.source_id) && nodeIds.has(rel.target_id)
    );
  }, [allRelationships, filteredNodes, searchQuery]);

  // Use filtered data for display
  const nodes = filteredNodes;
  const relationships = filteredRelationships;

  // Calculate stats from API data or use graph stats if available
  const stats = useMemo(() => {
    // If we have graph stats from the API, use those
    if (graphStats) {
      return {
        totalNodes: graphStats.total_nodes,
        totalRelationships: graphStats.total_relationships,
        nodesByType: graphStats.nodes_by_type,
      };
    }

    // Fall back to calculating from the loaded data
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
  }, [graphStats, nodes, relationships]);

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
      </div>

      {/* Legend */}
      <Card className="mb-6">
        <CardHeader className="py-3">
          <CardTitle className="text-sm font-medium">Node Type Legend</CardTitle>
        </CardHeader>
        <CardContent className="py-3">
          <div className="flex flex-wrap gap-3">
            {Object.entries(ENTITY_TYPE_COLORS).map(([type, color]) => (
              <div
                key={type}
                className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() => setSearchQuery(type)}
              >
                <div
                  className="h-3 w-3 rounded-full border border-white/20"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs text-[var(--color-foreground)]">{type}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Nodes</CardDescription>
            <CardTitle className="text-2xl">
              {isLoading || isLoadingStats ? (
                <span className="inline-block h-8 w-16 animate-pulse rounded bg-[var(--color-muted)]" />
              ) : (
                stats.totalNodes
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading || isLoadingStats ? (
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
              {isLoading || isLoadingStats ? (
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
