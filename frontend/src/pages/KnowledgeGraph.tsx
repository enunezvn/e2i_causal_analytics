/**
 * Knowledge Graph Page
 * ====================
 *
 * Main page component for the Knowledge Graph visualization.
 * Displays the interactive graph with nodes and relationships
 * from the E2I Causal Analytics system.
 *
 * @module pages/KnowledgeGraph
 */

import { useState, useMemo } from 'react';
import { KnowledgeGraph as KnowledgeGraphViz } from '@/components/visualizations/KnowledgeGraph';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { GraphNode, GraphRelationship, EntityType, RelationshipType } from '@/types/graph';

// =============================================================================
// SAMPLE DATA
// =============================================================================

/**
 * Sample nodes for demonstration
 * In production, this would come from useNodes() hook
 */
const SAMPLE_NODES: GraphNode[] = [
  {
    id: 'patient_001',
    type: 'Patient' as EntityType,
    name: 'Patient A',
    properties: { age: 45, condition: 'Type 2 Diabetes' },
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-15T10:00:00Z',
  },
  {
    id: 'hcp_001',
    type: 'HCP' as EntityType,
    name: 'Dr. Smith',
    properties: { specialty: 'Endocrinology', region: 'Northeast' },
    created_at: '2024-01-10T09:00:00Z',
    updated_at: '2024-01-10T09:00:00Z',
  },
  {
    id: 'brand_001',
    type: 'Brand' as EntityType,
    name: 'Kisqali',
    properties: { category: 'Oncology', manufacturer: 'Novartis' },
    created_at: '2024-01-05T08:00:00Z',
    updated_at: '2024-01-05T08:00:00Z',
  },
  {
    id: 'kpi_001',
    type: 'KPI' as EntityType,
    name: 'TRx Volume',
    properties: { metric_type: 'prescription', unit: 'count' },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'region_001',
    type: 'Region' as EntityType,
    name: 'Northeast',
    properties: { country: 'USA', population: 55000000 },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'treatment_001',
    type: 'Treatment' as EntityType,
    name: 'CDK4/6 Inhibitor Therapy',
    properties: { indication: 'Breast Cancer', stage: 'Phase 3' },
    created_at: '2024-01-08T12:00:00Z',
    updated_at: '2024-01-08T12:00:00Z',
  },
];

/**
 * Sample relationships for demonstration
 * In production, this would come from useRelationships() hook
 */
const SAMPLE_RELATIONSHIPS: GraphRelationship[] = [
  {
    id: 'rel_001',
    type: 'TREATED_BY' as RelationshipType,
    source_id: 'patient_001',
    target_id: 'hcp_001',
    properties: { since: '2023-06-01' },
    confidence: 0.95,
    created_at: '2024-01-15T10:00:00Z',
  },
  {
    id: 'rel_002',
    type: 'PRESCRIBES' as RelationshipType,
    source_id: 'hcp_001',
    target_id: 'brand_001',
    properties: { frequency: 'regular' },
    confidence: 0.88,
    created_at: '2024-01-12T14:00:00Z',
  },
  {
    id: 'rel_003',
    type: 'IMPACTS' as RelationshipType,
    source_id: 'brand_001',
    target_id: 'kpi_001',
    properties: { effect_size: 0.34 },
    confidence: 0.76,
    created_at: '2024-01-10T11:00:00Z',
  },
  {
    id: 'rel_004',
    type: 'PRACTICES_IN' as RelationshipType,
    source_id: 'hcp_001',
    target_id: 'region_001',
    properties: {},
    confidence: 1.0,
    created_at: '2024-01-10T09:00:00Z',
  },
  {
    id: 'rel_005',
    type: 'RELATES_TO' as RelationshipType,
    source_id: 'brand_001',
    target_id: 'treatment_001',
    properties: { relationship: 'primary_drug' },
    confidence: 0.92,
    created_at: '2024-01-08T12:00:00Z',
  },
];

// =============================================================================
// PAGE COMPONENT
// =============================================================================

function KnowledgeGraphPage() {
  // State for selected elements
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphRelationship | null>(null);

  // Calculate stats
  const stats = useMemo(() => {
    const nodesByType = SAMPLE_NODES.reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalNodes: SAMPLE_NODES.length,
      totalRelationships: SAMPLE_RELATIONSHIPS.length,
      nodesByType,
    };
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Knowledge Graph</h1>
        <p className="text-[var(--color-muted-foreground)]">
          Explore the knowledge graph visualization with interactive nodes and edges.
          Click on nodes to see details, drag to pan, and scroll to zoom.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Nodes</CardDescription>
            <CardTitle className="text-2xl">{stats.totalNodes}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1">
              {Object.entries(stats.nodesByType).map(([type, count]) => (
                <Badge key={type} variant="secondary" className="text-xs">
                  {type}: {count}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Relationships</CardDescription>
            <CardTitle className="text-2xl">{stats.totalRelationships}</CardTitle>
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
            nodes={SAMPLE_NODES}
            relationships={SAMPLE_RELATIONSHIPS}
            layout="cose"
            minHeight={500}
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
