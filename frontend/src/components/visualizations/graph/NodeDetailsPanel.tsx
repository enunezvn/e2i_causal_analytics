/**
 * NodeDetailsPanel Component
 * ==========================
 *
 * A slide-out panel that displays detailed information about a selected
 * node or edge in the knowledge graph. Provides a clean, organized view
 * of entity properties, relationships, and metadata.
 *
 * Features:
 * - Animated slide-in/out transitions
 * - Entity type badge with color coding
 * - Properties grid display
 * - Relationship summary for edges
 * - Confidence score visualization
 * - Close button and keyboard accessibility
 *
 * @module components/visualizations/graph/NodeDetailsPanel
 */

import * as React from 'react';
import { X, Calendar, Hash, Tag, ArrowRight, Percent } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import type { GraphNode, GraphRelationship } from '@/types/graph';

// =============================================================================
// TYPES
// =============================================================================

export interface NodeDetailsPanelProps {
  /** The selected node to display (mutually exclusive with edge) */
  node?: GraphNode | null;
  /** The selected edge to display (mutually exclusive with node) */
  edge?: GraphRelationship | null;
  /** Source node for edge display */
  sourceNode?: GraphNode | null;
  /** Target node for edge display */
  targetNode?: GraphNode | null;
  /** Whether the panel is visible */
  isOpen: boolean;
  /** Called when the panel should close */
  onClose: () => void;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Color mapping for entity types (matches KnowledgeGraph.tsx)
 */
const ENTITY_TYPE_COLORS: Record<string, string> = {
  Patient: '#3b82f6',
  HCP: '#10b981',
  Brand: '#f59e0b',
  Region: '#8b5cf6',
  KPI: '#ef4444',
  CausalPath: '#06b6d4',
  Trigger: '#f97316',
  Agent: '#ec4899',
  Episode: '#6366f1',
  Community: '#14b8a6',
  Treatment: '#84cc16',
  Prediction: '#a855f7',
  Experiment: '#22c55e',
  AgentActivity: '#64748b',
};

/**
 * Color mapping for relationship types
 */
const RELATIONSHIP_TYPE_COLORS: Record<string, string> = {
  CAUSES: '#ef4444',
  IMPACTS: '#f97316',
  INFLUENCES: '#f59e0b',
  TREATED_BY: '#10b981',
  PRESCRIBED: '#3b82f6',
  PRESCRIBES: '#3b82f6',
  DISCOVERED: '#8b5cf6',
  GENERATED: '#ec4899',
  MENTIONS: '#6b7280',
  MEMBER_OF: '#14b8a6',
  RELATES_TO: '#9ca3af',
  RECEIVED: '#22c55e',
  LOCATED_IN: '#8b5cf6',
  PRACTICES_IN: '#06b6d4',
  MEASURED_IN: '#a855f7',
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

/**
 * Property row component for displaying key-value pairs
 */
interface PropertyRowProps {
  label: string;
  value: React.ReactNode;
  icon?: React.ReactNode;
}

function PropertyRow({ label, value, icon }: PropertyRowProps) {
  return (
    <div className="flex items-start gap-2 py-2">
      {icon && (
        <span className="text-[var(--color-muted-foreground)] mt-0.5 shrink-0">
          {icon}
        </span>
      )}
      <div className="flex-1 min-w-0">
        <dt className="text-xs font-medium text-[var(--color-muted-foreground)] uppercase tracking-wider">
          {label}
        </dt>
        <dd className="text-sm text-[var(--color-foreground)] mt-0.5 break-words">
          {value}
        </dd>
      </div>
    </div>
  );
}

/**
 * Confidence meter component
 */
interface ConfidenceMeterProps {
  value: number;
}

function ConfidenceMeter({ value }: ConfidenceMeterProps) {
  const percentage = Math.round(value * 100);
  const getColor = () => {
    if (percentage >= 80) return 'bg-[#22c55e]'; // green
    if (percentage >= 60) return 'bg-[#f59e0b]'; // amber
    return 'bg-[#ef4444]'; // red
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-[var(--color-muted)]/30 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', getColor())}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-sm font-medium text-[var(--color-foreground)] tabular-nums">
        {percentage}%
      </span>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * NodeDetailsPanel displays detailed information about a selected graph element.
 *
 * @example
 * ```tsx
 * <NodeDetailsPanel
 *   node={selectedNode}
 *   isOpen={!!selectedNode}
 *   onClose={() => setSelectedNode(null)}
 * />
 * ```
 */
function NodeDetailsPanel({
  node,
  edge,
  sourceNode,
  targetNode,
  isOpen,
  onClose,
  className,
}: NodeDetailsPanelProps) {
  // Handle keyboard close
  React.useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape' && isOpen) {
        onClose();
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Get the entity type color
  const getTypeColor = (type: string): string => {
    return ENTITY_TYPE_COLORS[type] || '#6b7280';
  };

  // Get the relationship type color
  const getRelTypeColor = (type: string): string => {
    return RELATIONSHIP_TYPE_COLORS[type] || '#9ca3af';
  };

  // Format date string
  const formatDate = (dateStr: string | undefined): string => {
    if (!dateStr) return 'N/A';
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  // Render node details
  const renderNodeDetails = (nodeData: GraphNode) => (
    <>
      {/* Header */}
      <div className="flex items-start gap-3 mb-4">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: getTypeColor(nodeData.type) + '20' }}
        >
          <div
            className="w-6 h-6 rounded-full"
            style={{ backgroundColor: getTypeColor(nodeData.type) }}
          />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-[var(--color-foreground)] text-lg truncate">
            {nodeData.name}
          </h3>
          <Badge
            variant="outline"
            className="mt-1"
            style={{
              borderColor: getTypeColor(nodeData.type),
              color: getTypeColor(nodeData.type),
            }}
          >
            {nodeData.type}
          </Badge>
        </div>
      </div>

      <Separator className="my-4" />

      {/* Basic Info */}
      <div className="space-y-1">
        <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2">
          Basic Information
        </h4>
        <dl>
          <PropertyRow
            label="ID"
            value={<code className="text-xs bg-[var(--color-muted)]/30 px-1.5 py-0.5 rounded">{nodeData.id}</code>}
            icon={<Hash className="w-4 h-4" />}
          />
          <PropertyRow
            label="Created"
            value={formatDate(nodeData.created_at)}
            icon={<Calendar className="w-4 h-4" />}
          />
          {nodeData.updated_at && nodeData.updated_at !== nodeData.created_at && (
            <PropertyRow
              label="Updated"
              value={formatDate(nodeData.updated_at)}
              icon={<Calendar className="w-4 h-4" />}
            />
          )}
        </dl>
      </div>

      {/* Properties */}
      {Object.keys(nodeData.properties).length > 0 && (
        <>
          <Separator className="my-4" />
          <div>
            <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2">
              Properties
            </h4>
            <dl className="space-y-1">
              {Object.entries(nodeData.properties).map(([key, value]) => (
                <PropertyRow
                  key={key}
                  label={key.replace(/_/g, ' ')}
                  value={
                    typeof value === 'object'
                      ? JSON.stringify(value, null, 2)
                      : String(value)
                  }
                  icon={<Tag className="w-4 h-4" />}
                />
              ))}
            </dl>
          </div>
        </>
      )}
    </>
  );

  // Render edge details
  const renderEdgeDetails = (edgeData: GraphRelationship) => (
    <>
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: getRelTypeColor(edgeData.type) + '20' }}
        >
          <ArrowRight
            className="w-5 h-5"
            style={{ color: getRelTypeColor(edgeData.type) }}
          />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-[var(--color-foreground)] text-lg">
            Relationship
          </h3>
          <Badge
            variant="outline"
            className="mt-1"
            style={{
              borderColor: getRelTypeColor(edgeData.type),
              color: getRelTypeColor(edgeData.type),
            }}
          >
            {edgeData.type}
          </Badge>
        </div>
      </div>

      <Separator className="my-4" />

      {/* Connection */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-3">
          Connection
        </h4>
        <div className="flex items-center gap-2 text-sm">
          <div className="flex-1 p-2 bg-[var(--color-muted)]/20 rounded-lg">
            <div className="font-medium text-[var(--color-foreground)] truncate">
              {sourceNode?.name || edgeData.source_id}
            </div>
            {sourceNode && (
              <div className="text-xs text-[var(--color-muted-foreground)]">
                {sourceNode.type}
              </div>
            )}
          </div>
          <ArrowRight className="w-4 h-4 text-[var(--color-muted-foreground)] shrink-0" />
          <div className="flex-1 p-2 bg-[var(--color-muted)]/20 rounded-lg">
            <div className="font-medium text-[var(--color-foreground)] truncate">
              {targetNode?.name || edgeData.target_id}
            </div>
            {targetNode && (
              <div className="text-xs text-[var(--color-muted-foreground)]">
                {targetNode.type}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Confidence */}
      {edgeData.confidence !== undefined && (
        <>
          <Separator className="my-4" />
          <div>
            <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2 flex items-center gap-2">
              <Percent className="w-4 h-4" />
              Confidence Score
            </h4>
            <ConfidenceMeter value={edgeData.confidence} />
          </div>
        </>
      )}

      {/* Basic Info */}
      <Separator className="my-4" />
      <div className="space-y-1">
        <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2">
          Details
        </h4>
        <dl>
          <PropertyRow
            label="ID"
            value={<code className="text-xs bg-[var(--color-muted)]/30 px-1.5 py-0.5 rounded">{edgeData.id}</code>}
            icon={<Hash className="w-4 h-4" />}
          />
          {edgeData.created_at && (
            <PropertyRow
              label="Created"
              value={formatDate(edgeData.created_at)}
              icon={<Calendar className="w-4 h-4" />}
            />
          )}
        </dl>
      </div>

      {/* Properties */}
      {Object.keys(edgeData.properties).length > 0 && (
        <>
          <Separator className="my-4" />
          <div>
            <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2">
              Properties
            </h4>
            <dl className="space-y-1">
              {Object.entries(edgeData.properties).map(([key, value]) => (
                <PropertyRow
                  key={key}
                  label={key.replace(/_/g, ' ')}
                  value={
                    typeof value === 'object'
                      ? JSON.stringify(value, null, 2)
                      : String(value)
                  }
                  icon={<Tag className="w-4 h-4" />}
                />
              ))}
            </dl>
          </div>
        </>
      )}
    </>
  );

  return (
    <div
      className={cn(
        'fixed top-0 right-0 h-full w-80 bg-[var(--color-background)] border-l border-[var(--color-border)] shadow-xl z-50',
        'transform transition-transform duration-300 ease-in-out',
        isOpen ? 'translate-x-0' : 'translate-x-full',
        className
      )}
      role="dialog"
      aria-label={node ? 'Node details' : 'Relationship details'}
      aria-hidden={!isOpen}
    >
      {/* Close button */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-3 right-3 z-10"
        onClick={onClose}
        aria-label="Close panel"
      >
        <X className="w-4 h-4" />
      </Button>

      {/* Content */}
      <div className="h-full overflow-y-auto p-6 pt-14">
        {node && renderNodeDetails(node)}
        {edge && renderEdgeDetails(edge)}
        {!node && !edge && (
          <div className="flex items-center justify-center h-full text-[var(--color-muted-foreground)]">
            <p>Select a node or edge to view details</p>
          </div>
        )}
      </div>
    </div>
  );
}

NodeDetailsPanel.displayName = 'NodeDetailsPanel';

export { NodeDetailsPanel };
export default NodeDetailsPanel;
