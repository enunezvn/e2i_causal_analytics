/**
 * GraphFilters Component
 * ======================
 *
 * Filtering controls for the knowledge graph visualization.
 * Allows filtering nodes by entity type and relationships by type.
 *
 * Features:
 * - Entity type checkboxes with color indicators
 * - Relationship type checkboxes
 * - Select all / clear all actions
 * - Active filter count badge
 *
 * @module components/visualizations/graph/GraphFilters
 */

import * as React from 'react';
import { useCallback, useMemo } from 'react';
import { Filter, Check, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { EntityType, RelationshipType } from '@/types/graph';

// =============================================================================
// TYPES
// =============================================================================

export interface GraphFiltersProps {
  /** Currently selected entity types */
  selectedEntityTypes: EntityType[];
  /** Currently selected relationship types */
  selectedRelationshipTypes: RelationshipType[];
  /** Whether filters are disabled */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Called when entity type selection changes */
  onEntityTypesChange: (types: EntityType[]) => void;
  /** Called when relationship type selection changes */
  onRelationshipTypesChange: (types: RelationshipType[]) => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

/** All available entity types */
const ALL_ENTITY_TYPES = Object.values(EntityType);

/** All available relationship types */
const ALL_RELATIONSHIP_TYPES = Object.values(RelationshipType);

/**
 * Color mapping for entity types (matches KnowledgeGraph)
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

/** Entity type display labels */
const ENTITY_TYPE_LABELS: Record<string, string> = {
  Patient: 'Patient',
  HCP: 'HCP',
  Brand: 'Brand',
  Region: 'Region',
  KPI: 'KPI',
  CausalPath: 'Causal Path',
  Trigger: 'Trigger',
  Agent: 'Agent',
  Episode: 'Episode',
  Community: 'Community',
  Treatment: 'Treatment',
  Prediction: 'Prediction',
  Experiment: 'Experiment',
  AgentActivity: 'Agent Activity',
};

/** Relationship type display labels */
const RELATIONSHIP_TYPE_LABELS: Record<string, string> = {
  TREATED_BY: 'Treated By',
  PRESCRIBED: 'Prescribed',
  PRESCRIBES: 'Prescribes',
  CAUSES: 'Causes',
  IMPACTS: 'Impacts',
  INFLUENCES: 'Influences',
  DISCOVERED: 'Discovered',
  GENERATED: 'Generated',
  MENTIONS: 'Mentions',
  MEMBER_OF: 'Member Of',
  RELATES_TO: 'Relates To',
  RECEIVED: 'Received',
  LOCATED_IN: 'Located In',
  PRACTICES_IN: 'Practices In',
  MEASURED_IN: 'Measured In',
};

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * GraphFilters provides filtering controls for the knowledge graph.
 *
 * @example
 * ```tsx
 * const [entityTypes, setEntityTypes] = useState<EntityType[]>(Object.values(EntityType));
 * const [relTypes, setRelTypes] = useState<RelationshipType[]>(Object.values(RelationshipType));
 *
 * <GraphFilters
 *   selectedEntityTypes={entityTypes}
 *   selectedRelationshipTypes={relTypes}
 *   onEntityTypesChange={setEntityTypes}
 *   onRelationshipTypesChange={setRelTypes}
 * />
 * ```
 */
const GraphFilters = React.forwardRef<HTMLDivElement, GraphFiltersProps>(
  (
    {
      selectedEntityTypes,
      selectedRelationshipTypes,
      disabled = false,
      className,
      onEntityTypesChange,
      onRelationshipTypesChange,
    },
    ref
  ) => {
    // Calculate active filter counts
    const entityFilterCount = useMemo(
      () => ALL_ENTITY_TYPES.length - selectedEntityTypes.length,
      [selectedEntityTypes]
    );

    const relationshipFilterCount = useMemo(
      () => ALL_RELATIONSHIP_TYPES.length - selectedRelationshipTypes.length,
      [selectedRelationshipTypes]
    );

    const totalFilterCount = entityFilterCount + relationshipFilterCount;

    // Handle entity type toggle
    const handleEntityTypeToggle = useCallback(
      (type: EntityType, checked: boolean) => {
        if (checked) {
          onEntityTypesChange([...selectedEntityTypes, type]);
        } else {
          onEntityTypesChange(selectedEntityTypes.filter((t) => t !== type));
        }
      },
      [selectedEntityTypes, onEntityTypesChange]
    );

    // Handle relationship type toggle
    const handleRelationshipTypeToggle = useCallback(
      (type: RelationshipType, checked: boolean) => {
        if (checked) {
          onRelationshipTypesChange([...selectedRelationshipTypes, type]);
        } else {
          onRelationshipTypesChange(
            selectedRelationshipTypes.filter((t) => t !== type)
          );
        }
      },
      [selectedRelationshipTypes, onRelationshipTypesChange]
    );

    // Select all entity types
    const selectAllEntityTypes = useCallback(() => {
      onEntityTypesChange([...ALL_ENTITY_TYPES]);
    }, [onEntityTypesChange]);

    // Clear all entity types
    const clearAllEntityTypes = useCallback(() => {
      onEntityTypesChange([]);
    }, [onEntityTypesChange]);

    // Select all relationship types
    const selectAllRelationshipTypes = useCallback(() => {
      onRelationshipTypesChange([...ALL_RELATIONSHIP_TYPES]);
    }, [onRelationshipTypesChange]);

    // Clear all relationship types
    const clearAllRelationshipTypes = useCallback(() => {
      onRelationshipTypesChange([]);
    }, [onRelationshipTypesChange]);

    // Reset all filters
    const resetAllFilters = useCallback(() => {
      onEntityTypesChange([...ALL_ENTITY_TYPES]);
      onRelationshipTypesChange([...ALL_RELATIONSHIP_TYPES]);
    }, [onEntityTypesChange, onRelationshipTypesChange]);

    return (
      <div
        ref={ref}
        className={cn(
          'p-4 bg-[var(--color-card)] border border-[var(--color-border)] rounded-lg shadow-sm',
          className
        )}
        role="region"
        aria-label="Graph filters"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-[var(--color-muted-foreground)]" />
            <span className="text-sm font-medium">Filters</span>
            {totalFilterCount > 0 && (
              <Badge variant="secondary" className="text-xs">
                {totalFilterCount} hidden
              </Badge>
            )}
          </div>
          {totalFilterCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={resetAllFilters}
              disabled={disabled}
              className="h-7 text-xs"
            >
              Reset
            </Button>
          )}
        </div>

        {/* Entity Types Section */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-[var(--color-muted-foreground)] uppercase tracking-wide">
              Entity Types
            </span>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={selectAllEntityTypes}
                disabled={
                  disabled ||
                  selectedEntityTypes.length === ALL_ENTITY_TYPES.length
                }
                className="h-6 w-6 p-0"
                title="Select all"
              >
                <Check className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAllEntityTypes}
                disabled={disabled || selectedEntityTypes.length === 0}
                className="h-6 w-6 p-0"
                title="Clear all"
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {ALL_ENTITY_TYPES.map((type) => (
              <div key={type} className="flex items-center gap-2">
                <Checkbox
                  id={`entity-${type}`}
                  checked={selectedEntityTypes.includes(type)}
                  onCheckedChange={(checked) =>
                    handleEntityTypeToggle(type, checked === true)
                  }
                  disabled={disabled}
                />
                <Label
                  htmlFor={`entity-${type}`}
                  className="flex items-center gap-2 text-sm cursor-pointer whitespace-nowrap"
                >
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{
                      backgroundColor: ENTITY_TYPE_COLORS[type] || '#6b7280',
                    }}
                    aria-hidden="true"
                  />
                  {ENTITY_TYPE_LABELS[type] || type}
                </Label>
              </div>
            ))}
          </div>
        </div>

        <Separator className="my-4" />

        {/* Relationship Types Section */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-[var(--color-muted-foreground)] uppercase tracking-wide">
              Relationship Types
            </span>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={selectAllRelationshipTypes}
                disabled={
                  disabled ||
                  selectedRelationshipTypes.length ===
                    ALL_RELATIONSHIP_TYPES.length
                }
                className="h-6 w-6 p-0"
                title="Select all"
              >
                <Check className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAllRelationshipTypes}
                disabled={disabled || selectedRelationshipTypes.length === 0}
                className="h-6 w-6 p-0"
                title="Clear all"
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto">
            {ALL_RELATIONSHIP_TYPES.map((type) => (
              <div key={type} className="flex items-center gap-2">
                <Checkbox
                  id={`rel-${type}`}
                  checked={selectedRelationshipTypes.includes(type)}
                  onCheckedChange={(checked) =>
                    handleRelationshipTypeToggle(type, checked === true)
                  }
                  disabled={disabled}
                />
                <Label
                  htmlFor={`rel-${type}`}
                  className="text-sm cursor-pointer whitespace-nowrap"
                >
                  {RELATIONSHIP_TYPE_LABELS[type] || type}
                </Label>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
);

GraphFilters.displayName = 'GraphFilters';

export {
  GraphFilters,
  ENTITY_TYPE_COLORS,
  ENTITY_TYPE_LABELS,
  RELATIONSHIP_TYPE_LABELS,
  ALL_ENTITY_TYPES,
  ALL_RELATIONSHIP_TYPES,
};
export default GraphFilters;
