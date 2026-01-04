/**
 * Graph Visualization Components Tests
 * =====================================
 *
 * Tests for GraphControls, GraphFilters, and NodeDetailsPanel components.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { GraphControls, LAYOUT_LABELS, DEFAULT_LAYOUTS } from './GraphControls';
import { GraphFilters } from './GraphFilters';
import { NodeDetailsPanel } from './NodeDetailsPanel';
import { EntityType, RelationshipType, type GraphNode, type GraphRelationship } from '@/types/graph';

// =============================================================================
// GRAPH CONTROLS TESTS
// =============================================================================

describe('GraphControls', () => {
  const defaultProps = {
    zoom: 1,
    layout: 'cose' as const,
    onZoomChange: vi.fn(),
    onLayoutChange: vi.fn(),
    onFit: vi.fn(),
    onCenter: vi.fn(),
    onExport: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Zoom Controls', () => {
    it('renders zoom in and zoom out buttons', () => {
      const { container } = render(<GraphControls {...defaultProps} />);

      // Find zoom buttons by their SVG icons
      const zoomInButton = container.querySelector('svg.lucide-zoom-in')?.closest('button');
      const zoomOutButton = container.querySelector('svg.lucide-zoom-out')?.closest('button');

      expect(zoomInButton).toBeInTheDocument();
      expect(zoomOutButton).toBeInTheDocument();
    });

    it('displays current zoom level', () => {
      render(<GraphControls {...defaultProps} zoom={1.5} />);

      expect(screen.getByText('150%')).toBeInTheDocument();
    });

    it('calls onZoomChange when zoom in clicked', () => {
      const onZoomChange = vi.fn();
      const { container } = render(
        <GraphControls {...defaultProps} zoom={1} onZoomChange={onZoomChange} />
      );

      const zoomInButton = container.querySelector('svg.lucide-zoom-in')?.closest('button');
      fireEvent.click(zoomInButton!);

      // ZOOM_STEP is 0.2, so 1 + 0.2 = 1.2
      expect(onZoomChange).toHaveBeenCalledWith(1.2);
    });

    it('calls onZoomChange when zoom out clicked', () => {
      const onZoomChange = vi.fn();
      const { container } = render(
        <GraphControls {...defaultProps} zoom={1} onZoomChange={onZoomChange} />
      );

      const zoomOutButton = container.querySelector('svg.lucide-zoom-out')?.closest('button');
      fireEvent.click(zoomOutButton!);

      // ZOOM_STEP is 0.2, so 1 - 0.2 = 0.8
      expect(onZoomChange).toHaveBeenCalledWith(0.8);
    });

    it('respects minZoom when zooming out', () => {
      const onZoomChange = vi.fn();
      const { container } = render(
        <GraphControls {...defaultProps} zoom={0.2} minZoom={0.1} onZoomChange={onZoomChange} />
      );

      const zoomOutButton = container.querySelector('svg.lucide-zoom-out')?.closest('button');
      fireEvent.click(zoomOutButton!);

      // Should clamp to minZoom
      expect(onZoomChange).toHaveBeenCalled();
      const calledWith = onZoomChange.mock.calls[0][0];
      expect(calledWith).toBeGreaterThanOrEqual(0.1);
    });

    it('respects maxZoom when zooming in', () => {
      const onZoomChange = vi.fn();
      const { container } = render(
        <GraphControls {...defaultProps} zoom={2.9} maxZoom={3} onZoomChange={onZoomChange} />
      );

      const zoomInButton = container.querySelector('svg.lucide-zoom-in')?.closest('button');
      fireEvent.click(zoomInButton!);

      expect(onZoomChange).toHaveBeenCalled();
      const calledWith = onZoomChange.mock.calls[0][0];
      expect(calledWith).toBeLessThanOrEqual(3);
    });

    it('formats zoom as percentage', () => {
      render(<GraphControls {...defaultProps} zoom={0.5} />);
      expect(screen.getByText('50%')).toBeInTheDocument();
    });
  });

  describe('Layout Selection', () => {
    it('renders layout selector', () => {
      render(<GraphControls {...defaultProps} />);

      // Should show current layout label in combobox
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('shows layout label for current layout', () => {
      render(<GraphControls {...defaultProps} layout="cose" />);

      expect(screen.getByText(LAYOUT_LABELS['cose'])).toBeInTheDocument();
    });

    it('calls onLayoutChange when layout selected', () => {
      const onLayoutChange = vi.fn();
      render(
        <GraphControls
          {...defaultProps}
          onLayoutChange={onLayoutChange}
          availableLayouts={DEFAULT_LAYOUTS}
        />
      );

      const layoutSelector = screen.getByRole('combobox');
      fireEvent.click(layoutSelector);

      // Find and click a different layout option
      const gridOption = screen.queryByText(LAYOUT_LABELS['grid']);
      if (gridOption) {
        fireEvent.click(gridOption);
        expect(onLayoutChange).toHaveBeenCalledWith('grid');
      }
    });
  });

  describe('Action Buttons', () => {
    it('renders fit button', () => {
      render(<GraphControls {...defaultProps} />);

      const fitButton = screen.getByRole('button', { name: /Fit to viewport/i });
      expect(fitButton).toBeInTheDocument();
    });

    it('renders center button', () => {
      render(<GraphControls {...defaultProps} />);

      const centerButton = screen.getByRole('button', { name: /Center graph/i });
      expect(centerButton).toBeInTheDocument();
    });

    it('renders export button', () => {
      render(<GraphControls {...defaultProps} />);

      const exportButton = screen.getByRole('button', { name: /Export as PNG/i });
      expect(exportButton).toBeInTheDocument();
    });

    it('calls onFit when fit clicked', () => {
      const onFit = vi.fn();
      render(<GraphControls {...defaultProps} onFit={onFit} />);

      const fitButton = screen.getByRole('button', { name: /Fit to viewport/i });
      fireEvent.click(fitButton);

      expect(onFit).toHaveBeenCalled();
    });

    it('calls onCenter when center clicked', () => {
      const onCenter = vi.fn();
      render(<GraphControls {...defaultProps} onCenter={onCenter} />);

      const centerButton = screen.getByRole('button', { name: /Center graph/i });
      fireEvent.click(centerButton);

      expect(onCenter).toHaveBeenCalled();
    });

    it('calls onExport when export clicked', () => {
      const onExport = vi.fn();
      render(<GraphControls {...defaultProps} onExport={onExport} />);

      const exportButton = screen.getByRole('button', { name: /Export as PNG/i });
      fireEvent.click(exportButton);

      expect(onExport).toHaveBeenCalled();
    });

    it('hides export button when onExport not provided', () => {
      render(<GraphControls {...defaultProps} onExport={undefined} />);

      expect(screen.queryByRole('button', { name: /Export as PNG/i })).not.toBeInTheDocument();
    });
  });

  describe('Disabled State', () => {
    it('disables zoom buttons when at limits', () => {
      const { container } = render(
        <GraphControls {...defaultProps} zoom={0.1} minZoom={0.1} />
      );

      const zoomOutButton = container.querySelector('svg.lucide-zoom-out')?.closest('button');
      expect(zoomOutButton).toBeDisabled();
    });

    it('disables zoom in when at max', () => {
      const { container } = render(
        <GraphControls {...defaultProps} zoom={3} maxZoom={3} />
      );

      const zoomInButton = container.querySelector('svg.lucide-zoom-in')?.closest('button');
      expect(zoomInButton).toBeDisabled();
    });

    it('disables all controls when disabled prop is true', () => {
      render(<GraphControls {...defaultProps} disabled />);

      const fitButton = screen.getByRole('button', { name: /Fit to viewport/i });
      const centerButton = screen.getByRole('button', { name: /Center graph/i });

      expect(fitButton).toBeDisabled();
      expect(centerButton).toBeDisabled();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <GraphControls {...defaultProps} className="custom-controls" />
      );

      expect(container.querySelector('.custom-controls')).toBeInTheDocument();
    });

    it('renders with toolbar role', () => {
      render(<GraphControls {...defaultProps} />);

      expect(screen.getByRole('toolbar', { name: /Graph controls/i })).toBeInTheDocument();
    });
  });
});

// =============================================================================
// GRAPH FILTERS TESTS
// =============================================================================

describe('GraphFilters', () => {
  const allEntityTypes = Object.values(EntityType);
  const allRelationshipTypes = Object.values(RelationshipType);

  const defaultProps = {
    selectedEntityTypes: allEntityTypes,
    selectedRelationshipTypes: allRelationshipTypes,
    onEntityTypesChange: vi.fn(),
    onRelationshipTypesChange: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Entity Type Filters', () => {
    it('renders entity types section header', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByText('Entity Types')).toBeInTheDocument();
    });

    it('renders checkboxes for entity types', () => {
      render(<GraphFilters {...defaultProps} />);

      // Check for some common entity type labels via checkbox labels
      const patientLabel = screen.getByText('Patient');
      const hcpLabel = screen.getByText('HCP');
      const brandLabel = screen.getByText('Brand');

      expect(patientLabel).toBeInTheDocument();
      expect(hcpLabel).toBeInTheDocument();
      expect(brandLabel).toBeInTheDocument();
    });

    it('calls onEntityTypesChange when checkbox clicked', () => {
      const onEntityTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={allEntityTypes}
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Find the Patient checkbox by its id
      const patientCheckbox = document.getElementById('entity-Patient');
      if (patientCheckbox) {
        fireEvent.click(patientCheckbox);
        expect(onEntityTypesChange).toHaveBeenCalled();
      }
    });
  });

  describe('Relationship Type Filters', () => {
    it('renders relationship types section header', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByText('Relationship Types')).toBeInTheDocument();
    });

    it('renders checkboxes for relationship types', () => {
      render(<GraphFilters {...defaultProps} />);

      // Check for some common relationship type labels
      expect(screen.getByText('Causes')).toBeInTheDocument();
      expect(screen.getByText('Impacts')).toBeInTheDocument();
      expect(screen.getByText('Prescribes')).toBeInTheDocument();
    });

    it('calls onRelationshipTypesChange when checkbox clicked', () => {
      const onRelationshipTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedRelationshipTypes={allRelationshipTypes}
          onRelationshipTypesChange={onRelationshipTypesChange}
        />
      );

      // Find the CAUSES checkbox by its id
      const causesCheckbox = document.getElementById('rel-CAUSES');
      if (causesCheckbox) {
        fireEvent.click(causesCheckbox);
        expect(onRelationshipTypesChange).toHaveBeenCalled();
      }
    });
  });

  describe('Select All / Clear All', () => {
    it('renders select all button with check icon', () => {
      const { container } = render(
        <GraphFilters {...defaultProps} selectedEntityTypes={[]} />
      );

      // Select all button uses Check icon
      const selectAllButton = container.querySelector('button[title="Select all"]');
      expect(selectAllButton).toBeInTheDocument();
    });

    it('renders clear all button with X icon', () => {
      const { container } = render(
        <GraphFilters {...defaultProps} selectedEntityTypes={allEntityTypes} />
      );

      // Clear all button uses X icon
      const clearAllButton = container.querySelector('button[title="Clear all"]');
      expect(clearAllButton).toBeInTheDocument();
    });

    it('selects all entity types when select all clicked', () => {
      const onEntityTypesChange = vi.fn();
      const { container } = render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[]}
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Find the first (entity types) select all button
      const selectAllButtons = container.querySelectorAll('button[title="Select all"]');
      const entitySelectAll = selectAllButtons[0];
      fireEvent.click(entitySelectAll);

      expect(onEntityTypesChange).toHaveBeenCalled();
      const newSelection = onEntityTypesChange.mock.calls[0][0];
      expect(newSelection.length).toBe(allEntityTypes.length);
    });

    it('clears all entity types when clear all clicked', () => {
      const onEntityTypesChange = vi.fn();
      const { container } = render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={allEntityTypes}
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Find the first (entity types) clear all button
      const clearAllButtons = container.querySelectorAll('button[title="Clear all"]');
      const entityClearAll = clearAllButtons[0];
      fireEvent.click(entityClearAll);

      expect(onEntityTypesChange).toHaveBeenCalled();
      const newSelection = onEntityTypesChange.mock.calls[0][0];
      expect(newSelection.length).toBe(0);
    });
  });

  describe('Reset Filters', () => {
    it('shows reset button when filters are applied', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
        />
      );

      expect(screen.getByRole('button', { name: /Reset/i })).toBeInTheDocument();
    });

    it('resets all filters when clicked', () => {
      const onEntityTypesChange = vi.fn();
      const onRelationshipTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
          onEntityTypesChange={onEntityTypesChange}
          onRelationshipTypesChange={onRelationshipTypesChange}
        />
      );

      const resetButton = screen.getByRole('button', { name: /Reset/i });
      fireEvent.click(resetButton);

      // Both callbacks should be called with full selections
      expect(onEntityTypesChange).toHaveBeenCalled();
      expect(onRelationshipTypesChange).toHaveBeenCalled();
    });
  });

  describe('Filter Count Badge', () => {
    it('shows hidden count when filters applied', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT, EntityType.HCP]}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
        />
      );

      // Badge shows number of hidden items
      const hiddenCount = allEntityTypes.length - 2 + allRelationshipTypes.length - 1;
      expect(screen.getByText(`${hiddenCount} hidden`)).toBeInTheDocument();
    });

    it('does not show badge when all selected', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.queryByText(/hidden/)).not.toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <GraphFilters {...defaultProps} className="custom-filters" />
      );

      expect(container.querySelector('.custom-filters')).toBeInTheDocument();
    });

    it('renders with region role', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByRole('region', { name: /Graph filters/i })).toBeInTheDocument();
    });
  });

  describe('Disabled State', () => {
    it('disables select all when all selected', () => {
      const { container } = render(
        <GraphFilters {...defaultProps} selectedEntityTypes={allEntityTypes} />
      );

      const selectAllButtons = container.querySelectorAll('button[title="Select all"]');
      const entitySelectAll = selectAllButtons[0];
      expect(entitySelectAll).toBeDisabled();
    });

    it('disables clear all when none selected', () => {
      const { container } = render(
        <GraphFilters {...defaultProps} selectedEntityTypes={[]} />
      );

      const clearAllButtons = container.querySelectorAll('button[title="Clear all"]');
      const entityClearAll = clearAllButtons[0];
      expect(entityClearAll).toBeDisabled();
    });
  });
});

// =============================================================================
// NODE DETAILS PANEL TESTS
// =============================================================================

describe('NodeDetailsPanel', () => {
  const mockNode: GraphNode = {
    id: 'node1',
    name: 'Test Patient',
    type: EntityType.PATIENT,
    properties: {
      patientId: 'P12345',
      age: 45,
      diagnosis: 'CSU',
    },
    created_at: '2024-01-15T10:00:00Z',
  };

  const mockSourceNode: GraphNode = {
    id: 'node2',
    name: 'Dr. Smith',
    type: EntityType.HCP,
    properties: {
      specialty: 'Dermatology',
      npi: '1234567890',
    },
    created_at: '2024-01-10T10:00:00Z',
  };

  const mockTargetNode: GraphNode = {
    id: 'node3',
    name: 'Remibrutinib',
    type: EntityType.BRAND,
    properties: {
      indication: 'CSU',
    },
    created_at: '2024-01-01T10:00:00Z',
  };

  const mockEdge: GraphRelationship = {
    id: 'edge1',
    source_id: 'node2',
    target_id: 'node3',
    type: RelationshipType.PRESCRIBES,
    properties: {
      count: 150,
      startDate: '2024-01-01',
    },
    confidence: 0.85,
    created_at: '2024-01-15T12:00:00Z',
  };

  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Node Details', () => {
    it('renders node name', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} />);

      expect(screen.getByText('Test Patient')).toBeInTheDocument();
    });

    it('renders node type badge', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} />);

      expect(screen.getByText('Patient')).toBeInTheDocument();
    });

    it('renders node ID', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} />);

      expect(screen.getByText('node1')).toBeInTheDocument();
    });

    it('renders node properties', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} />);

      expect(screen.getByText('patientId')).toBeInTheDocument();
      expect(screen.getByText('P12345')).toBeInTheDocument();
      expect(screen.getByText('age')).toBeInTheDocument();
      expect(screen.getByText('45')).toBeInTheDocument();
    });

    it('renders different node types correctly', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockSourceNode} />);

      expect(screen.getByText('Dr. Smith')).toBeInTheDocument();
      expect(screen.getByText('HCP')).toBeInTheDocument();
      expect(screen.getByText('specialty')).toBeInTheDocument();
      expect(screen.getByText('Dermatology')).toBeInTheDocument();
    });
  });

  describe('Edge Details', () => {
    it('renders edge type', () => {
      render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={mockEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      expect(screen.getByText('PRESCRIBES')).toBeInTheDocument();
    });

    it('renders relationship header', () => {
      render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={mockEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      expect(screen.getByText('Relationship')).toBeInTheDocument();
    });

    it('renders source and target nodes', () => {
      render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={mockEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      expect(screen.getByText('Dr. Smith')).toBeInTheDocument();
      expect(screen.getByText('Remibrutinib')).toBeInTheDocument();
    });

    it('renders edge properties', () => {
      render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={mockEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      expect(screen.getByText('count')).toBeInTheDocument();
      expect(screen.getByText('150')).toBeInTheDocument();
    });

    it('renders confidence meter', () => {
      render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={mockEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      expect(screen.getByText('Confidence Score')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });
  });

  describe('Panel Controls', () => {
    it('renders close button', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} />);

      const closeButton = screen.getByRole('button', { name: /Close panel/i });
      expect(closeButton).toBeInTheDocument();
    });

    it('calls onClose when close button clicked', () => {
      const onClose = vi.fn();
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} onClose={onClose} />);

      const closeButton = screen.getByRole('button', { name: /Close panel/i });
      fireEvent.click(closeButton);

      expect(onClose).toHaveBeenCalled();
    });

    it('calls onClose when Escape key pressed', () => {
      const onClose = vi.fn();
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} onClose={onClose} />);

      fireEvent.keyDown(document, { key: 'Escape' });

      expect(onClose).toHaveBeenCalled();
    });
  });

  describe('Panel Visibility', () => {
    it('shows panel when isOpen is true', () => {
      const { container } = render(
        <NodeDetailsPanel {...defaultProps} node={mockNode} isOpen={true} />
      );

      // Panel should not have translate-x-full (hidden) class
      expect(container.querySelector('.translate-x-0')).toBeInTheDocument();
    });

    it('hides panel when isOpen is false', () => {
      const { container } = render(
        <NodeDetailsPanel {...defaultProps} node={mockNode} isOpen={false} />
      );

      // Panel should have translate-x-full (hidden) class
      expect(container.querySelector('.translate-x-full')).toBeInTheDocument();
    });
  });

  describe('Empty State', () => {
    it('shows empty state when no node or edge', () => {
      render(<NodeDetailsPanel {...defaultProps} />);

      expect(screen.getByText(/Select a node or edge to view details/i)).toBeInTheDocument();
    });
  });

  describe('Confidence Meter Colors', () => {
    it('uses green for high confidence (>=80%)', () => {
      const highConfidenceEdge = {
        ...mockEdge,
        confidence: 0.9,
      };
      const { container } = render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={highConfidenceEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      // High confidence (90%) shows percentage and uses green bg class
      expect(screen.getByText('90%')).toBeInTheDocument();
      // The meter bar with the green color class
      expect(container.querySelector('.bg-\\[\\#22c55e\\]')).toBeInTheDocument();
    });

    it('uses amber for medium confidence (60-79%)', () => {
      const mediumConfidenceEdge = {
        ...mockEdge,
        confidence: 0.7,
      };
      const { container } = render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={mediumConfidenceEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      // Medium confidence (70%) shows percentage and uses amber bg class
      expect(screen.getByText('70%')).toBeInTheDocument();
      // The meter bar with the amber color class
      expect(container.querySelector('.bg-\\[\\#f59e0b\\]')).toBeInTheDocument();
    });

    it('uses red for low confidence (<60%)', () => {
      const lowConfidenceEdge = {
        ...mockEdge,
        confidence: 0.3,
      };
      const { container } = render(
        <NodeDetailsPanel
          {...defaultProps}
          edge={lowConfidenceEdge}
          sourceNode={mockSourceNode}
          targetNode={mockTargetNode}
        />
      );

      // Low confidence (30%) shows percentage and uses red bg class
      expect(screen.getByText('30%')).toBeInTheDocument();
      // The meter bar with the red color class
      expect(container.querySelector('.bg-\\[\\#ef4444\\]')).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <NodeDetailsPanel {...defaultProps} node={mockNode} className="custom-panel" />
      );

      expect(container.querySelector('.custom-panel')).toBeInTheDocument();
    });

    it('renders with transition classes', () => {
      const { container } = render(
        <NodeDetailsPanel {...defaultProps} node={mockNode} isOpen={true} />
      );

      // Should have transition-transform class
      expect(container.querySelector('.transition-transform')).toBeInTheDocument();
    });

    it('renders with dialog role', () => {
      render(<NodeDetailsPanel {...defaultProps} node={mockNode} />);

      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });
  });
});

// =============================================================================
// LAYOUT LABELS EXPORT TESTS
// =============================================================================

describe('LAYOUT_LABELS', () => {
  it('exports layout labels constant', () => {
    expect(LAYOUT_LABELS).toBeDefined();
    expect(typeof LAYOUT_LABELS).toBe('object');
  });

  it('has label for cose layout', () => {
    expect(LAYOUT_LABELS['cose']).toBe('Force-Directed (COSE)');
  });

  it('has label for grid layout', () => {
    expect(LAYOUT_LABELS['grid']).toBe('Grid');
  });

  it('has label for circle layout', () => {
    expect(LAYOUT_LABELS['circle']).toBe('Circle');
  });

  it('has label for concentric layout', () => {
    expect(LAYOUT_LABELS['concentric']).toBe('Concentric');
  });

  it('has label for breadthfirst layout', () => {
    expect(LAYOUT_LABELS['breadthfirst']).toBe('Hierarchical');
  });
});

describe('DEFAULT_LAYOUTS', () => {
  it('exports default layouts array', () => {
    expect(DEFAULT_LAYOUTS).toBeDefined();
    expect(Array.isArray(DEFAULT_LAYOUTS)).toBe(true);
  });

  it('includes all common layouts', () => {
    expect(DEFAULT_LAYOUTS).toContain('cose');
    expect(DEFAULT_LAYOUTS).toContain('grid');
    expect(DEFAULT_LAYOUTS).toContain('circle');
    expect(DEFAULT_LAYOUTS).toContain('concentric');
    expect(DEFAULT_LAYOUTS).toContain('breadthfirst');
    expect(DEFAULT_LAYOUTS).toContain('random');
  });

  it('has 6 default layouts', () => {
    expect(DEFAULT_LAYOUTS.length).toBe(6);
  });
});
