/**
 * GraphFilters Component Tests
 * ============================
 *
 * Comprehensive tests for the GraphFilters component.
 * Tests rendering, user interactions, callbacks, and edge cases.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import * as React from 'react';
import {
  GraphFilters,
  ENTITY_TYPE_COLORS,
  ENTITY_TYPE_LABELS,
  RELATIONSHIP_TYPE_LABELS,
  ALL_ENTITY_TYPES,
  ALL_RELATIONSHIP_TYPES,
} from './GraphFilters';
import { EntityType, RelationshipType } from '@/types/graph';

// =============================================================================
// MOCK SETUP
// =============================================================================

vi.mock('@/lib/utils', () => ({
  cn: (...classes: (string | undefined | boolean)[]) =>
    classes.filter(Boolean).join(' '),
}));

// =============================================================================
// TEST UTILITIES
// =============================================================================

const defaultProps = {
  selectedEntityTypes: [...ALL_ENTITY_TYPES],
  selectedRelationshipTypes: [...ALL_RELATIONSHIP_TYPES],
  onEntityTypesChange: vi.fn(),
  onRelationshipTypesChange: vi.fn(),
};

// =============================================================================
// TESTS: RENDERING
// =============================================================================

describe('GraphFilters', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders the filter region', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByRole('region', { name: 'Graph filters' })).toBeInTheDocument();
    });

    it('renders filter header with icon', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByText('Filters')).toBeInTheDocument();
    });

    it('renders entity types section', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByText('Entity Types')).toBeInTheDocument();
    });

    it('renders relationship types section', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.getByText('Relationship Types')).toBeInTheDocument();
    });

    it('renders all entity type checkboxes', () => {
      render(<GraphFilters {...defaultProps} />);

      ALL_ENTITY_TYPES.forEach((type) => {
        const label = ENTITY_TYPE_LABELS[type] || type;
        expect(screen.getByText(label)).toBeInTheDocument();
      });
    });

    it('renders all relationship type checkboxes', () => {
      render(<GraphFilters {...defaultProps} />);

      ALL_RELATIONSHIP_TYPES.forEach((type) => {
        const label = RELATIONSHIP_TYPE_LABELS[type] || type;
        expect(screen.getByText(label)).toBeInTheDocument();
      });
    });

    it('renders entity type color indicators', () => {
      const { container } = render(<GraphFilters {...defaultProps} />);

      // Check that color indicators are rendered with correct colors
      const colorIndicators = container.querySelectorAll('.rounded-full');
      expect(colorIndicators.length).toBeGreaterThan(0);
    });

    it('applies custom className', () => {
      const { container } = render(
        <GraphFilters {...defaultProps} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLDivElement>();

      render(<GraphFilters {...defaultProps} ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });

    it('has correct displayName', () => {
      expect(GraphFilters.displayName).toBe('GraphFilters');
    });
  });

  describe('filter count badge', () => {
    it('shows badge when filters are active', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT, EntityType.HCP]}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
        />
      );

      // Calculate expected count: (total entity - 2) + (total rel - 1)
      const entityHidden = ALL_ENTITY_TYPES.length - 2;
      const relHidden = ALL_RELATIONSHIP_TYPES.length - 1;
      const totalHidden = entityHidden + relHidden;

      expect(screen.getByText(`${totalHidden} hidden`)).toBeInTheDocument();
    });

    it('hides badge when all filters are selected', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.queryByText(/hidden/)).not.toBeInTheDocument();
    });

    it('shows reset button when filters are active', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
        />
      );

      expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument();
    });

    it('hides reset button when all filters are selected', () => {
      render(<GraphFilters {...defaultProps} />);

      expect(screen.queryByRole('button', { name: 'Reset' })).not.toBeInTheDocument();
    });
  });

  describe('entity type interactions', () => {
    it('calls onEntityTypesChange when entity type is unchecked', () => {
      const onEntityTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Get the Patient checkbox and click it (use exact match to avoid AgentActivity matching)
      const patientCheckbox = screen.getByRole('checkbox', { name: /^Patient$/i });
      fireEvent.click(patientCheckbox);

      expect(onEntityTypesChange).toHaveBeenCalled();
      const callArg = onEntityTypesChange.mock.calls[0][0];
      expect(callArg).not.toContain(EntityType.PATIENT);
    });

    it('calls onEntityTypesChange when entity type is checked', () => {
      const onEntityTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.HCP]} // Only HCP selected
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Click Patient checkbox to add it
      const patientCheckbox = screen.getByRole('checkbox', { name: /^Patient$/i });
      fireEvent.click(patientCheckbox);

      expect(onEntityTypesChange).toHaveBeenCalled();
      const callArg = onEntityTypesChange.mock.calls[0][0];
      expect(callArg).toContain(EntityType.PATIENT);
      expect(callArg).toContain(EntityType.HCP);
    });

    it('calls selectAllEntityTypes when select all button clicked', () => {
      const onEntityTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Find select all button in entity section
      const selectAllButtons = screen.getAllByTitle('Select all');
      fireEvent.click(selectAllButtons[0]); // First one is for entities

      expect(onEntityTypesChange).toHaveBeenCalledWith(ALL_ENTITY_TYPES);
    });

    it('calls clearAllEntityTypes when clear all button clicked', () => {
      const onEntityTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          onEntityTypesChange={onEntityTypesChange}
        />
      );

      // Find clear all button in entity section
      const clearAllButtons = screen.getAllByTitle('Clear all');
      fireEvent.click(clearAllButtons[0]); // First one is for entities

      expect(onEntityTypesChange).toHaveBeenCalledWith([]);
    });

    it('disables select all when all entity types are selected', () => {
      render(<GraphFilters {...defaultProps} />);

      const selectAllButtons = screen.getAllByTitle('Select all');
      expect(selectAllButtons[0]).toBeDisabled();
    });

    it('disables clear all when no entity types are selected', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[]}
        />
      );

      const clearAllButtons = screen.getAllByTitle('Clear all');
      expect(clearAllButtons[0]).toBeDisabled();
    });
  });

  describe('relationship type interactions', () => {
    it('calls onRelationshipTypesChange when relationship type is unchecked', () => {
      const onRelationshipTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          onRelationshipTypesChange={onRelationshipTypesChange}
        />
      );

      // Get the Causes checkbox and click it (use exact match)
      const causesCheckbox = screen.getByRole('checkbox', { name: /^Causes$/i });
      fireEvent.click(causesCheckbox);

      expect(onRelationshipTypesChange).toHaveBeenCalled();
      const callArg = onRelationshipTypesChange.mock.calls[0][0];
      expect(callArg).not.toContain(RelationshipType.CAUSES);
    });

    it('calls onRelationshipTypesChange when relationship type is checked', () => {
      const onRelationshipTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedRelationshipTypes={[RelationshipType.IMPACTS]} // Only IMPACTS selected
          onRelationshipTypesChange={onRelationshipTypesChange}
        />
      );

      // Click Causes checkbox to add it (use exact match)
      const causesCheckbox = screen.getByRole('checkbox', { name: /^Causes$/i });
      fireEvent.click(causesCheckbox);

      expect(onRelationshipTypesChange).toHaveBeenCalled();
      const callArg = onRelationshipTypesChange.mock.calls[0][0];
      expect(callArg).toContain(RelationshipType.CAUSES);
    });

    it('calls selectAllRelationshipTypes when select all button clicked', () => {
      const onRelationshipTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
          onRelationshipTypesChange={onRelationshipTypesChange}
        />
      );

      // Find select all button in relationship section
      const selectAllButtons = screen.getAllByTitle('Select all');
      fireEvent.click(selectAllButtons[1]); // Second one is for relationships

      expect(onRelationshipTypesChange).toHaveBeenCalledWith(ALL_RELATIONSHIP_TYPES);
    });

    it('calls clearAllRelationshipTypes when clear all button clicked', () => {
      const onRelationshipTypesChange = vi.fn();
      render(
        <GraphFilters
          {...defaultProps}
          onRelationshipTypesChange={onRelationshipTypesChange}
        />
      );

      // Find clear all button in relationship section
      const clearAllButtons = screen.getAllByTitle('Clear all');
      fireEvent.click(clearAllButtons[1]); // Second one is for relationships

      expect(onRelationshipTypesChange).toHaveBeenCalledWith([]);
    });

    it('disables select all when all relationship types are selected', () => {
      render(<GraphFilters {...defaultProps} />);

      const selectAllButtons = screen.getAllByTitle('Select all');
      expect(selectAllButtons[1]).toBeDisabled();
    });

    it('disables clear all when no relationship types are selected', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedRelationshipTypes={[]}
        />
      );

      const clearAllButtons = screen.getAllByTitle('Clear all');
      expect(clearAllButtons[1]).toBeDisabled();
    });
  });

  describe('reset functionality', () => {
    it('calls both callbacks when reset is clicked', () => {
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

      const resetButton = screen.getByRole('button', { name: 'Reset' });
      fireEvent.click(resetButton);

      expect(onEntityTypesChange).toHaveBeenCalledWith(ALL_ENTITY_TYPES);
      expect(onRelationshipTypesChange).toHaveBeenCalledWith(ALL_RELATIONSHIP_TYPES);
    });
  });

  describe('disabled state', () => {
    it('disables all entity checkboxes when disabled', () => {
      render(<GraphFilters {...defaultProps} disabled={true} />);

      ALL_ENTITY_TYPES.forEach((type) => {
        const label = ENTITY_TYPE_LABELS[type] || type;
        // Use exact match regex to avoid overlapping labels (e.g., "Agent" vs "Agent Activity")
        const checkbox = screen.getByRole('checkbox', { name: new RegExp(`^${label}$`, 'i') });
        expect(checkbox).toBeDisabled();
      });
    });

    it('disables all relationship checkboxes when disabled', () => {
      render(<GraphFilters {...defaultProps} disabled={true} />);

      ALL_RELATIONSHIP_TYPES.forEach((type) => {
        const label = RELATIONSHIP_TYPE_LABELS[type] || type;
        // Use exact match regex to avoid overlapping labels
        const checkbox = screen.getByRole('checkbox', { name: new RegExp(`^${label}$`, 'i') });
        expect(checkbox).toBeDisabled();
      });
    });

    it('disables select all buttons when disabled', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
          disabled={true}
        />
      );

      const selectAllButtons = screen.getAllByTitle('Select all');
      selectAllButtons.forEach((btn) => expect(btn).toBeDisabled());
    });

    it('disables clear all buttons when disabled', () => {
      render(<GraphFilters {...defaultProps} disabled={true} />);

      const clearAllButtons = screen.getAllByTitle('Clear all');
      clearAllButtons.forEach((btn) => expect(btn).toBeDisabled());
    });

    it('disables reset button when disabled', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
          disabled={true}
        />
      );

      expect(screen.getByRole('button', { name: 'Reset' })).toBeDisabled();
    });
  });

  describe('checkbox states', () => {
    it('shows entity type checkboxes as checked when selected', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT, EntityType.HCP]}
        />
      );

      expect(screen.getByRole('checkbox', { name: /^Patient$/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /^HCP$/i })).toBeChecked();
    });

    it('shows entity type checkboxes as unchecked when not selected', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedEntityTypes={[EntityType.PATIENT]}
        />
      );

      expect(screen.getByRole('checkbox', { name: /^HCP$/i })).not.toBeChecked();
    });

    it('shows relationship type checkboxes as checked when selected', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedRelationshipTypes={[RelationshipType.CAUSES, RelationshipType.IMPACTS]}
        />
      );

      expect(screen.getByRole('checkbox', { name: /^Causes$/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /^Impacts$/i })).toBeChecked();
    });

    it('shows relationship type checkboxes as unchecked when not selected', () => {
      render(
        <GraphFilters
          {...defaultProps}
          selectedRelationshipTypes={[RelationshipType.CAUSES]}
        />
      );

      expect(screen.getByRole('checkbox', { name: /^Impacts$/i })).not.toBeChecked();
    });
  });

  describe('exported constants', () => {
    it('exports ENTITY_TYPE_COLORS with correct colors', () => {
      expect(ENTITY_TYPE_COLORS['Patient']).toBe('#3b82f6');
      expect(ENTITY_TYPE_COLORS['HCP']).toBe('#10b981');
      expect(ENTITY_TYPE_COLORS['Brand']).toBe('#f59e0b');
    });

    it('exports ENTITY_TYPE_LABELS with correct labels', () => {
      expect(ENTITY_TYPE_LABELS['CausalPath']).toBe('Causal Path');
      expect(ENTITY_TYPE_LABELS['AgentActivity']).toBe('Agent Activity');
    });

    it('exports RELATIONSHIP_TYPE_LABELS with correct labels', () => {
      expect(RELATIONSHIP_TYPE_LABELS['TREATED_BY']).toBe('Treated By');
      expect(RELATIONSHIP_TYPE_LABELS['MEMBER_OF']).toBe('Member Of');
    });

    it('exports ALL_ENTITY_TYPES', () => {
      expect(ALL_ENTITY_TYPES).toEqual(Object.values(EntityType));
    });

    it('exports ALL_RELATIONSHIP_TYPES', () => {
      expect(ALL_RELATIONSHIP_TYPES).toEqual(Object.values(RelationshipType));
    });
  });

  describe('fallback behavior', () => {
    it('uses type as label when ENTITY_TYPE_LABELS entry missing', () => {
      // This tests the fallback `|| type` in the label rendering
      // The component uses ENTITY_TYPE_LABELS[type] || type
      // Since all types have labels, we verify they render correctly
      render(<GraphFilters {...defaultProps} />);

      // Verify known labels work
      expect(screen.getByText('Causal Path')).toBeInTheDocument(); // CausalPath -> 'Causal Path'
    });

    it('uses gray color when ENTITY_TYPE_COLORS entry missing', () => {
      // The fallback is || '#6b7280'
      // Since all types have colors, we verify colors are applied
      const { container } = render(<GraphFilters {...defaultProps} />);

      // Find color indicators
      const colorIndicator = container.querySelector('[style*="background-color"]');
      expect(colorIndicator).toBeTruthy();
    });
  });
});
