/**
 * DataQuality Page Tests
 * ======================
 *
 * Tests for the Data Quality monitoring dashboard page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import DataQuality from './DataQuality';

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;
});

describe('DataQuality', () => {
  it('renders page header with title and description', () => {
    render(<DataQuality />);

    expect(screen.getByText('Data Quality')).toBeInTheDocument();
    expect(
      screen.getByText(/Data profiling, completeness metrics, accuracy checks/i)
    ).toBeInTheDocument();
  });

  it('displays 5 quality score KPI cards', () => {
    render(<DataQuality />);

    expect(screen.getByText('Overall Quality')).toBeInTheDocument();
    expect(screen.getByText('Completeness')).toBeInTheDocument();
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Consistency')).toBeInTheDocument();
    expect(screen.getByText('Timeliness')).toBeInTheDocument();
  });

  it('displays data sources section', () => {
    render(<DataQuality />);

    expect(screen.getByText('Data Sources')).toBeInTheDocument();
    // Data source names appear in both data sources section and validation rules table
    expect(screen.getAllByText('HCP Master').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Sales Transactions').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Prescriptions (TRx)').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Territory Mapping').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Market Access Data').length).toBeGreaterThanOrEqual(1);
  });

  it('displays refresh and export buttons', () => {
    render(<DataQuality />);

    expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Export Report/i })).toBeInTheDocument();
  });

  it('handles export button click', () => {
    const mockClick = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'a') {
        const link = originalCreateElement('a');
        link.click = mockClick;
        return link;
      }
      return originalCreateElement(tag);
    });

    render(<DataQuality />);

    const exportButton = screen.getByRole('button', { name: /Export Report/i });
    fireEvent.click(exportButton);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  it('displays tabs for validation rules, profiling, and issues', () => {
    render(<DataQuality />);

    expect(screen.getByRole('tab', { name: /Validation Rules/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Data Profiling/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Quality Issues/i })).toBeInTheDocument();
  });

  it('shows validation rules table by default', () => {
    render(<DataQuality />);

    // Table headers for validation rules
    expect(screen.getByText('Rule Name')).toBeInTheDocument();
    expect(screen.getByText('Data Source')).toBeInTheDocument();
    expect(screen.getByText('Target Field')).toBeInTheDocument();
  });

  it('displays search input for rules', () => {
    render(<DataQuality />);

    expect(screen.getByPlaceholderText('Search rules...')).toBeInTheDocument();
  });

  it('shows data source type badges', () => {
    render(<DataQuality />);

    // Data source types displayed as text
    const databaseBadges = screen.getAllByText('database');
    expect(databaseBadges.length).toBeGreaterThanOrEqual(1);
    const apiBadges = screen.getAllByText('api');
    expect(apiBadges.length).toBeGreaterThanOrEqual(1);
  });

  it('displays row counts for data sources', () => {
    render(<DataQuality />);

    // HCP Master has 125420 rows displayed as "125.4K rows"
    expect(screen.getByText('125.4K rows')).toBeInTheDocument();
    // Sales Transactions has 2450000 rows displayed as "2.5M rows"
    expect(screen.getByText('2.5M rows')).toBeInTheDocument();
  });

  it('displays quality issues count badge', () => {
    render(<DataQuality />);

    // There are 2 critical/error issues (Market Access Sync Failed + Foreign Key Violations)
    const issueCountBadge = screen.getByText('2');
    expect(issueCountBadge).toBeInTheDocument();
  });

  it('displays validation rule names', () => {
    render(<DataQuality />);

    expect(screen.getByText('HCP ID Not Null')).toBeInTheDocument();
    expect(screen.getByText('Valid NPI Format')).toBeInTheDocument();
    expect(screen.getByText('Sales Amount Range')).toBeInTheDocument();
    expect(screen.getByText('Unique Transaction ID')).toBeInTheDocument();
  });
});
