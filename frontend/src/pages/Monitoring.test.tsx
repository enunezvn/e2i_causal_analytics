/**
 * Monitoring Page Tests
 * =====================
 *
 * Tests for the Monitoring dashboard page with API usage,
 * user activity, error tracking, and system metrics.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Monitoring from './Monitoring';

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;
});

describe('Monitoring', () => {
  it('renders page header with title and description', () => {
    render(<Monitoring />);

    expect(screen.getByText('Monitoring')).toBeInTheDocument();
    expect(
      screen.getByText(/User activity logs, API usage statistics, error tracking/i)
    ).toBeInTheDocument();
  });

  it('displays 6 overview KPI cards', () => {
    render(<Monitoring />);

    // Some labels appear in both KPI cards and table headers
    expect(screen.getByText('Total Requests')).toBeInTheDocument();
    expect(screen.getAllByText('Error Rate').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Avg Latency').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Active Users')).toBeInTheDocument();
    expect(screen.getByText('Total Errors')).toBeInTheDocument();
    expect(screen.getByText('Uptime')).toBeInTheDocument();
  });

  it('displays time range selector', () => {
    render(<Monitoring />);

    // Time range dropdown - look for trigger button
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('displays refresh and export buttons', () => {
    render(<Monitoring />);

    expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
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

    render(<Monitoring />);

    const exportButton = screen.getByRole('button', { name: /Export/i });
    fireEvent.click(exportButton);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  it('displays 4 main tabs', () => {
    render(<Monitoring />);

    expect(screen.getByRole('tab', { name: /API Usage/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /User Activity/i })).toBeInTheDocument();
    // Errors tab includes badge count
    const errorsTab = screen.getByRole('tab', { name: /Errors/i });
    expect(errorsTab).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /System/i })).toBeInTheDocument();
  });

  it('shows API Usage tab content by default', () => {
    render(<Monitoring />);

    expect(screen.getByText('Request Volume & Errors')).toBeInTheDocument();
    expect(screen.getByText('Response Latency')).toBeInTheDocument();
    expect(screen.getByText('Endpoint Statistics')).toBeInTheDocument();
  });

  it('displays endpoint statistics table headers', () => {
    render(<Monitoring />);

    // Table headers - may appear multiple times if there are multiple tables
    expect(screen.getAllByText('Endpoint').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Method').length).toBeGreaterThanOrEqual(1);
  });

  it('displays HTTP method badges', () => {
    render(<Monitoring />);

    // Method badges
    const getBadges = screen.getAllByText('GET');
    expect(getBadges.length).toBeGreaterThanOrEqual(1);
    const postBadges = screen.getAllByText('POST');
    expect(postBadges.length).toBeGreaterThanOrEqual(1);
  });

  it('shows endpoint paths in statistics table', () => {
    render(<Monitoring />);

    expect(screen.getByText('/api/v1/query')).toBeInTheDocument();
    expect(screen.getByText('/api/v1/kpis')).toBeInTheDocument();
    expect(screen.getByText('/api/v1/health')).toBeInTheDocument();
  });

  it('displays error count badge on Errors tab', () => {
    render(<Monitoring />);

    // Errors tab has a badge with count of critical/error level logs
    // There are 4 error/critical level entries in sample data
    const errorsTab = screen.getByRole('tab', { name: /Errors/i });
    expect(errorsTab).toBeInTheDocument();
    // Badge should contain "4" (3 errors + 1 critical)
    expect(screen.getByText('4')).toBeInTheDocument();
  });
});
