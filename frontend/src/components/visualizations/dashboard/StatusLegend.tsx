/**
 * StatusLegend Component
 * ======================
 *
 * Displays status indicator legend with thresholds for:
 * - Performance Status Colors (Green/Yellow/Red)
 * - Data Quality Thresholds
 * - Model Performance Thresholds
 * - Trigger Performance Thresholds
 *
 * @module components/visualizations/dashboard/StatusLegend
 */

import React from 'react';

// ============================================================================
// Types
// ============================================================================

interface ThresholdItem {
  label: string;
  criteria: string;
}

interface ColorLegendItem {
  color: string;
  label: string;
  criteria: string;
}

export interface StatusLegendProps {
  /** Optional className for custom styling */
  className?: string;
}

// ============================================================================
// Data Constants
// ============================================================================

const STATUS_COLORS: ColorLegendItem[] = [
  {
    color: '#c6f6d5',
    label: 'Green - Optimal/Good',
    criteria: 'Meeting or exceeding target performance',
  },
  {
    color: '#feebc8',
    label: 'Yellow - Monitor',
    criteria: 'Below target but within acceptable range',
  },
  {
    color: '#fed7d7',
    label: 'Red - Critical',
    criteria: 'Below acceptable threshold, requires action',
  },
];

const DATA_QUALITY_THRESHOLDS: ThresholdItem[] = [
  { label: 'Source Coverage', criteria: 'Green: ≥80% | Yellow: 60-79% | Red: <60%' },
  { label: 'Data Lag', criteria: 'Green: ≤14 days | Yellow: 15-30 days | Red: >30 days' },
  { label: 'Match Rate', criteria: 'Green: ≥85% | Yellow: 70-84% | Red: <70%' },
  { label: 'Completeness', criteria: 'Green: ≥95% | Yellow: 90-94% | Red: <90%' },
];

const MODEL_PERFORMANCE_THRESHOLDS: ThresholdItem[] = [
  { label: 'ROC-AUC', criteria: 'Green: ≥0.80 | Yellow: 0.70-0.79 | Red: <0.70' },
  { label: 'Feature Drift (PSI)', criteria: 'Green: <0.10 | Yellow: 0.10-0.25 | Red: >0.25' },
  { label: 'Label Quality (IAA)', criteria: 'Green: κ≥0.70 | Yellow: κ=0.50-0.69 | Red: κ<0.50' },
  { label: 'Fairness Gap', criteria: 'Green: ≤5pp | Yellow: 6-10pp | Red: >10pp' },
];

const TRIGGER_PERFORMANCE_THRESHOLDS: ThresholdItem[] = [
  { label: 'Precision', criteria: 'Green: ≥40% | Yellow: 25-39% | Red: <25%' },
  { label: 'Acceptance Rate', criteria: 'Green: ≥65% | Yellow: 45-64% | Red: <45%' },
  { label: 'Lead Time', criteria: 'Green: 14-21 days | Yellow: 7-13 or 22-30 days | Red: <7 or >30 days' },
  { label: 'False Alert Rate', criteria: 'Green: ≤40% | Yellow: 41-60% | Red: >60%' },
];

// ============================================================================
// Sub-components
// ============================================================================

interface LegendSectionProps {
  title: string;
  emoji: string;
  children: React.ReactNode;
}

const LegendSection: React.FC<LegendSectionProps> = ({ title, emoji, children }) => (
  <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
    <div className="text-base font-semibold text-gray-800 mb-4 flex items-center gap-2">
      <span>{emoji}</span>
      <span>{title}</span>
    </div>
    <div className="space-y-3">{children}</div>
  </div>
);

interface ColorLegendItemRowProps {
  item: ColorLegendItem;
}

const ColorLegendItemRow: React.FC<ColorLegendItemRowProps> = ({ item }) => (
  <div className="flex items-start gap-3">
    <div
      className="w-6 h-6 rounded-md flex-shrink-0 border border-gray-200"
      style={{ backgroundColor: item.color }}
    />
    <div className="flex-1">
      <div className="font-medium text-gray-800 text-sm">{item.label}</div>
      <div className="text-xs text-gray-500 mt-0.5">{item.criteria}</div>
    </div>
  </div>
);

interface ThresholdItemRowProps {
  item: ThresholdItem;
}

const ThresholdItemRow: React.FC<ThresholdItemRowProps> = ({ item }) => (
  <div className="py-2 border-b border-gray-100 last:border-b-0">
    <div className="font-medium text-gray-800 text-sm">{item.label}</div>
    <div className="text-xs text-gray-500 mt-1 font-mono bg-gray-50 px-2 py-1 rounded">
      {item.criteria}
    </div>
  </div>
);

// ============================================================================
// Main Component
// ============================================================================

export const StatusLegend: React.FC<StatusLegendProps> = ({ className = '' }) => {
  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center mb-6">
        <h2 className="text-xl font-bold text-gray-800 flex items-center justify-center gap-2">
          <span>Status Indicator Legend & Thresholds</span>
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Reference guide for interpreting status indicators and performance thresholds
        </p>
      </div>

      {/* Legend Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Performance Status Colors */}
        <LegendSection title="Performance Status Colors" emoji="📊">
          {STATUS_COLORS.map((item, index) => (
            <ColorLegendItemRow key={index} item={item} />
          ))}
        </LegendSection>

        {/* Data Quality Thresholds */}
        <LegendSection title="Data Quality Thresholds" emoji="🎯">
          {DATA_QUALITY_THRESHOLDS.map((item, index) => (
            <ThresholdItemRow key={index} item={item} />
          ))}
        </LegendSection>

        {/* Model Performance Thresholds */}
        <LegendSection title="Model Performance Thresholds" emoji="🤖">
          {MODEL_PERFORMANCE_THRESHOLDS.map((item, index) => (
            <ThresholdItemRow key={index} item={item} />
          ))}
        </LegendSection>

        {/* Trigger Performance Thresholds */}
        <LegendSection title="Trigger Performance Thresholds" emoji="🎯">
          {TRIGGER_PERFORMANCE_THRESHOLDS.map((item, index) => (
            <ThresholdItemRow key={index} item={item} />
          ))}
        </LegendSection>
      </div>
    </div>
  );
};

export default StatusLegend;
