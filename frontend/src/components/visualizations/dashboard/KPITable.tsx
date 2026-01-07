/**
 * KPITable Component
 * ==================
 *
 * Reusable table component for displaying KPI definitions with:
 * - Gradient headers
 * - Formula text styling (monospace)
 * - Hover states
 * - Section title with emoji support
 *
 * @module components/visualizations/dashboard/KPITable
 */

import React from 'react';

// ============================================================================
// Types
// ============================================================================

export interface KPITableRow {
  /** KPI name */
  name: string;
  /** KPI definition/description */
  definition: string;
  /** KPI formula (displayed in monospace) */
  formula: string;
}

export interface KPITableProps {
  /** Section title displayed above the table */
  title: string;
  /** Emoji to display before the title */
  emoji?: string;
  /** Array of KPI rows to display */
  rows: KPITableRow[];
  /** Optional className for custom styling */
  className?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export const KPITable: React.FC<KPITableProps> = ({
  title,
  emoji = '',
  rows,
  className = '',
}) => {
  return (
    <div className={`mb-6 ${className}`}>
      {/* Section Title */}
      <div className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
        {emoji && <span>{emoji}</span>}
        <span>{title}</span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
              <th className="w-1/4 px-4 py-3 text-left text-sm font-semibold">
                KPI
              </th>
              <th className="w-[35%] px-4 py-3 text-left text-sm font-semibold">
                Definition
              </th>
              <th className="w-[40%] px-4 py-3 text-left text-sm font-semibold">
                Formula
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr
                key={index}
                className={`
                  border-b border-gray-100 last:border-b-0
                  hover:bg-indigo-50 transition-colors duration-150
                  ${index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                `}
              >
                <td className="px-4 py-3">
                  <strong className="text-gray-800 text-sm">{row.name}</strong>
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">
                  {row.definition}
                </td>
                <td className="px-4 py-3">
                  <code className="text-xs bg-gray-100 text-indigo-700 px-2 py-1 rounded font-mono">
                    {row.formula}
                  </code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default KPITable;
