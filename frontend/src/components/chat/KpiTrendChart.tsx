/**
 * KPI Trend Chart (chat generative UI)
 * ====================================
 *
 * Inline line chart rendered inside a chatbot response via CopilotKit
 * generative UI (a `useCopilotAction` `render`). It is intentionally a pure,
 * presentational component driven entirely by props so it is easy to test and
 * so the data path stays honest: the action handler fetches REAL KPI history
 * (`getKPIHistory`) and passes it here — this component never fabricates points.
 *
 * States:
 * - loading: the action handler is still fetching
 * - empty: no real series exists (point-in-time KPIs legitimately have none)
 * - data: a monthly line chart of real values
 *
 * @module components/chat/KpiTrendChart
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';
import { Loader2, LineChart as LineChartIcon } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import type { KPIHistoryResponse } from '@/types/kpi';

export interface KpiTrendChartProps {
  /** KPI identifier being charted (used for titles/empty copy). */
  kpiId: string;
  /** Optional chart title; defaults to "<kpiId> trend". */
  title?: string;
  /** Real KPI history from getKPIHistory; undefined while loading. */
  data?: KPIHistoryResponse;
  /** True while the action handler is still fetching. */
  loading?: boolean;
}

/** Format a YYYY-MM-DD month key to a compact axis label (e.g. "Jan '26").
 *
 * Parses the parts as a LOCAL date — `new Date('2026-01-01')` is UTC midnight,
 * which renders as the prior month in negative-offset timezones (the classic
 * ISO-date off-by-one). */
function formatMonth(metricDate: string): string {
  const [year, month, day] = metricDate.split('-').map(Number);
  if (!year || !month) return metricDate;
  const parsed = new Date(year, month - 1, day || 1);
  if (Number.isNaN(parsed.getTime())) return metricDate;
  const monthLabel = parsed.toLocaleDateString(undefined, { month: 'short' });
  const yearLabel = `'${String(parsed.getFullYear()).slice(-2)}`;
  return `${monthLabel} ${yearLabel}`;
}

export function KpiTrendChart({ kpiId, title, data, loading }: KpiTrendChartProps) {
  const heading = title || `${kpiId} trend`;

  if (loading) {
    return (
      <Card className="my-2">
        <CardContent className="py-4 flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
          Loading {kpiId} trend…
        </CardContent>
      </Card>
    );
  }

  // No well-formed result (e.g. the fetch failed / the KPI id was invalid).
  // Distinct from a successful-but-empty series so we don't pass off an error
  // as "no data".
  if (!data) {
    return (
      <Card className="my-2">
        <CardContent className="py-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-2 font-medium text-foreground">
            <LineChartIcon className="h-4 w-4" />
            {heading}
          </div>
          <p className="mt-1">Couldn’t load the trend for “{kpiId}”.</p>
        </CardContent>
      </Card>
    );
  }

  const points = data.points ?? [];

  if (points.length === 0) {
    return (
      <Card className="my-2">
        <CardContent className="py-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-2 font-medium text-foreground">
            <LineChartIcon className="h-4 w-4" />
            {heading}
          </div>
          <p className="mt-1">
            No historical series available for “{kpiId}”. (Point-in-time KPIs have no monthly history.)
          </p>
        </CardContent>
      </Card>
    );
  }

  const chartData = points.map((p) => ({
    label: formatMonth(p.metric_date),
    value: p.value,
  }));

  // Scope line: brand/region are '' when global/all.
  const scopeParts = [data.brand, data.region].filter((s) => s && s.length > 0);
  const scope = scopeParts.length > 0 ? scopeParts.join(' · ') : 'All brands · all regions';

  return (
    <Card className="my-2">
      <CardContent className="py-3">
        <div className="flex items-center gap-2 mb-1">
          <LineChartIcon className="h-4 w-4 text-primary" />
          <span className="font-medium text-sm">{heading}</span>
        </div>
        <p className="text-xs text-muted-foreground mb-2">
          {scope} · {points.length} month{points.length === 1 ? '' : 's'}
        </p>
        <div className="h-40 w-full" data-testid="kpi-trend-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="label" stroke="var(--muted-foreground)" fontSize={10} />
              <YAxis stroke="var(--muted-foreground)" fontSize={10} width={40} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 2 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

export default KpiTrendChart;
