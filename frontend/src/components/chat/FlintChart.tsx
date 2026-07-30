/**
 * Flint Chart (chat generative UI)
 * ================================
 *
 * Renders a Plotly figure compiled by `flint-chart` inside a chatbot response.
 * Presentational and prop-driven, like KpiTrendChart: the action handler does
 * the fetching and the compiling, this only draws what it is handed.
 *
 * BUNDLING
 * --------
 * Both the compiler and the renderer are dynamically imported, so neither is in
 * the eager bundle: a session that never asks for a chart downloads neither.
 * Measured on this build — importing `lib/flint-chart` statically from the
 * provider added ~136 kB gzip to the main chunk (1,054 -> 1,191 kB); loading it
 * here instead returns the main chunk to its baseline and gives Flint its own
 * chunk. Plotly (~1.46 MB gzip) splits the same way, and was already a declared
 * dependency with no importer, so Flint's Plotly backend costs no new package —
 * no Vega runtime is added.
 *
 * This is also why the router hands over a LOGICAL encoding rather than
 * template channels: mapping one to the other needs Flint's template metadata,
 * which must not be pulled into the eager graph.
 *
 * States, kept distinct on purpose so an error never reads as "no data":
 * - loading:  handler still fetching
 * - empty:    the KPI genuinely has no data (`emptyReason` explains which case)
 * - error:    the request could not be charted (bad field, unknown chart type)
 * - figure:   a real chart
 *
 * @module components/chat/FlintChart
 */

import { useEffect, useRef, useState } from 'react';
import { Loader2, BarChart3, AlertTriangle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import type { KpiChartData } from '@/lib/kpi-chart-router';

/** Minimal surface of Plotly this component drives. */
interface PlotlyModule {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => void;
  purge: (el: HTMLElement) => void;
}

/**
 * Load Plotly once per session and share the promise.
 *
 * Each `import()` of the same specifier resolves the same module, but issuing a
 * fresh one from both the render effect and its cleanup made the two paths race
 * on unmount — the cleanup could resolve and purge a container the next mount
 * had already claimed. One shared promise keeps mount and purge on the same
 * module instance and skips the redundant resolution.
 */
let plotlyPromise: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min').then(
      (mod) => (mod.default ?? mod) as unknown as PlotlyModule
    );
  }
  return plotlyPromise;
}

export interface FlintChartProps {
  /** Chart heading, used for the loading and error frames. */
  title: string;
  /** Scope/provenance line under the heading. */
  subtitle?: string;
  /** Routed real data to compile and draw; undefined while loading. */
  chartData?: KpiChartData;
  /** Why there is no data to chart (honest empty state). */
  emptyReason?: string;
  /** Why the chart could not be built (distinct from having no data). */
  error?: string;
  /** True while the action handler is still fetching. */
  loading?: boolean;
  /** Chart height in px. */
  height?: number;
}

function Frame({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="my-2">
      <CardContent className="py-3">
        <div className="flex items-center gap-2 mb-1">
          <BarChart3 className="h-4 w-4 text-primary" />
          <span className="font-medium text-sm">{title}</span>
        </div>
        {subtitle ? <p className="text-xs text-muted-foreground mb-2">{subtitle}</p> : null}
        {children}
      </CardContent>
    </Card>
  );
}

export function FlintChart({
  title,
  subtitle,
  chartData,
  emptyReason,
  error,
  loading,
  height = 240,
}: FlintChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  const hasData = Boolean(chartData && chartData.rows.length > 0 && !emptyReason);

  useEffect(() => {
    if (!chartData || !hasData || !containerRef.current) return;
    const node = containerRef.current;
    let cancelled = false;
    setRenderError(null);

    // Compiler and renderer both load here, off the eager graph. They are
    // independent, so fetch them concurrently rather than in series.
    Promise.all([import('@/lib/flint-chart'), loadPlotly()])
      .then(([flint, Plotly]) => {
        if (cancelled || !node.isConnected) return;

        const compiled = flint.assembleKpiFigure({
          rows: chartData.rows,
          semanticTypes: chartData.semanticTypes,
          chartType: chartData.chartType,
          encodings: flint.encodingsFor(chartData.chartType, chartData.encoding),
        });
        if (!compiled.ok) {
          setRenderError(compiled.reason);
          return;
        }
        Plotly.react(node, compiled.figure.data, { ...compiled.figure.layout, autosize: true }, {
          displayModeBar: false,
          responsive: true,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setRenderError(err instanceof Error ? err.message : 'Chart renderer failed to load.');
      });

    return () => {
      cancelled = true;
      // Plotly attaches listeners and a WebGL context; purge on unmount so a
      // long chat thread does not accumulate them. Only if it actually loaded —
      // there is nothing to purge otherwise, and forcing the import here would
      // load the renderer solely to tear it down.
      plotlyPromise?.then((Plotly) => Plotly.purge(node)).catch(() => {
        /* renderer never loaded; nothing attached to this node */
      });
    };
  }, [chartData, hasData]);

  if (loading) {
    return (
      <Card className="my-2">
        <CardContent className="py-4 flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
          Building {title}…
        </CardContent>
      </Card>
    );
  }

  const failure = error ?? renderError;
  if (failure) {
    return (
      <Frame title={title} subtitle={subtitle}>
        <p className="flex items-start gap-2 text-sm text-muted-foreground">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-amber-500" />
          <span>Couldn’t build this chart: {failure}</span>
        </p>
      </Frame>
    );
  }

  if (!hasData) {
    return (
      <Frame title={title} subtitle={subtitle}>
        <p className="text-sm text-muted-foreground">
          {emptyReason ?? 'No chart is available for this request.'}
        </p>
      </Frame>
    );
  }

  return (
    <Frame title={chartData?.title ?? title} subtitle={chartData?.subtitle ?? subtitle}>
      <div
        ref={containerRef}
        style={{ height }}
        className="w-full"
        data-testid="flint-chart"
      />
    </Frame>
  );
}

export default FlintChart;
