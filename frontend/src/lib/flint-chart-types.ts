/**
 * Flint chart types and vocabulary (no compiler)
 * ==============================================
 *
 * The parts of the chart layer that callers need EAGERLY: the chart-type list
 * the chat action advertises to the model, and the shapes the router speaks.
 *
 * Deliberately free of any `flint-chart` import. The compiler and its template
 * metadata live in `./flint-chart`, which is loaded lazily by FlintChart —
 * pulling it into the eager graph measured at ~136 kB gzip on the main chunk.
 * Anything added here must stay import-free of the package.
 *
 * @module lib/flint-chart-types
 */

/** A single data row handed to Flint. Values are real API figures. */
export type ChartRow = Record<string, string | number | null>;

/**
 * Chart types this module exposes to the chat, all backed by Plotly templates
 * and all satisfiable by the encodings the KPI router produces.
 *
 * Deliberately excludes band charts (Range Area Chart and friends): they need a
 * `y2` channel, and no routing branch emits one, so advertising them would give
 * the model an option that always compiles to zero traces.
 * `flint-chart.test.ts` compiles every entry here against every routed data
 * shape, so a dead option cannot reach the model unnoticed.
 */
export const SUPPORTED_CHART_TYPES = [
  'Line Chart',
  'Area Chart',
  'Bar Chart',
  'Grouped Bar Chart',
  'Stacked Bar Chart',
  'Scatter Plot',
  'Lollipop Chart',
  'Waterfall Chart',
  'Histogram',
  'Boxplot',
  'Heatmap',
  'KPI Card',
] as const;

export type SupportedChartType = (typeof SUPPORTED_CHART_TYPES)[number];

/**
 * Channel → field name. Mirrors Flint's encoding shorthand.
 *
 * Channels are NOT uniform across templates: most take x/y/color, Histogram
 * bins a single `x` measure and rejects `y`, and KPI Card takes
 * metric/value/goal. `encodingsFor` in ./flint-chart maps a logical shape onto
 * whichever channels the chosen template actually declares.
 */
export type ChartEncodings = Partial<
  Record<'x' | 'y' | 'y2' | 'color' | 'size' | 'metric' | 'value' | 'goal', string>
>;

/**
 * A chart shape described in terms of the DATA rather than a template's
 * channels: what runs along the axis, what is measured, what splits it into
 * series. The router speaks this; templates get channels derived from it.
 */
export interface LogicalEncoding {
  /** Field along the discrete/time axis (month, kpi name, bucket). */
  axis: string;
  /** The measure being plotted. */
  value: string;
  /** Optional field splitting the data into series. */
  series?: string;
  /** Optional target value field, drawn as a goal on a KPI Card. */
  goal?: string;
}

/** A Plotly figure: `data` traces plus `layout`. Rendered by FlintChart. */
export interface PlotlyFigure {
  data: unknown[];
  layout: Record<string, unknown>;
}
