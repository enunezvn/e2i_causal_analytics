/**
 * Flint chart assembly (chat generative UI)
 * =========================================
 *
 * Compiles REAL KPI data into a Plotly figure via `flint-chart`, Microsoft's
 * semantic-level chart compiler. Flint derives scales, axis formatting, tick
 * steps, colour schemes and label sizing from the data plus a declared
 * semantic type, so the chat can offer many chart types without a hand-tuned
 * React component per shape.
 *
 * THE DATA RULE
 * -------------
 * The chat model supplies WHAT to chart (kpi, chart type, comparison axis) and
 * never the numbers. `data.values` is populated here, from responses the action
 * handler fetched off the KPI API. Flint accepts inline rows, which makes
 * "let the model write the spec" the obvious-looking integration and also the
 * one that would let a model emit plausible-wrong pharma figures straight into
 * a chart. Rows only ever enter through {@link assembleKpiFigure}.
 *
 * WHY THE VALIDATION SHIM
 * -----------------------
 * Measured against flint-chart@0.4.1: `assemblePlotly` does not validate. An
 * encoding naming a field absent from the data compiles to a spec with that
 * field typed nominal — no throw, no `warnings` entry — and renders as an empty
 * chart. An unrecognised semantic type is likewise accepted silently. Since
 * both values originate with an LLM, {@link validateChartRequest} checks them
 * before assembly so a bad request surfaces as an explicit error state instead
 * of a blank plot.
 *
 * @module lib/flint-chart
 */

import { assemblePlotly, plAllTemplateDefs, SemanticTypes } from 'flint-chart';
import { SUPPORTED_CHART_TYPES } from './flint-chart-types';
import type {
  ChartEncodings,
  ChartRow,
  LogicalEncoding,
  PlotlyFigure,
  SupportedChartType,
} from './flint-chart-types';

// Re-exported so callers that already load this module (tests, FlintChart) can
// take the vocabulary from one place. Eager callers must import
// './flint-chart-types' directly -- importing here pulls in the compiler.
export { SUPPORTED_CHART_TYPES };
export type {
  ChartEncodings,
  ChartRow,
  LogicalEncoding,
  PlotlyFigure,
  SupportedChartType,
};

export interface ChartRequest {
  rows: ChartRow[];
  semanticTypes: Record<string, string>;
  chartType: string;
  encodings: ChartEncodings;
  width?: number;
  height?: number;
}

export type ChartResult =
  | { ok: true; figure: PlotlyFigure }
  | { ok: false; reason: string };

/** Plotly template names Flint actually ships, for validating chartType. */
const PLOTLY_CHART_TYPES: ReadonlySet<string> = new Set(
  (plAllTemplateDefs as ReadonlyArray<{ chart: string }>).map((d) => d.chart)
);

/** The semantic-type vocabulary Flint recognises; anything else degrades silently. */
const VALID_SEMANTIC_TYPES: ReadonlySet<string> = new Set(
  Array.isArray(SemanticTypes)
    ? (SemanticTypes as readonly string[])
    : Object.keys(SemanticTypes ?? {})
);

/** Channels each Plotly template declares, keyed by chart type. */
const TEMPLATE_CHANNELS: ReadonlyMap<string, ReadonlySet<string>> = new Map(
  (plAllTemplateDefs as ReadonlyArray<{ chart: string; channels: string[] }>).map((d) => [
    d.chart,
    new Set(d.channels),
  ])
);

/**
 * Map a logical shape onto the channels a given chart type declares.
 *
 * Three families, because Flint's templates genuinely differ:
 * - KPI Card renders a Plotly `indicator`: metric label, value, optional goal.
 * - Histogram bins one measure along x and has no y channel at all.
 * - Everything else is the familiar x / y / color.
 */
export function encodingsFor(
  chartType: string,
  logical: LogicalEncoding
): ChartEncodings {
  const channels = TEMPLATE_CHANNELS.get(chartType);

  if (channels?.has('metric') && channels.has('value')) {
    return {
      metric: logical.axis,
      value: logical.value,
      ...(logical.goal && channels.has('goal') ? { goal: logical.goal } : {}),
    };
  }

  if (channels?.has('x') && !channels.has('y')) {
    // Distribution templates (Histogram) bin the measure itself.
    return {
      x: logical.value,
      ...(logical.series && channels.has('color') ? { color: logical.series } : {}),
    };
  }

  return {
    x: logical.axis,
    y: logical.value,
    ...(logical.series ? { color: logical.series } : {}),
  };
}

/**
 * Reject a chart request Flint would otherwise compile into a silently-empty
 * plot. Returns null when the request is sound.
 *
 * Checks, in the order a bad LLM request tends to fail:
 * 1. rows present — an empty series is the caller's honest-empty state, not a chart
 * 2. chart type is a real Plotly template
 * 3. every encoded field exists on the rows
 * 4. every declared semantic type is in Flint's vocabulary
 */
export function validateChartRequest(request: ChartRequest): string | null {
  const { rows, chartType, encodings, semanticTypes } = request;

  if (!Array.isArray(rows) || rows.length === 0) {
    return 'No data points to chart.';
  }
  if (!PLOTLY_CHART_TYPES.has(chartType)) {
    return `Unsupported chart type “${chartType}”.`;
  }

  // Union of keys across rows: segmented series can leave a bucket absent from
  // an early month, and that is not a hallucinated field.
  const present = new Set<string>();
  for (const row of rows) for (const key of Object.keys(row)) present.add(key);

  const encoded = Object.entries(encodings).filter(([, field]) => Boolean(field));
  if (encoded.length === 0) {
    return 'No fields were mapped to chart axes.';
  }
  const channels = TEMPLATE_CHANNELS.get(chartType);
  for (const [channel, field] of encoded) {
    if (!present.has(field as string)) {
      return `Field “${field}” (${channel}) is not present in the data.`;
    }
    // A channel the template does not declare is dropped during assembly, which
    // is how a Histogram given a `y` field silently loses it.
    if (channels && !channels.has(channel)) {
      return `Chart type “${chartType}” has no “${channel}” channel.`;
    }
  }
  for (const [field, type] of Object.entries(semanticTypes)) {
    if (!VALID_SEMANTIC_TYPES.has(type)) {
      return `Unknown semantic type “${type}” for field “${field}”.`;
    }
  }
  return null;
}

/**
 * Compile validated rows into a Plotly figure.
 *
 * @param request - Real data rows plus the model's chart-shape choices.
 * @returns The figure, or an explicit reason the request could not be charted.
 */
export function assembleKpiFigure(request: ChartRequest): ChartResult {
  const reason = validateChartRequest(request);
  if (reason) return { ok: false, reason };

  try {
    const figure = assemblePlotly({
      data: { values: request.rows },
      semantic_types: request.semanticTypes,
      chart_spec: {
        chartType: request.chartType,
        encodings: Object.fromEntries(
          Object.entries(request.encodings)
            .filter(([, field]) => Boolean(field))
            .map(([channel, field]) => [channel, { field }])
        ),
        baseSize: { width: request.width ?? 460, height: request.height ?? 240 },
      },
    }) as unknown as PlotlyFigure;

    if (!figure || !Array.isArray(figure.data)) {
      return { ok: false, reason: 'Chart compiler returned no traces.' };
    }
    return { ok: true, figure };
  } catch (error) {
    // Flint throws on unknown chart types and malformed encodings. Surface the
    // message rather than rendering an empty frame.
    return {
      ok: false,
      reason: error instanceof Error ? error.message : 'Chart assembly failed.',
    };
  }
}
