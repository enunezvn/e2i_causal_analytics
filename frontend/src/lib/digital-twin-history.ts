/**
 * Digital-twin simulation history grouping.
 * =========================================
 *
 * Re-running the same intervention for the same brand produces multiple distinct
 * simulations (unique ids/timestamps, slightly different ATEs). They are NOT
 * duplicates, but a flat list of near-identical rows reads as duplicated.
 * {@link groupSimulationsByInterventionBrand} collapses runs of the same
 * (brand, intervention_type) into one group — the latest run + a count + the
 * full run list — so the history shows one row per (brand, intervention) with an
 * "N runs" affordance, while preserving every run (no data dropped).
 *
 * Pure function; no rendering, no I/O.
 *
 * @module lib/digital-twin-history
 */

/** Minimal shape the grouping needs from a history item. */
export interface SimLike {
  brand: string;
  intervention_type: string;
  created_at: string;
}

/** A (brand, intervention_type) group with its latest run + all runs. */
export interface SimGroup<T extends SimLike> {
  /** Stable group key: `${brand}|${intervention_type}`. */
  key: string;
  brand: string;
  intervention_type: string;
  /** Most recent run in the group (by created_at). */
  latest: T;
  /** Total runs in the group. */
  count: number;
  /** All runs, newest first. */
  runs: T[];
}

function parseTime(v: string): number {
  const t = Date.parse(v);
  return Number.isNaN(t) ? 0 : t;
}

/**
 * Group simulations by (brand, intervention_type), newest-first within each
 * group and across groups (ordered by each group's latest run).
 *
 * @param items history items (order-independent)
 * @returns one {@link SimGroup} per distinct (brand, intervention_type)
 */
export function groupSimulationsByInterventionBrand<T extends SimLike>(
  items: readonly T[]
): SimGroup<T>[] {
  const buckets = new Map<string, T[]>();
  for (const item of items) {
    const key = `${item.brand}|${item.intervention_type}`;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(item);
    else buckets.set(key, [item]);
  }

  const groups: SimGroup<T>[] = [];
  for (const [key, runs] of buckets) {
    const sorted = [...runs].sort((a, b) => parseTime(b.created_at) - parseTime(a.created_at));
    const latest = sorted[0];
    groups.push({
      key,
      brand: latest.brand,
      intervention_type: latest.intervention_type,
      latest,
      count: sorted.length,
      runs: sorted,
    });
  }

  // Most-recently-active group first.
  groups.sort((a, b) => parseTime(b.latest.created_at) - parseTime(a.latest.created_at));
  return groups;
}
