/**
 * SHAP encoded-feature → raw-covariate grouping.
 * ==============================================
 *
 * The gold-standard cohort models score an ENCODED feature vector produced by a
 * bundled FeatureBuilder, so SHAP runs over encoded columns, not the raw
 * covariates a user reasons about:
 *
 *   - a numeric covariate `X`      → `X` (bare) + `X__isna` (missingness twin)
 *   - a categorical covariate `X`  → one-hot `X_<value>` columns (incl `X_nan`)
 *
 * Surfaced raw, a single covariate reads as many near-duplicate rows
 * ("geographic region" ×5; an `X` / `X isna` twin). {@link groupByCovariate}
 * folds the encoded columns back to their parent covariate — importance summed
 * across the children (the standard way to report a one-hot categorical's total
 * SHAP importance) — so the ranking shows one row per real covariate, expandable
 * to the per-category detail. Raw SHAP values are never altered; this is a
 * presentation transform only.
 *
 * @module lib/shap-covariates
 */

/** Minimal numeric view of a feature the grouping needs. */
export interface Groupable {
  /** Encoded feature name (e.g. `geographic_region_west`, `disease_severity__isna`). */
  name: string;
  /** Importance magnitude (mean |SHAP| for cohort, |SHAP| for an individual). */
  abs: number;
  /** Signed contribution (mean signed SHAP for cohort, signed SHAP for an individual). */
  signed: number;
}

/** A raw covariate with its encoded children folded in. */
export interface CovariateGroup<T> {
  /** Parent raw covariate name, or the encoded name when it matched no covariate. */
  covariate: string;
  /** Total importance = sum of |children|. */
  importance: number;
  /** Net signed effect = sum of children's signed contributions. */
  signed: number;
  /** Direction of the net signed effect. */
  direction: 'positive' | 'negative';
  /** Rank by importance (1 = most important). */
  rank: number;
  /** Original child features, sorted by |contribution| desc (the expandable detail). */
  categories: T[];
  /**
   * True when this row aggregates more than a bare covariate — i.e. it has >1
   * encoded child, or its single child's encoded name differs from the parent
   * (a one-hot category). The UI shows an expand control only for these.
   */
  isGrouped: boolean;
}

/**
 * Resolve the parent raw covariate for an ENCODED feature name.
 *
 * A column `c` belongs to covariate `k` when `c === k` (bare numeric),
 * `c === k + '__isna'` (numeric missingness twin), or `c` starts with `k + '_'`
 * (a one-hot category). When several covariates match (e.g. `region` and
 * `region_code` both prefix `region_code_x`), the LONGEST is chosen so the most
 * specific covariate wins. Returns `null` when nothing matches.
 */
export function parentCovariate(
  encodedName: string,
  keepColumns: readonly string[]
): string | null {
  let best: string | null = null;
  for (const k of keepColumns) {
    const matches =
      encodedName === k ||
      encodedName === `${k}__isna` ||
      encodedName.startsWith(`${k}_`);
    if (matches && (best === null || k.length > best.length)) {
      best = k;
    }
  }
  return best;
}

/**
 * Group encoded features by their parent raw covariate.
 *
 * @param features    encoded features (any shape; `accessor` projects the numbers)
 * @param keepColumns raw covariate names from the model; when empty/absent the
 *                    result is a flat passthrough (one ungrouped row per feature)
 * @param accessor    projects a feature to `{ name, abs, signed }`
 * @returns covariate groups ranked by importance desc (1-based `rank`)
 */
export function groupByCovariate<T>(
  features: readonly T[],
  keepColumns: readonly string[] | null | undefined,
  accessor: (f: T) => Groupable
): CovariateGroup<T>[] {
  const keep = keepColumns ?? [];

  // Bucket children under their parent covariate (or their own name when no
  // covariate list is available / nothing matches — a graceful flat fallback).
  const buckets = new Map<string, T[]>();
  const order: string[] = [];
  for (const f of features) {
    const { name } = accessor(f);
    const parent = (keep.length ? parentCovariate(name, keep) : null) ?? name;
    let bucket = buckets.get(parent);
    if (!bucket) {
      bucket = [];
      buckets.set(parent, bucket);
      order.push(parent);
    }
    bucket.push(f);
  }

  const groups: CovariateGroup<T>[] = order.map((covariate) => {
    const children = [...buckets.get(covariate)!].sort(
      (a, b) => accessor(b).abs - accessor(a).abs
    );
    let importance = 0;
    let signed = 0;
    for (const c of children) {
      const g = accessor(c);
      importance += g.abs;
      signed += g.signed;
    }
    const onlyChild = children.length === 1 ? accessor(children[0]).name : null;
    return {
      covariate,
      importance,
      signed,
      direction: signed >= 0 ? 'positive' : 'negative',
      rank: 0, // assigned after sort
      categories: children,
      isGrouped: children.length > 1 || (onlyChild !== null && onlyChild !== covariate),
    };
  });

  groups.sort((a, b) => b.importance - a.importance);
  groups.forEach((g, i) => {
    g.rank = i + 1;
  });
  return groups;
}
