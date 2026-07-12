/**
 * ObservabilityTab — LLM model, tokens, and $ cost per user / per session
 * (chat), plus platform/unattributed spend (background generation and anonymous
 * chat). GET /api/admin/observability/llm-usage.
 *
 * Honesty rules (spec 2026-07-12): attribution is chat-only (everything else
 * is the Platform section); pre-feature history is "untracked", never
 * estimated; unpriced models render "—", never $0.
 */

import { Fragment, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useLlmUsage } from '@/hooks/api/use-admin';

const fmtInt = (n: number) => n.toLocaleString();
const fmtCost = (n: number | null | undefined) => {
  if (n == null) return '—';
  if (n === 0) return '$0';
  if (n < 0.0001) return '<$0.0001';
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
};

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] p-4">
      <div className="text-xs uppercase text-[var(--color-muted-foreground)]">{label}</div>
      <div className="text-2xl font-semibold text-[var(--color-foreground)]">{value}</div>
    </div>
  );
}

function ModelChips({ models }: { models: string[] }) {
  return (
    <span className="flex flex-wrap gap-1">
      {models.map((m) => (
        <span
          key={m}
          className="rounded border border-[var(--color-border)] px-1.5 py-0.5 text-xs text-[var(--color-muted-foreground)]"
        >
          {m}
        </span>
      ))}
    </span>
  );
}

const TH = 'px-3 py-2 text-left text-xs font-medium uppercase text-[var(--color-muted-foreground)]';
const TD = 'px-3 py-2 text-sm text-[var(--color-foreground)]';

export function ObservabilityTab() {
  const [days, setDays] = useState(30);
  const [expandedUser, setExpandedUser] = useState<string | null>(null);
  const { data, isLoading, isError } = useLlmUsage(days);

  if (isLoading) {
    return (
      <p className="p-6 text-sm text-[var(--color-muted-foreground)]">Loading LLM usage…</p>
    );
  }
  if (isError || !data) {
    return (
      <p className="p-6 text-sm text-[var(--color-muted-foreground)]">
        Failed to load LLM usage.
      </p>
    );
  }

  const { summary, daily, by_user, sessions, platform, unpriced_models } = data;
  const windowStart = Date.now() - days * 86_400_000;
  const trackingLater =
    summary.tracking_since && new Date(summary.tracking_since).getTime() > windowStart;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-[var(--color-foreground)]">
            LLM observability
          </h2>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Models used, tokens consumed, and cost — per user and per chat session. Costs are
            computed from the pricing table (v{data.pricing_version}).
          </p>
        </div>
        <select
          aria-label="Time range"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
        >
          <option value={7}>7 days</option>
          <option value={30}>30 days</option>
          <option value={90}>90 days</option>
        </select>
      </div>

      {trackingLater && (
        <p className="rounded-lg border border-[var(--color-border)] bg-[var(--color-muted)] px-4 py-2 text-sm text-[var(--color-muted-foreground)]">
          Usage tracking began {new Date(summary.tracking_since as string).toLocaleDateString()};
          earlier sessions are untracked.
        </p>
      )}

      {unpriced_models.length > 0 && (
        <p className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
          Models missing from the pricing table (cost shown as —): {unpriced_models.join(', ')}
        </p>
      )}

      {summary.calls === 0 ? (
        <p className="rounded-lg border border-dashed border-[var(--color-border)] p-6 text-center text-sm text-[var(--color-muted-foreground)]">
          No LLM usage recorded in this window.
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Total cost" value={fmtCost(summary.total_cost_usd)} />
            <StatCard
              label="Tokens (in / out)"
              value={`${fmtInt(summary.input_tokens)} / ${fmtInt(summary.output_tokens)}`}
            />
            <StatCard label="LLM calls" value={fmtInt(summary.calls)} />
            <StatCard label="Active users" value={fmtInt(summary.distinct_users)} />
          </div>

          <section>
            <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
              Cost per day — chat vs platform
            </h3>
            <div className="h-64 rounded-lg border border-[var(--color-border)] p-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={daily}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmtCost(v as number | undefined)} />
                  <Legend />
                  <Bar dataKey="chat_cost_usd" stackId="c" fill="#6366f1" name="Chat" />
                  <Bar dataKey="platform_cost_usd" stackId="c" fill="#10b981" name="Platform" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section>
            <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
              Usage by user (chat)
            </h3>
            <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[var(--color-border)]">
                    <th className={TH}>User</th>
                    <th className={TH}>Sessions</th>
                    <th className={TH}>Calls</th>
                    <th className={TH}>Tokens (in / out)</th>
                    <th className={TH}>Cost</th>
                    <th className={TH}>Models</th>
                  </tr>
                </thead>
                <tbody>
                  {by_user.map((u) => (
                    <Fragment key={u.user_id}>
                      <tr className="border-b border-[var(--color-border)]">
                        <td className={TD}>
                          <button
                            type="button"
                            aria-expanded={expandedUser === u.user_id}
                            onClick={() =>
                              setExpandedUser(expandedUser === u.user_id ? null : u.user_id)
                            }
                            className="font-medium text-[var(--color-primary)]"
                          >
                            {expandedUser === u.user_id ? '▾ ' : '▸ '}
                            {u.email ?? u.user_id}
                          </button>
                        </td>
                        <td className={TD}>{u.sessions}</td>
                        <td className={TD}>{fmtInt(u.calls)}</td>
                        <td className={TD}>
                          {fmtInt(u.input_tokens)} / {fmtInt(u.output_tokens)}
                        </td>
                        <td className={TD}>{fmtCost(u.cost_usd)}</td>
                        <td className={TD}>
                          <ModelChips models={u.models} />
                        </td>
                      </tr>
                      {expandedUser === u.user_id &&
                        (sessions[u.user_id] ?? []).map((s) => (
                          <tr key={s.session_id} className="border-b border-[var(--color-border)] bg-[var(--color-muted)]">
                            <td className={`${TD} pl-8`}>
                              {s.title ?? s.session_id.split('~')[1] ?? s.session_id}
                              <span className="ml-2 text-xs text-[var(--color-muted-foreground)]">
                                {s.started_at
                                  ? new Date(s.started_at).toLocaleString()
                                  : 'start unknown'}
                              </span>
                            </td>
                            <td className={TD}>—</td>
                            <td className={TD}>{fmtInt(s.calls)}</td>
                            <td className={TD}>
                              {fmtInt(s.input_tokens)} / {fmtInt(s.output_tokens)}
                            </td>
                            <td className={TD}>{fmtCost(s.cost_usd)}</td>
                            <td className={TD}>
                              <ModelChips models={s.models} />
                            </td>
                          </tr>
                        ))}
                    </Fragment>
                  ))}
                  {by_user.length === 0 && (
                    <tr>
                      <td colSpan={6} className={`${TD} text-center text-[var(--color-muted-foreground)]`}>
                        No attributed chat usage in this window.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h3 className="mb-1 text-sm font-medium text-[var(--color-foreground)]">
              Platform &amp; unattributed LLM usage
            </h3>
            <p className="mb-2 text-sm text-[var(--color-muted-foreground)]">
              Background platform spend (insights, agents, generation) plus anonymous chat
              sessions — spend not attributable to a signed-in user.
            </p>
            <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[var(--color-border)]">
                    <th className={TH}>Surface</th>
                    <th className={TH}>Component</th>
                    <th className={TH}>Model</th>
                    <th className={TH}>Calls</th>
                    <th className={TH}>Tokens (in / out)</th>
                    <th className={TH}>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {platform.map((p) => (
                    <tr
                      key={`${p.surface}|${p.component}|${p.model}`}
                      className="border-b border-[var(--color-border)]"
                    >
                      <td className={TD}>{p.surface}</td>
                      <td className={TD}>{p.component ?? '—'}</td>
                      <td className={TD}>{p.model}</td>
                      <td className={TD}>{fmtInt(p.calls)}</td>
                      <td className={TD}>
                        {fmtInt(p.input_tokens)} / {fmtInt(p.output_tokens)}
                      </td>
                      <td className={TD}>{fmtCost(p.cost_usd)}</td>
                    </tr>
                  ))}
                  {platform.length === 0 && (
                    <tr>
                      <td colSpan={6} className={`${TD} text-center text-[var(--color-muted-foreground)]`}>
                        No platform LLM usage in this window.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
