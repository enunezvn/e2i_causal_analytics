/**
 * ActivityTab — platform activity over time, per-user drill-down, and the
 * admin audit feed (security_audit_log).
 *
 * Data sources (all real): auth.audit_log_entries via SECURITY DEFINER RPCs
 * (login history since 2026-02), user_activity_log (per-minute API buckets,
 * accruing from this feature's middleware), chatbot_user_profiles counters.
 */

import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAuditFeed, usePlatformActivity, useUserActivity } from '@/hooks/api/use-admin';
import type { AdminUser } from '@/types/admin';

interface ActivityTabProps {
  users: AdminUser[];
}

export function ActivityTab({ users }: ActivityTabProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [days, setDays] = useState(90);

  const platform = usePlatformActivity(30);
  const userActivity = useUserActivity(selectedId, days);
  const audit = useAuditFeed(30);

  const loginsByDay = useMemo(() => {
    const events = userActivity.data?.auth_events ?? [];
    return events.filter((e) => e.event_type === 'login');
  }, [userActivity.data]);

  const apiByGroup = useMemo(() => {
    const rows = userActivity.data?.api_activity ?? [];
    const agg = new Map<string, number>();
    for (const row of rows) {
      agg.set(row.endpoint_group, (agg.get(row.endpoint_group) ?? 0) + row.request_count);
    }
    return Array.from(agg.entries())
      .map(([endpoint_group, requests]) => ({ endpoint_group, requests }))
      .sort((a, b) => b.requests - a.requests)
      .slice(0, 12);
  }, [userActivity.data]);

  const chat = userActivity.data?.chat;

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">
          Platform activity (30 days)
        </h2>
        <p className="mb-3 text-sm text-[var(--color-muted-foreground)]">
          Daily logins and distinct active users, from the auth audit log.
        </p>
        <div className="h-64 rounded-lg border border-[var(--color-border)] p-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={platform.data?.days ?? []}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="logins" stroke="#6366f1" name="Logins" dot={false} />
              <Line
                type="monotone"
                dataKey="active_users"
                stroke="#10b981"
                name="Active users"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section>
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-[var(--color-foreground)]">User activity</h2>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Login history, API usage, and chat engagement for one user.
            </p>
          </div>
          <div className="flex gap-2">
            <select
              aria-label="Select user"
              value={selectedId ?? ''}
              onChange={(e) => setSelectedId(e.target.value || null)}
              className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
            >
              <option value="">Select a user…</option>
              {users.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.email}
                </option>
              ))}
            </select>
            <select
              aria-label="Time range"
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
            >
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
              <option value={365}>1 year</option>
            </select>
          </div>
        </div>

        {!selectedId && (
          <p className="rounded-lg border border-dashed border-[var(--color-border)] p-6 text-center text-sm text-[var(--color-muted-foreground)]">
            Select a user to see their activity over time.
          </p>
        )}

        {selectedId && userActivity.data && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="rounded-lg border border-[var(--color-border)] p-4">
                <div className="text-xs uppercase text-[var(--color-muted-foreground)]">
                  Conversations
                </div>
                <div className="text-2xl font-semibold text-[var(--color-foreground)]">
                  {chat?.total_conversations ?? 0}
                </div>
              </div>
              <div className="rounded-lg border border-[var(--color-border)] p-4">
                <div className="text-xs uppercase text-[var(--color-muted-foreground)]">
                  Chat messages
                </div>
                <div className="text-2xl font-semibold text-[var(--color-foreground)]">
                  {chat?.total_messages ?? 0}
                </div>
              </div>
              <div className="rounded-lg border border-[var(--color-border)] p-4">
                <div className="text-xs uppercase text-[var(--color-muted-foreground)]">
                  Last chat activity
                </div>
                <div className="text-2xl font-semibold text-[var(--color-foreground)]">
                  {chat?.last_active_at ? new Date(chat.last_active_at).toLocaleDateString() : '—'}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div>
                <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
                  Logins per day
                </h3>
                <div className="h-52 rounded-lg border border-[var(--color-border)] p-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={loginsByDay}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                      <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                      <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Bar dataKey="event_count" fill="#6366f1" name="Logins" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
                  API requests by area
                </h3>
                <div className="h-52 rounded-lg border border-[var(--color-border)] p-3">
                  {apiByGroup.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={apiByGroup} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                        <YAxis
                          type="category"
                          dataKey="endpoint_group"
                          width={110}
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip />
                        <Bar dataKey="requests" fill="#10b981" name="Requests" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="pt-8 text-center text-sm text-[var(--color-muted-foreground)]">
                      No API activity recorded yet — tracking accrues from deployment of this
                      feature onward.
                    </p>
                  )}
                </div>
              </div>
            </div>

            <div>
              <h3 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
                Recent auth events
              </h3>
              <ul className="divide-y divide-[var(--color-border)] rounded-lg border border-[var(--color-border)] text-sm">
                {(userActivity.data.recent_events ?? []).map((e, i) => (
                  <li key={i} className="flex justify-between px-4 py-2">
                    <span className="text-[var(--color-foreground)]">{e.action}</span>
                    <span className="text-[var(--color-muted-foreground)]">
                      {new Date(e.occurred_at).toLocaleString()}
                    </span>
                  </li>
                ))}
                {!userActivity.data.recent_events?.length && (
                  <li className="px-4 py-2 text-[var(--color-muted-foreground)]">No events.</li>
                )}
              </ul>
            </div>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">Admin audit</h2>
        <p className="mb-3 text-sm text-[var(--color-muted-foreground)]">
          Administrative actions recorded in the security audit log (30 days).
        </p>
        <ul className="divide-y divide-[var(--color-border)] rounded-lg border border-[var(--color-border)] text-sm">
          {(audit.data?.events ?? []).map((e) => (
            <li key={e.event_id} className="flex flex-wrap justify-between gap-2 px-4 py-2">
              <span className="text-[var(--color-foreground)]">{e.message}</span>
              <span className="text-[var(--color-muted-foreground)]">
                {e.user_email ?? 'system'} · {new Date(e.timestamp).toLocaleString()}
              </span>
            </li>
          ))}
          {!audit.data?.events?.length && (
            <li className="px-4 py-2 text-[var(--color-muted-foreground)]">
              No admin actions recorded yet.
            </li>
          )}
        </ul>
      </section>
    </div>
  );
}
