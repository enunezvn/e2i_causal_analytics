/**
 * AcceptInvite Page (PUBLIC route /accept-invite)
 * ===============================================
 *
 * Invite links minted by /api/admin/users/invite land here:
 *   /accept-invite?token_hash=<hashed_token>
 * Flow: verifyOtp({type:'invite'}) -> session -> set password -> enter app.
 * Admin-generated recovery links reuse this page: verifyOtp falls back to
 * type:'recovery' when the invite type is rejected.
 */

import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { supabase } from '@/lib/supabase';

type Phase = 'verifying' | 'set-password' | 'done' | 'error';

export default function AcceptInvite() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tokenHash = searchParams.get('token_hash');

  const [phase, setPhase] = useState<Phase>(tokenHash ? 'verifying' : 'error');
  const [error, setError] = useState<string | null>(
    tokenHash ? null : 'Invalid invite link — no token found. Ask your admin for a new link.'
  );
  const [email, setEmail] = useState<string | null>(null);
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!tokenHash) return;
    let cancelled = false;
    (async () => {
      // Invite links use type 'invite'; admin recovery links reuse this page
      // with type 'recovery' — try invite first, fall back.
      let result = await supabase.auth.verifyOtp({ type: 'invite', token_hash: tokenHash });
      if (result.error) {
        result = await supabase.auth.verifyOtp({ type: 'recovery', token_hash: tokenHash });
      }
      if (cancelled) return;
      if (result.error || !result.data.session) {
        setError(result.error?.message ?? 'Verification failed');
        setPhase('error');
        return;
      }
      setEmail(result.data.user?.email ?? null);
      setPhase('set-password');
    })();
    return () => {
      cancelled = true;
    };
  }, [tokenHash]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError('Passwords do not match');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    setError(null);
    setSubmitting(true);
    const { error: updateError } = await supabase.auth.updateUser({ password });
    setSubmitting(false);
    if (updateError) {
      setError(updateError.message);
      return;
    }
    setPhase('done');
    navigate('/', { replace: true });
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--color-background)] px-4">
      <div className="w-full max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-8 shadow-sm">
        <h1 className="mb-2 text-2xl font-semibold text-[var(--color-foreground)]">
          Welcome to E2I Analytics
        </h1>

        {phase === 'verifying' && (
          <p className="text-[var(--color-muted-foreground)]">Verifying your invite…</p>
        )}

        {phase === 'error' && (
          <div
            role="alert"
            className="mt-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
          >
            {error}
          </div>
        )}

        {phase === 'set-password' && (
          <form onSubmit={handleSubmit} className="mt-4 space-y-4">
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {email ? `Signed in as ${email}. ` : ''}Choose a password to finish setting up
              your account.
            </p>
            <div>
              <label
                htmlFor="new-password"
                className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
              >
                New password
              </label>
              <input
                id="new-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
              />
            </div>
            <div>
              <label
                htmlFor="confirm-password"
                className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
              >
                Confirm password
              </label>
              <input
                id="confirm-password"
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                required
                autoComplete="new-password"
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
              />
            </div>
            {error && (
              <div
                role="alert"
                className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
              >
                {error}
              </div>
            )}
            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {submitting ? 'Saving…' : 'Set password'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
