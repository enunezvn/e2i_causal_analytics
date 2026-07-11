/**
 * LinkDialog — one-time display of an invite/recovery link with copy button.
 * The backend never stores the link; once this dialog closes it is gone.
 */

import { useState } from 'react';
import type { LinkResponse } from '@/types/admin';

interface LinkDialogProps {
  link: LinkResponse | null;
  onClose: () => void;
}

export function LinkDialog({ link, onClose }: LinkDialogProps) {
  const [copied, setCopied] = useState(false);
  if (!link) return null;

  const label = link.link_type === 'invite' ? 'Invite link' : 'Password recovery link';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-lg rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-6 shadow-lg">
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">
          {label} for {link.email}
        </h2>
        <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
          This link is shown once — copy it now and send it to the user directly.
        </p>
        <code className="mt-4 block break-all rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-3 text-xs text-[var(--color-foreground)]">
          {link.invite_link}
        </code>
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(link.invite_link);
                setCopied(true);
              } catch {
                setCopied(false);
              }
            }}
            className="rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white"
          >
            {copied ? 'Copied!' : 'Copy link'}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-foreground)]"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
