/**
 * WarningBanner Component
 * ========================
 *
 * Prominent yellow/amber banner used to surface API-reported warnings
 * (e.g., when the backend falls through to a degraded or fallback path
 * and populates a `warnings: string[]` field in the response).
 *
 * Rendering the warnings prevents the silent-warning shape where the
 * backend honestly self-reports a caveat but the frontend drops it.
 *
 * @module components/ui/WarningBanner
 */

import * as React from 'react';
import { AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface WarningBannerProps {
  /** Warning messages from the API response */
  messages: string[];
  /** Optional title override (defaults to "Analysis warnings") */
  title?: string;
  /** Additional CSS classes for the banner container */
  className?: string;
}

/**
 * Renders a list of warning messages in an amber banner.
 *
 * Returns `null` when `messages` is empty — callers can guard or
 * render unconditionally, the component decides.
 *
 * @example
 * ```tsx
 * <WarningBanner messages={response.warnings} />
 * ```
 */
export function WarningBanner({
  messages,
  title = 'Analysis warnings',
  className,
}: WarningBannerProps): React.ReactElement | null {
  if (!messages || messages.length === 0) {
    return null;
  }

  return (
    <div
      role="alert"
      data-testid="warning-banner"
      className={cn(
        'relative w-full rounded-lg border border-amber-300 bg-amber-50 p-4',
        'dark:border-amber-800 dark:bg-amber-950/30',
        className,
      )}
    >
      <div className="flex items-start gap-3">
        <AlertTriangle
          aria-hidden="true"
          className="h-5 w-5 flex-shrink-0 text-amber-600 dark:text-amber-400"
        />
        <div className="flex-1">
          <h3 className="mb-2 text-sm font-semibold text-amber-900 dark:text-amber-200">
            {title}
          </h3>
          <ul className="space-y-1 text-sm text-amber-800 dark:text-amber-300">
            {messages.map((message, idx) => (
              <li key={idx} className="leading-relaxed">
                {message}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default WarningBanner;
