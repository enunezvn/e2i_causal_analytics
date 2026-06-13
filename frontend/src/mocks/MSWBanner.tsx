/**
 * MSW Mock-Data Banner
 * ====================
 *
 * Persistent, non-dismissable banner shown whenever Mock Service Worker is
 * actively intercepting API requests. Before this banner, MSW announced
 * itself only via a console.info - easy to miss, and every API response on
 * screen (KPIs, causal effects, agent statuses) was plausible-but-simulated
 * pharma data with no visual cue.
 *
 * Reads window.__E2I_MSW_ACTIVE__, published by initMSW() (./browser.ts)
 * after the worker starts. This component intentionally imports NOTHING from
 * msw so that rendering it can never pull MSW code into a bundle; in
 * production builds the flag is never set (initMSW is DEV-only) and the
 * <MSWBanner /> mount in main.tsx is DEV-gated and tree-shaken away.
 */

export function MSWBanner() {
  if (typeof window === 'undefined' || window.__E2I_MSW_ACTIVE__ !== true) {
    return null;
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed bottom-0 inset-x-0 z-[9999] flex items-center justify-center gap-2 bg-amber-500 px-4 py-1.5 text-sm font-medium text-black shadow-[0_-1px_4px_rgba(0,0,0,0.2)]"
    >
      <span aria-hidden="true">⚠️</span>
      <span>
        Mock data mode: MSW is intercepting API requests — all data shown is
        simulated, not real. Set <code className="font-mono">VITE_MSW_ENABLED=false</code> to
        use the live backend.
      </span>
    </div>
  );
}

export default MSWBanner;
