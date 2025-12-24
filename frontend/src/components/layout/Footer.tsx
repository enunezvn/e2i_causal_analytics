/**
 * Footer Component
 * ================
 *
 * Application footer for the E2I Causal Analytics dashboard.
 * Contains copyright, version info, and helpful links.
 *
 * Usage:
 *   import { Footer } from '@/components/layout/Footer'
 *   <Footer />
 */

import { Link } from 'react-router-dom';

/**
 * Footer props interface
 */
interface FooterProps {
  className?: string;
}

/**
 * External link icon
 */
function ExternalLinkIcon({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
      />
    </svg>
  );
}

/**
 * Footer component
 *
 * Renders the application footer with:
 * - Copyright information
 * - Quick links for navigation
 * - API documentation link
 * - Version information
 */
export function Footer({ className = '' }: FooterProps) {
  const currentYear = new Date().getFullYear();

  return (
    <footer
      className={`
        border-t
        bg-[var(--color-background)]
        border-[var(--color-border)]
        ${className}
      `}
      role="contentinfo"
    >
      <div className="px-4 py-6 lg:px-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          {/* Copyright and brand */}
          <div className="flex flex-col gap-1">
            <p className="text-sm text-[var(--color-foreground)]">
              <span className="font-semibold">E2I Causal Analytics</span>
            </p>
            <p className="text-xs text-[var(--color-muted)]">
              &copy; {currentYear} All rights reserved.
            </p>
          </div>

          {/* Quick links */}
          <nav
            className="flex flex-wrap items-center gap-4 text-sm"
            aria-label="Footer navigation"
          >
            <Link
              to="/"
              className="text-[var(--color-muted)] hover:text-[var(--color-foreground)] transition-colors"
            >
              Dashboard
            </Link>
            <Link
              to="/system-health"
              className="text-[var(--color-muted)] hover:text-[var(--color-foreground)] transition-colors"
            >
              System Status
            </Link>
            <a
              href="/api/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="
                inline-flex items-center gap-1
                text-[var(--color-muted)]
                hover:text-[var(--color-foreground)]
                transition-colors
              "
            >
              API Docs
              <ExternalLinkIcon className="h-3 w-3" />
            </a>
          </nav>

          {/* Version info */}
          <div className="flex items-center gap-2 text-xs text-[var(--color-muted)]">
            <span className="inline-flex items-center gap-1">
              <span
                className="h-2 w-2 rounded-full bg-[var(--color-success)]"
                aria-hidden="true"
              />
              <span className="sr-only">Status:</span>
              Online
            </span>
            <span className="hidden sm:inline">|</span>
            <span className="hidden sm:inline">Version 1.0.0</span>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
