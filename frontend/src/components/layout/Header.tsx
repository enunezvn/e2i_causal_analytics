/**
 * Header Component
 * ================
 *
 * Top navigation bar for the E2I Causal Analytics dashboard.
 * Includes logo, mobile menu toggle, and user actions.
 *
 * Usage:
 *   import { Header } from '@/components/layout/Header'
 *   <Header />
 */

import { useLocation, Link } from 'react-router-dom';
import { useSidebarState } from '@/stores/ui-store';
import { getRouteConfig } from '@/router/routes';
import { UserMenu } from '@/components/auth';

/**
 * Header props interface
 */
interface HeaderProps {
  className?: string;
}

/**
 * Menu icon component (hamburger)
 */
function MenuIcon({ className = 'h-6 w-6' }: { className?: string }) {
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
        d="M4 6h16M4 12h16M4 18h16"
      />
    </svg>
  );
}

/**
 * Close icon component (X)
 */
function CloseIcon({ className = 'h-6 w-6' }: { className?: string }) {
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
        d="M6 18L18 6M6 6l12 12"
      />
    </svg>
  );
}

/**
 * Bell icon for notifications
 */
function BellIcon({ className = 'h-5 w-5' }: { className?: string }) {
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
        d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
      />
    </svg>
  );
}


/**
 * Header component
 *
 * Renders the top navigation bar with:
 * - Logo/brand on the left
 * - Mobile menu toggle
 * - Current page title (breadcrumb)
 * - User actions (notifications, profile) on the right
 */
export function Header({ className = '' }: HeaderProps) {
  const location = useLocation();
  const { isOpen, toggle } = useSidebarState();

  // Get current route configuration for title
  const currentRoute = getRouteConfig(location.pathname);
  const pageTitle = currentRoute?.title ?? 'Dashboard';

  return (
    <header
      className={`
        sticky top-0 z-40
        flex h-16 items-center justify-between
        border-b px-4 lg:px-6
        bg-[var(--color-background)]
        border-[var(--color-border)]
        ${className}
      `.trim()}
      role="banner"
    >
      {/* Left section: Logo and menu toggle */}
      <div className="flex items-center gap-4">
        {/* Mobile menu toggle */}
        <button
          type="button"
          onClick={toggle}
          className="
            lg:hidden
            inline-flex items-center justify-center
            rounded-md p-2
            text-[var(--color-muted)]
            hover:bg-[var(--color-secondary)]
            hover:text-[var(--color-foreground)]
            focus-ring
          "
          aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}
          aria-expanded={isOpen}
          aria-controls="sidebar"
        >
          {isOpen ? <CloseIcon /> : <MenuIcon />}
        </button>

        {/* Logo / Brand */}
        <Link
          to="/"
          className="flex items-center gap-2 font-semibold text-lg"
          aria-label="E2I Causal Analytics Home"
        >
          {/* Logo icon */}
          <div
            className="
              flex h-8 w-8 items-center justify-center
              rounded-lg
              bg-[var(--color-primary)]
              text-[var(--color-primary-foreground)]
            "
            aria-hidden="true"
          >
            <svg
              className="h-5 w-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <span className="hidden sm:inline-block text-[var(--color-foreground)]">
            E2I Analytics
          </span>
        </Link>

        {/* Page title / Breadcrumb - visible on larger screens */}
        <div className="hidden md:flex items-center">
          <span className="text-[var(--color-muted)]">/</span>
          <span className="ml-2 text-sm font-medium text-[var(--color-foreground)]">
            {pageTitle}
          </span>
        </div>
      </div>

      {/* Right section: Actions */}
      <div className="flex items-center gap-2">
        {/* Notifications button */}
        <button
          type="button"
          className="
            inline-flex items-center justify-center
            rounded-md p-2
            text-[var(--color-muted)]
            hover:bg-[var(--color-secondary)]
            hover:text-[var(--color-foreground)]
            focus-ring
          "
          aria-label="View notifications"
        >
          <BellIcon />
          {/* Notification indicator dot */}
          <span className="sr-only">You have unread notifications</span>
        </button>

        {/* User menu */}
        <UserMenu />
      </div>
    </header>
  );
}

export default Header;
