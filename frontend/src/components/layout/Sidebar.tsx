/**
 * Sidebar Component
 * =================
 *
 * Navigation sidebar for the E2I Causal Analytics dashboard.
 * Contains navigation links to all dashboard sections.
 *
 * Usage:
 *   import { Sidebar } from '@/components/layout/Sidebar'
 *   <Sidebar />
 */

import { NavLink, useLocation } from 'react-router-dom';
import { useSidebarState } from '@/stores/ui-store';
import { getNavigationRoutes, type RouteConfig } from '@/router/routes';

/**
 * Sidebar props interface
 */
interface SidebarProps {
  className?: string;
}

/**
 * Icon component map - maps route icon names to SVG components
 */
function NavIcon({ icon, className = 'h-5 w-5' }: { icon?: string; className?: string }) {
  const iconMap: Record<string, JSX.Element> = {
    home: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
    'share-2': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
      </svg>
    ),
    'git-branch': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 3v12m0 0c0 2.761 2.239 5 5 5h2c2.761 0 5-2.239 5-5V9M6 15c0-2.761 2.239-5 5-5h2c2.761 0 5 2.239 5 5" />
      </svg>
    ),
    'bar-chart-2': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M18 20V10M12 20V4M6 20v-6" />
      </svg>
    ),
    layers: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
    'trending-up': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    target: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <circle cx="12" cy="12" r="10" />
        <circle cx="12" cy="12" r="6" />
        <circle cx="12" cy="12" r="2" />
      </svg>
    ),
    zap: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    'check-circle': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    activity: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
    monitor: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 21h8M12 17v4" />
      </svg>
    ),
    bot: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <rect x="3" y="11" width="18" height="10" rx="2" />
        <circle cx="12" cy="5" r="2" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v4M7 15h.01M17 15h.01" />
      </svg>
    ),
    'book-open': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    ),
    brain: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 1 7.92 12.446a9 9 0 1 1 -16.626 0a7.5 7.5 0 0 1 7.92 -12.446c.13 0 .261 0 .393 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v9m0 0l-3 -2m3 2l3 -2" />
      </svg>
    ),
    'flask-conical': (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 3h6M10 3v6.172a2 2 0 0 1-.586 1.414l-5.828 5.828a2 2 0 0 0-.586 1.414V20a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1.172a2 2 0 0 0-.586-1.414l-5.828-5.828a2 2 0 0 1-.586-1.414V3" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 15h12" />
      </svg>
    ),
    flask: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 3h6M10 3v6l-6 8v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2l-6-8V3" />
      </svg>
    ),
    calculator: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <rect x="4" y="2" width="16" height="20" rx="2" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 6h8M8 10h2M14 10h2M8 14h2M14 14h2M8 18h2M14 18h2" />
      </svg>
    ),
    users: (
      <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
  };

  return iconMap[icon ?? 'home'] ?? iconMap.home;
}

/**
 * Navigation item component
 */
function NavItem({ route, isCollapsed }: { route: RouteConfig; isCollapsed: boolean }) {
  const location = useLocation();
  const isActive = location.pathname === route.path;

  return (
    <NavLink
      to={route.path}
      className={`
        flex items-center gap-3 px-3 py-2 rounded-md
        text-sm font-medium
        transition-colors duration-150
        ${
          isActive
            ? 'bg-[var(--color-primary)] text-[var(--color-primary-foreground)]'
            : 'text-[var(--color-sidebar-foreground)] hover:bg-[var(--color-sidebar-hover)] hover:text-[var(--color-foreground)]'
        }
        ${isCollapsed ? 'justify-center' : ''}
      `}
      title={isCollapsed ? route.title : undefined}
      aria-current={isActive ? 'page' : undefined}
    >
      <NavIcon icon={route.icon} className="h-5 w-5 flex-shrink-0" />
      {!isCollapsed && <span className="truncate">{route.title}</span>}
    </NavLink>
  );
}

/**
 * Collapse toggle icon
 */
function CollapseIcon({ isCollapsed, className = 'h-5 w-5' }: { isCollapsed: boolean; className?: string }) {
  return (
    <svg
      className={`${className} transition-transform duration-200 ${isCollapsed ? 'rotate-180' : ''}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
    </svg>
  );
}

/**
 * Sidebar component
 *
 * Renders the navigation sidebar with:
 * - Navigation links to all dashboard sections
 * - Collapsible state for more screen space
 * - Active state highlighting
 * - Mobile-responsive behavior
 */
export function Sidebar({ className = '' }: SidebarProps) {
  const { isOpen, isCollapsed, setOpen, toggleCollapsed } = useSidebarState();
  const navigationRoutes = getNavigationRoutes();

  // Group routes by category
  const mainRoutes = navigationRoutes.slice(0, 1); // Home
  const analyticsRoutes = navigationRoutes.slice(1, 8); // Analytics sections
  const systemRoutes = navigationRoutes.slice(8); // System & Monitoring

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        id="sidebar"
        className={`
          fixed inset-y-0 left-0 z-40
          flex flex-col
          bg-[var(--color-sidebar)]
          border-r border-[var(--color-border)]
          transition-all duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          ${isCollapsed ? 'w-16' : 'w-64'}
          ${className}
        `}
        aria-label="Main navigation"
      >
        {/* Sidebar header */}
        <div className="flex h-16 items-center justify-between px-4 border-b border-[var(--color-border)]">
          {!isCollapsed && (
            <span className="text-lg font-semibold text-[var(--color-foreground)]">
              Navigation
            </span>
          )}
          {/* Collapse toggle - hidden on mobile */}
          <button
            type="button"
            onClick={toggleCollapsed}
            className="
              hidden lg:flex
              items-center justify-center
              rounded-md p-2
              text-[var(--color-muted)]
              hover:bg-[var(--color-sidebar-hover)]
              hover:text-[var(--color-foreground)]
              focus-ring
            "
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <CollapseIcon isCollapsed={isCollapsed} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-3">
          {/* Main section */}
          <div className="space-y-1">
            {mainRoutes.map((route) => (
              <NavItem key={route.path} route={route} isCollapsed={isCollapsed} />
            ))}
          </div>

          {/* Analytics section */}
          <div className="mt-6">
            {!isCollapsed && (
              <h3 className="px-3 mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--color-muted)]">
                Analytics
              </h3>
            )}
            <div className="space-y-1">
              {analyticsRoutes.map((route) => (
                <NavItem key={route.path} route={route} isCollapsed={isCollapsed} />
              ))}
            </div>
          </div>

          {/* System section */}
          <div className="mt-6">
            {!isCollapsed && (
              <h3 className="px-3 mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--color-muted)]">
                System
              </h3>
            )}
            <div className="space-y-1">
              {systemRoutes.map((route) => (
                <NavItem key={route.path} route={route} isCollapsed={isCollapsed} />
              ))}
            </div>
          </div>
        </nav>

        {/* Sidebar footer */}
        <div className="border-t border-[var(--color-border)] p-3">
          {!isCollapsed ? (
            <div className="text-xs text-[var(--color-muted)]">
              <p>E2I Causal Analytics</p>
              <p className="mt-1">v1.0.0</p>
            </div>
          ) : (
            <div className="flex justify-center">
              <span className="text-xs text-[var(--color-muted)]">v1</span>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
