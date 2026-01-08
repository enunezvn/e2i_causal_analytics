/**
 * UserMenu Component
 * ==================
 *
 * Dropdown menu for authenticated user actions.
 * Shows user info and provides logout option.
 *
 * Features:
 * - User avatar/initials
 * - User email display
 * - Logout action
 * - Admin indicator (if applicable)
 *
 * Usage:
 *   import { UserMenu } from '@/components/auth'
 *   <UserMenu />
 *
 * @module components/auth/UserMenu
 */

import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/use-auth';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';

// =============================================================================
// ICONS
// =============================================================================

function UserIcon({ className = 'h-5 w-5' }: { className?: string }) {
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
        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
      />
    </svg>
  );
}

function LogOutIcon({ className = 'h-4 w-4' }: { className?: string }) {
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
        d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
      />
    </svg>
  );
}

function ShieldIcon({ className = 'h-4 w-4' }: { className?: string }) {
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
        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
      />
    </svg>
  );
}

// =============================================================================
// TYPES
// =============================================================================

export interface UserMenuProps {
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * UserMenu
 *
 * Dropdown menu showing user info and actions when authenticated.
 * Shows login button when not authenticated.
 */
export function UserMenu({ className = '' }: UserMenuProps) {
  const navigate = useNavigate();
  const { isAuthenticated, isAdmin, userInfo, logout, isLoading } = useAuth();

  // Handle logout
  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch {
      // Error is handled in auth store
    }
  };

  // Get user initials for avatar
  const getInitials = (name: string | null, email: string | null): string => {
    if (name) {
      return name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2);
    }
    if (email) {
      return email.charAt(0).toUpperCase();
    }
    return 'U';
  };

  // Not authenticated - show login button
  if (!isAuthenticated) {
    return (
      <Button
        variant="outline"
        size="sm"
        onClick={() => navigate('/login')}
        className={className}
      >
        Sign in
      </Button>
    );
  }

  const initials = getInitials(userInfo.name, userInfo.email);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={`
            inline-flex items-center justify-center gap-2
            rounded-md p-1.5
            text-[var(--color-muted)]
            hover:bg-[var(--color-secondary)]
            hover:text-[var(--color-foreground)]
            focus:outline-none focus:ring-2 focus:ring-[var(--color-ring)] focus:ring-offset-2
            ${className}
          `.trim()}
          aria-label="User menu"
        >
          {/* User Avatar */}
          {userInfo.avatarUrl ? (
            <img
              src={userInfo.avatarUrl}
              alt={userInfo.name ?? 'User'}
              className="h-8 w-8 rounded-full object-cover"
            />
          ) : (
            <div
              className="
                flex h-8 w-8 items-center justify-center
                rounded-full
                bg-[var(--color-primary)]
                text-[var(--color-primary-foreground)]
                text-sm font-medium
              "
            >
              {initials}
            </div>
          )}
          {/* User name - visible on larger screens */}
          <span className="hidden lg:inline-block text-sm font-medium max-w-[120px] truncate">
            {userInfo.name ?? userInfo.email}
          </span>
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-56">
        {/* User info */}
        <DropdownMenuLabel className="font-normal">
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium leading-none">
              {userInfo.name ?? 'User'}
            </p>
            <p className="text-xs leading-none text-[var(--color-muted-foreground)]">
              {userInfo.email}
            </p>
            {isAdmin && (
              <div className="flex items-center gap-1 mt-1">
                <ShieldIcon className="h-3 w-3 text-[var(--color-primary)]" />
                <span className="text-xs text-[var(--color-primary)]">Admin</span>
              </div>
            )}
          </div>
        </DropdownMenuLabel>

        <DropdownMenuSeparator />

        {/* Profile - future feature */}
        <DropdownMenuItem disabled className="cursor-not-allowed opacity-50">
          <UserIcon className="h-4 w-4 mr-2" />
          <span>Profile</span>
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        {/* Logout */}
        <DropdownMenuItem
          onClick={handleLogout}
          disabled={isLoading}
          className="text-[var(--color-destructive)] focus:text-[var(--color-destructive)] cursor-pointer"
        >
          <LogOutIcon className="h-4 w-4 mr-2" />
          <span>{isLoading ? 'Signing out...' : 'Sign out'}</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default UserMenu;
