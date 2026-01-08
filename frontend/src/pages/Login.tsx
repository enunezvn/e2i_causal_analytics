/**
 * Login Page
 * ==========
 *
 * Authentication page for user login.
 * Uses react-hook-form with Zod validation.
 *
 * Features:
 * - Email/password login form
 * - Form validation with helpful error messages
 * - Loading state during authentication
 * - Error display for auth failures
 * - Link to signup page
 * - Post-login redirect support
 *
 * @module pages/Login
 */

import * as React from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/hooks/use-auth';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '@/components/ui/card';

// =============================================================================
// VALIDATION SCHEMA
// =============================================================================

const loginSchema = z.object({
  email: z
    .string()
    .min(1, 'Email is required')
    .email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

// =============================================================================
// COMPONENT
// =============================================================================

export function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isLoading, error, clearError, isAuthenticated, redirectTo, setRedirectTo } =
    useAuth();

  // Get intended destination from location state or redirectTo
  const from = (location.state as { from?: string })?.from ?? redirectTo ?? '/';

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  // Form setup
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });

  // Clear error when component unmounts or form changes
  React.useEffect(() => {
    return () => {
      clearError();
    };
  }, [clearError]);

  // Handle form submission
  const onSubmit = async (data: LoginFormData) => {
    try {
      await login(data);
      // Clear redirect destination after successful login
      setRedirectTo(null);
      // Navigation happens via the useEffect above
    } catch {
      // Error is already set in auth store
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)] p-4">
      <div className="w-full max-w-md">
        {/* Logo/Brand */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-[var(--color-foreground)]">
            E2I Analytics
          </h1>
          <p className="text-[var(--color-muted-foreground)] mt-2">
            Pharmaceutical Commercial Analytics
          </p>
        </div>

        {/* Login Card */}
        <Card>
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center">Sign in</CardTitle>
            <CardDescription className="text-center">
              Enter your credentials to access the dashboard
            </CardDescription>
          </CardHeader>

          <form onSubmit={handleSubmit(onSubmit)}>
            <CardContent className="space-y-4">
              {/* Auth Error */}
              {error && (
                <div className="p-3 rounded-md bg-[var(--color-destructive)]/10 border border-[var(--color-destructive)]/20">
                  <p className="text-sm text-[var(--color-destructive)]">
                    {error.message}
                  </p>
                </div>
              )}

              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="user@example.com"
                  autoComplete="email"
                  disabled={isLoading || isSubmitting}
                  {...register('email')}
                  aria-invalid={errors.email ? 'true' : 'false'}
                />
                {errors.email && (
                  <p className="text-sm text-[var(--color-destructive)]">
                    {errors.email.message}
                  </p>
                )}
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="password">Password</Label>
                  <Link
                    to="/forgot-password"
                    className="text-sm text-[var(--color-primary)] hover:underline"
                    tabIndex={-1}
                  >
                    Forgot password?
                  </Link>
                </div>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  disabled={isLoading || isSubmitting}
                  {...register('password')}
                  aria-invalid={errors.password ? 'true' : 'false'}
                />
                {errors.password && (
                  <p className="text-sm text-[var(--color-destructive)]">
                    {errors.password.message}
                  </p>
                )}
              </div>
            </CardContent>

            <CardFooter className="flex flex-col space-y-4">
              {/* Submit Button */}
              <Button
                type="submit"
                className="w-full"
                disabled={isLoading || isSubmitting}
              >
                {isLoading || isSubmitting ? (
                  <>
                    <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Signing in...
                  </>
                ) : (
                  'Sign in'
                )}
              </Button>

              {/* Signup Link */}
              <p className="text-sm text-center text-[var(--color-muted-foreground)]">
                Don&apos;t have an account?{' '}
                <Link
                  to="/signup"
                  className="text-[var(--color-primary)] hover:underline font-medium"
                >
                  Sign up
                </Link>
              </p>
            </CardFooter>
          </form>
        </Card>

        {/* Footer */}
        <p className="text-center text-xs text-[var(--color-muted-foreground)] mt-8">
          &copy; {new Date().getFullYear()} E2I Causal Analytics. All rights reserved.
        </p>
      </div>
    </div>
  );
}

export default Login;
