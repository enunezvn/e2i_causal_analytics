/**
 * Reset Password Page
 * ===================
 *
 * Completes the password recovery flow. This is the redirect target of
 * AuthProvider.resetPassword ({origin}/reset-password): the recovery email
 * link carries a token that supabase-js (detectSessionInUrl: true)
 * exchanges for a recovery session on page load, after which the existing
 * updatePassword action (supabase.auth.updateUser) can set the new password.
 *
 * If the user lands here without a valid recovery session (expired/used
 * link, direct navigation), updatePassword fails with a Supabase error
 * ("Auth session missing!") which is surfaced with an escape hatch back to
 * /forgot-password.
 *
 * The underlying update->re-login flow was proven against the public origin
 * (PUT /auth/v1/user then POST /auth/v1/token with the new password) on
 * fix/fe-serving-auth-build.
 *
 * @module pages/ResetPassword
 */

import * as React from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Link } from 'react-router-dom';
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

const resetPasswordSchema = z
  .object({
    password: z.string().min(8, 'Password must be at least 8 characters'),
    confirmPassword: z.string().min(1, 'Please confirm your password'),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type ResetPasswordFormData = z.infer<typeof resetPasswordSchema>;

// =============================================================================
// COMPONENT
// =============================================================================

export function ResetPassword() {
  const { updatePassword, isLoading, error, clearError } = useAuth();
  const [updated, setUpdated] = React.useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<ResetPasswordFormData>({
    resolver: zodResolver(resetPasswordSchema),
    defaultValues: { password: '', confirmPassword: '' },
  });

  // Clear auth error when leaving the page
  React.useEffect(() => {
    return () => {
      clearError();
    };
  }, [clearError]);

  const onSubmit = async (data: ResetPasswordFormData) => {
    try {
      await updatePassword(data.password);
      setUpdated(true);
    } catch {
      // Error is already set in the auth store
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

        <Card>
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center">
              {updated ? 'Password updated' : 'Choose a new password'}
            </CardTitle>
            <CardDescription className="text-center">
              {updated
                ? 'Your password has been changed successfully'
                : 'Enter and confirm your new password'}
            </CardDescription>
          </CardHeader>

          {updated ? (
            <CardContent className="space-y-4">
              <div className="p-3 rounded-md bg-[var(--color-primary)]/10 border border-[var(--color-primary)]/20">
                <p className="text-sm text-[var(--color-foreground)]">
                  You can now sign in with your new password.
                </p>
              </div>
              <p className="text-sm text-center">
                <Link
                  to="/login"
                  className="text-[var(--color-primary)] hover:underline"
                >
                  Sign in
                </Link>
              </p>
            </CardContent>
          ) : (
            <form noValidate onSubmit={handleSubmit(onSubmit)}>
              <CardContent className="space-y-4">
                {/* Auth Error (e.g. expired/missing recovery session) */}
                {error && (
                  <div className="p-3 rounded-md bg-[var(--color-destructive)]/10 border border-[var(--color-destructive)]/20">
                    <p className="text-sm text-[var(--color-destructive)]">
                      {error.message}
                    </p>
                    <p className="text-sm mt-2">
                      <Link
                        to="/forgot-password"
                        className="text-[var(--color-primary)] hover:underline"
                      >
                        Request a new link
                      </Link>
                    </p>
                  </div>
                )}

                {/* New Password */}
                <div className="space-y-2">
                  <Label htmlFor="password">New Password</Label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Enter your new password"
                    autoComplete="new-password"
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

                {/* Confirm Password */}
                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm Password</Label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    placeholder="Re-enter your new password"
                    autoComplete="new-password"
                    disabled={isLoading || isSubmitting}
                    {...register('confirmPassword')}
                    aria-invalid={errors.confirmPassword ? 'true' : 'false'}
                  />
                  {errors.confirmPassword && (
                    <p className="text-sm text-[var(--color-destructive)]">
                      {errors.confirmPassword.message}
                    </p>
                  )}
                </div>
              </CardContent>

              <CardFooter className="flex flex-col space-y-4">
                <Button
                  type="submit"
                  className="w-full"
                  disabled={isLoading || isSubmitting}
                >
                  {isLoading || isSubmitting ? (
                    <>
                      <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Updating...
                    </>
                  ) : (
                    'Update password'
                  )}
                </Button>
              </CardFooter>
            </form>
          )}
        </Card>
      </div>
    </div>
  );
}

export default ResetPassword;
