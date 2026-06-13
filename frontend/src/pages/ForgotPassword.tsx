/**
 * Forgot Password Page
 * ====================
 *
 * Requests a password recovery email via the AuthProvider resetPassword
 * action (supabase.auth.resetPasswordForEmail). The recovery email links the
 * user back to {origin}/reset-password, where ResetPassword completes the
 * flow with updatePassword.
 *
 * History: the login page has linked /forgot-password since the auth feature
 * landed (031ec2db) and AuthProvider shipped the resetPassword action, but
 * this page was never created - the link 404'd to NotFound until
 * fix/fe-serving-auth-build.
 *
 * NOTE: actually delivering the recovery email requires SMTP to be configured
 * on the self-hosted Supabase (GoTrue). If it is not, Supabase returns
 * "Error sending recovery email", which this page surfaces honestly.
 *
 * @module pages/ForgotPassword
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

const forgotPasswordSchema = z.object({
  email: z
    .string()
    .min(1, 'Email is required')
    .email('Please enter a valid email address'),
});

type ForgotPasswordFormData = z.infer<typeof forgotPasswordSchema>;

// =============================================================================
// COMPONENT
// =============================================================================

export function ForgotPassword() {
  const { resetPassword, isLoading, error, clearError } = useAuth();
  const [submittedEmail, setSubmittedEmail] = React.useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<ForgotPasswordFormData>({
    resolver: zodResolver(forgotPasswordSchema),
    defaultValues: { email: '' },
  });

  // Clear auth error when leaving the page
  React.useEffect(() => {
    return () => {
      clearError();
    };
  }, [clearError]);

  const onSubmit = async (data: ForgotPasswordFormData) => {
    try {
      await resetPassword(data.email);
      setSubmittedEmail(data.email);
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
            <CardTitle className="text-2xl text-center">Reset your password</CardTitle>
            <CardDescription className="text-center">
              {submittedEmail
                ? 'Recovery email sent'
                : 'Enter your email and we will send you a reset link'}
            </CardDescription>
          </CardHeader>

          {submittedEmail ? (
            <CardContent className="space-y-4">
              <div className="p-3 rounded-md bg-[var(--color-primary)]/10 border border-[var(--color-primary)]/20">
                <p className="text-sm text-[var(--color-foreground)]">
                  Check your email: if an account exists for{' '}
                  <span className="font-medium">{submittedEmail}</span>, a password
                  reset link is on its way. The link opens the reset page in this
                  app.
                </p>
              </div>
              <p className="text-sm text-center text-[var(--color-muted-foreground)]">
                <Link to="/login" className="text-[var(--color-primary)] hover:underline">
                  Back to sign in
                </Link>
              </p>
            </CardContent>
          ) : (
            <form noValidate onSubmit={handleSubmit(onSubmit)}>
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
                      Sending...
                    </>
                  ) : (
                    'Send reset link'
                  )}
                </Button>

                <p className="text-sm text-center text-[var(--color-muted-foreground)]">
                  <Link
                    to="/login"
                    className="text-[var(--color-primary)] hover:underline"
                  >
                    Back to sign in
                  </Link>
                </p>
              </CardFooter>
            </form>
          )}
        </Card>
      </div>
    </div>
  );
}

export default ForgotPassword;
