/**
 * ResolveForm tests — REAL TanStack hooks, only the API module mocked.
 *
 * Pins the once-per-review-id auto-assessment under StrictMode (double-invoked
 * effects) INCLUDING delivery of the result to the form, across an
 * unmount/remount, the guard RELEASE on an automatic failure (the hook-level
 * onError is run by the Mutation itself, so it fires even after the form has
 * unmounted — a collapsed queue row — unlike per-call mutate callbacks), and
 * guard OWNERSHIP: a failed MANUAL request never releases an id another form's
 * automatic request still holds.
 */
import { StrictMode } from 'react';
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ResolveForm } from './ResolveForm';
import type { ResolveFormProps } from './ResolveForm';
import type { AgentAssessment, AgentAssessmentResponse, PendingReviewItem } from '@/types/expert-review';

vi.mock('@/api/expert-review', () => ({
  generateReviewAssessment: vi.fn(),
  resolveReview: vi.fn(),
  getExpertReview: vi.fn(),
  getPendingReviews: vi.fn(),
  getReviewSummary: vi.fn(),
}));
import { generateReviewAssessment } from '@/api/expert-review';

const api = vi.mocked(generateReviewAssessment);

const REVIEW: PendingReviewItem = { review_id: 'rev-1', brand: 'Kisqali', treatment_variable: 't', outcome_variable: 'y' };
const ASSESSMENT: AgentAssessment = { items: [], is_fallback: true };
const RESPONSE: AgentAssessmentResponse = { review_id: 'rev-1', assessment: ASSESSMENT, cached: false, persisted: true };

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function newGuard() {
  return { current: new Set<string>() };
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
}

// Relies on REAL timers: Node orders the form's coerced zero-delay timer before this
// 10 ms wait. Do not add vi.useFakeTimers to this file.
/** Let any deferred auto-assessment timer fire so a "still N calls" assertion is not vacuous. */
const flushTimers = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 10)));

function renderForm(props: Partial<ResolveFormProps> = {}, queryClient = newClient()) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <StrictMode>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </StrictMode>
  );
  return render(<ResolveForm review={REVIEW} onClose={() => undefined} {...props} />, { wrapper });
}

beforeEach(() => {
  // mockReset (not clear): a test that aborts early must not leak its `Once` queue.
  api.mockReset();
});

describe('ResolveForm auto-assessment (real hooks, StrictMode)', () => {
  it('generates exactly once on mount with a shared guard, and the result reaches the form', async () => {
    api.mockResolvedValue(RESPONSE);
    const guard = newGuard();
    renderForm({ autoAssessGuard: guard });
    // "Regenerate" + enabled = the mutation result was delivered (not orphaned by StrictMode's resubscribe).
    const button = await screen.findByRole('button', { name: /regenerate agent assessment/i });
    expect(button).toBeEnabled();
    expect(api).toHaveBeenCalledTimes(1);
    expect(api.mock.calls[0][0]).toBe('rev-1');
    expect(api.mock.calls[0][1]).toBeFalsy();
    expect(guard.current.has('rev-1')).toBe(true);
  });

  it('generates exactly once with its own local guard (no page guard)', async () => {
    api.mockResolvedValue(RESPONSE);
    renderForm();
    await screen.findByRole('button', { name: /regenerate agent assessment/i });
    expect(api).toHaveBeenCalledTimes(1);
  });

  it('does not generate again after an unmount and remount with the same guard (success path)', async () => {
    api.mockResolvedValue(RESPONSE);
    const guard = newGuard();
    const { unmount } = renderForm({ autoAssessGuard: guard });
    await screen.findByRole('button', { name: /regenerate agent assessment/i });
    unmount();
    renderForm({ autoAssessGuard: guard });
    // The remounted form has no cache yet (the page refetches it) and offers a manual Generate.
    await screen.findByRole('button', { name: /^generate agent assessment$/i });
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1);
  });

  it('releases the guard when the AUTOMATIC generation fails after the form unmounted, so a remount retries once', async () => {
    const first = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(first.promise).mockResolvedValueOnce(RESPONSE);
    const guard = newGuard();
    const { unmount } = renderForm({ autoAssessGuard: guard });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(guard.current.has('rev-1')).toBe(true); // positive control for the release below
    unmount(); // the row collapsed while the request was in flight
    first.reject(new Error('LM unavailable'));
    await waitFor(() => expect(guard.current.has('rev-1')).toBe(false));
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1); // no retry loop: the effect deps do not change on error
    renderForm({ autoAssessGuard: guard });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(api.mock.calls[1][0]).toBe('rev-1');
  });

  it('a failed MANUAL request from a second form never releases the id the automatic request still holds', async () => {
    const requestA = deferred<AgentAssessmentResponse>();
    const requestB = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(requestA.promise).mockReturnValueOnce(requestB.promise);
    const guard = newGuard();
    const queryClient = newClient();
    const formA = renderForm({ autoAssessGuard: guard }, queryClient); // automatic → request A in flight
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(guard.current.has('rev-1')).toBe(true);
    const formB = renderForm({ autoAssessGuard: guard }, queryClient); // linked card + queue row: same review
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1); // B's automatic generation is skipped by the shared guard
    await userEvent
      .setup()
      .click(within(formB.container).getByRole('button', { name: /^generate agent assessment$/i }));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2)); // B's MANUAL request B
    requestB.reject(new Error('LM unavailable'));
    await within(formB.container).findByText('Failed to generate agent assessment');
    expect(guard.current.has('rev-1')).toBe(true); // B's failure is not the automatic request's failure
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(2); // no request C
    requestA.resolve(RESPONSE);
    await within(formA.container).findByRole('button', { name: /regenerate agent assessment/i });
    expect(guard.current.has('rev-1')).toBe(true);
  });

  it('does not auto-generate for a cached assessment; Regenerate forces a fresh one', async () => {
    api.mockResolvedValue(RESPONSE);
    renderForm({ review: { ...REVIEW, agent_assessment_json: ASSESSMENT } });
    await flushTimers();
    expect(api).not.toHaveBeenCalled();
    await userEvent.setup().click(screen.getByRole('button', { name: /regenerate agent assessment/i }));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(api).toHaveBeenCalledWith('rev-1', true);
  });
});
