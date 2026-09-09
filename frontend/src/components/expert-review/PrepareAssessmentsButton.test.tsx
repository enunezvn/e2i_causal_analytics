/**
 * PrepareAssessmentsButton tests — real QueryClient, only the API module mocked,
 * DEFERRED promises so the strictly sequential walk, the first-error stop, the
 * guard release and the Stop button are observed mid-flight, not inferred.
 */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PrepareAssessmentsButton } from './PrepareAssessmentsButton';
import { queryKeys } from '@/lib/query-client';
import type { AgentAssessmentResponse, PendingReviewItem } from '@/types/expert-review';

vi.mock('@/api/expert-review', () => ({
  generateReviewAssessment: vi.fn(),
  resolveReview: vi.fn(),
  getExpertReview: vi.fn(),
  getPendingReviews: vi.fn(),
  getReviewSummary: vi.fn(),
}));
import { generateReviewAssessment } from '@/api/expert-review';

const api = vi.mocked(generateReviewAssessment);

const ROWS: PendingReviewItem[] = [{ review_id: 'rev-1' }, { review_id: 'rev-2' }];
const PENDING_PREFIX = [...queryKeys.expertReviews.all(), 'pending'];

function response(id: string): AgentAssessmentResponse {
  return { review_id: id, assessment: { items: [], is_fallback: true }, cached: false, persisted: true };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function renderButton() {
  const guard = { current: new Set<string>() };
  const queryClient = new QueryClient();
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  render(<PrepareAssessmentsButton reviews={ROWS} autoAssessGuard={guard} />, { wrapper });
  return { guard, invalidate, button: screen.getByRole('button', { name: /prepare assessments/i }) };
}

beforeEach(() => {
  // mockReset (not clear): a test that aborts early must not leak its `Once` queue
  // of deferred promises into the next test (measured: a never-resolving leftover
  // left the button stuck at "Preparing 0 / 2…").
  api.mockReset();
});

describe('PrepareAssessmentsButton', () => {
  it('walks the missing rows strictly one at a time, then invalidates the pending queue once', async () => {
    const first = deferred<AgentAssessmentResponse>();
    const second = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { guard, invalidate, button } = renderButton();
    // The visible label IS the accessible name (no aria-label masking the count / progress).
    expect(button).toHaveAccessibleName('Prepare assessments (2 missing)');

    await userEvent.setup().click(button);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(api).toHaveBeenLastCalledWith('rev-1');
    expect(guard.current.has('rev-1')).toBe(true); // positive control for the release assertion below
    expect(button).toHaveTextContent('Preparing 0 / 2');
    expect(button).toBeDisabled();

    first.resolve(response('rev-1'));
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(api).toHaveBeenLastCalledWith('rev-2');
    expect(button).toHaveTextContent('Preparing 1 / 2');
    expect(invalidate).not.toHaveBeenCalled();

    second.resolve(response('rev-2'));
    await waitFor(() => expect(button).toHaveTextContent('Prepare assessments (2 missing)'));
    expect(button).toBeEnabled();
    expect(invalidate).toHaveBeenCalledTimes(1);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: PENDING_PREFIX });
  });

  it('stops on the first error, says how far it got, and releases the failed id from the guard', async () => {
    api.mockRejectedValueOnce(new Error('LM unavailable'));
    const { guard, invalidate, button } = renderButton();
    await userEvent.setup().click(button);
    expect(await screen.findByText('Stopped after 0 of 2')).toBeInTheDocument();
    expect(screen.getByText('LM unavailable')).toBeInTheDocument();
    expect(api).toHaveBeenCalledTimes(1);
    expect(api).not.toHaveBeenCalledWith('rev-2');
    expect(guard.current.has('rev-1')).toBe(false);
    expect(guard.current.has('rev-2')).toBe(false);
    await waitFor(() => expect(invalidate).toHaveBeenCalledTimes(1));
    expect(button).toBeEnabled();
  });

  it('Stop ends the walk after the in-flight request; the next row is never requested', async () => {
    const first = deferred<AgentAssessmentResponse>();
    api.mockReturnValueOnce(first.promise).mockResolvedValue(response('rev-2'));
    const { button } = renderButton();
    const user = userEvent.setup();
    await user.click(button);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: /stop/i }));
    first.resolve(response('rev-1'));
    await waitFor(() => expect(button).toBeEnabled());
    expect(api).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: /stop/i })).not.toBeInTheDocument();
  });
});
