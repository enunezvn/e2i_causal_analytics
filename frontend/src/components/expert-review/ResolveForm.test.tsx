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
import { generateReviewAssessment, resolveReview } from '@/api/expert-review';

const api = vi.mocked(generateReviewAssessment);
const resolveApi = vi.mocked(resolveReview);

/** The structure version the form displays — the resolution is bound to it. */
const HASH = 'h'.repeat(64);
/**
 * The adjustment-set half of that version (#1991 debt 3, codex round 2). The
 * backend's DAG hash EXCLUDES adjustment sets, so this half is the only thing a
 * covariate-only advance moves — and the only thing that can catch one.
 */
const ADJ = 'w'.repeat(64);
const REVIEW: PendingReviewItem = { review_id: 'rev-1', brand: 'Kisqali', treatment_variable: 't', outcome_variable: 'y', dag_version_hash: HASH, adjustment_set_hash: ADJ };
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
  // resolveApi was NOT reset here, so its call COUNT accumulated across the file
  // and every `toHaveBeenCalledTimes(1)` below held only for whichever test
  // called it first. That is an ordering dependency, not a pin: adding a second
  // resolve test made the first one's neighbour fail with "got 2 times". Each
  // test sets its own resolved/rejected value, so a reset here costs nothing.
  resolveApi.mockReset();
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

  // Codex iter-2 F2: an AUTOMATIC request can succeed with persisted:false (HTTP
  // 200, the store rejected the cache write). Before, the guard stayed marked and
  // nothing was cached, so after a collapse/remount neither the form nor bulk
  // Prepare would ever regenerate it.
  it('automatic persisted:false shows the unsaved state, releases the guard, and a remount generates again', async () => {
    api.mockResolvedValueOnce({ ...RESPONSE, persisted: false }).mockResolvedValueOnce(RESPONSE);
    const guard = newGuard();
    const { unmount } = renderForm({ autoAssessGuard: guard });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
    expect(
      await screen.findByText(
        'Assessment generated but not saved — it will be lost when this row is collapsed; retry with Generate'
      )
    ).toBeInTheDocument();
    await waitFor(() => expect(guard.current.has('rev-1')).toBe(false));
    await flushTimers();
    expect(api).toHaveBeenCalledTimes(1); // no retry loop while mounted
    unmount();
    renderForm({ autoAssessGuard: guard });
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(api.mock.calls[1][0]).toBe('rev-1');
    expect(guard.current.has('rev-1')).toBe(true); // the second automatic request holds it again
  });

  it('a MANUAL Regenerate resolving persisted:false shows the unsaved state but never releases the guard', async () => {
    api.mockResolvedValueOnce(RESPONSE).mockResolvedValueOnce({ ...RESPONSE, persisted: false });
    const guard = newGuard();
    renderForm({ autoAssessGuard: guard });
    const button = await screen.findByRole('button', { name: /regenerate agent assessment/i });
    expect(guard.current.has('rev-1')).toBe(true);
    await userEvent.setup().click(button);
    await waitFor(() => expect(api).toHaveBeenCalledTimes(2));
    expect(
      await screen.findByText(
        'Assessment generated but not saved — it will be lost when this row is collapsed; retry with Generate'
      )
    ).toBeInTheDocument();
    await flushTimers();
    expect(guard.current.has('rev-1')).toBe(true); // ownership: only the automatic request releases
    expect(api).toHaveBeenCalledTimes(2);
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

  describe('the resolution is bound to the version the form displayed', () => {
    it('sends the reviewed dag_version_hash with the verdict', async () => {
      api.mockResolvedValue(RESPONSE);
      resolveApi.mockResolvedValue({ review_id: 'rev-1', approval_status: 'approved', success: true });
      renderForm();
      await userEvent.setup().click(await screen.findByRole('button', { name: /^approve$/i }));
      await waitFor(() => expect(resolveApi).toHaveBeenCalledTimes(1));
      expect(resolveApi).toHaveBeenCalledWith('rev-1', {
        approval_status: 'approved',
        checklist: {},
        comments: undefined,
        dag_version_hash: HASH,
        adjustment_set_hash: ADJ,
      });
    });

    it('sends an explicit null when the row carries no adjustment-set hash', async () => {
      api.mockResolvedValue(RESPONSE);
      resolveApi.mockResolvedValue({ review_id: 'rev-1', approval_status: 'approved', success: true });
      // A pre-142 row, or one the old api image minted: its adjustment set is
      // UNKNOWN. The form must still echo that, as a JSON null.
      renderForm({ review: { ...REVIEW, adjustment_set_hash: null } });
      await userEvent.setup().click(await screen.findByRole('button', { name: /^approve$/i }));
      await waitFor(() => expect(resolveApi).toHaveBeenCalledTimes(1));
      const body = resolveApi.mock.calls[0][1];
      // PRESENT with a null value, not absent: the backend 422s a missing key,
      // and `undefined` would vanish from the serialised body entirely.
      expect('adjustment_set_hash' in body).toBe(true);
      expect(body.adjustment_set_hash).toBeNull();
      expect(JSON.parse(JSON.stringify(body))).toHaveProperty('adjustment_set_hash', null);
    });

    it('shows the 409 reload instruction in the submit banner', async () => {
      api.mockResolvedValue(RESPONSE);
      // ApiError.message carries the backend 409 detail (src/api/main.py maps a
      // 409 detail verbatim onto the `message` field api-client reads).
      resolveApi.mockRejectedValue(
        new Error(
          'Review rev-1 has advanced to a new structure version since this form was opened; ' +
            'reload the review and resolve the current version.'
        )
      );
      renderForm();
      await userEvent.setup().click(await screen.findByRole('button', { name: /^reject$/i }));
      expect(await screen.findByText('Failed to submit review')).toBeInTheDocument();
      expect(await screen.findByText(/reload the review and resolve the current version/)).toBeInTheDocument();
    });
  });
});
