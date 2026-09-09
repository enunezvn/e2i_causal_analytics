import { describe, it, expect, vi, afterEach } from 'vitest';
// The "Open review" deep link is a router <Link>: render under the router-wrapped helper.
import { fireEvent, renderWithAllProviders, screen, waitFor } from '@/test/utils';
import { ReviewStatusPanel } from './ReviewStatusPanel';

const SWITCH_HALT =
  'Estimate withheld: CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true requires an active expert approval of the DAG structure for a REVIEW-band estimate, and this structure holds none (gate decision: pending_review). Re-run once the review is resolved.';
const GATE_BLOCKED = 'Refutation gate BLOCKED — the estimate did not survive robustness checks.';
// #1995: the agent's band + HITL sentence, now carried in refutation.review_caveat.
const APPROVAL_CAVEAT =
  'Refutation gate is BLOCK (failed robustness, confidence=0.41). This estimate did not pass and has been routed to expert review for adjudication. The DAG structure was expert-approved by admin@e2i.local (valid until 2027-03-09); that approval covers the DAG structure, not this estimate\'s statistical robustness.';
const REJECTION_CAVEAT =
  'Refutation gate is BLOCK (failed robustness, confidence=0.41). This estimate did not pass and has been routed to expert review for adjudication. The DAG structure was REJECTED by expert review by Dr. No (review rev-rejected): collider. A rejected structure is not re-queued; revise the DAG (a changed structure gets its own review) or ask an operator to queue a new review for this hash.';
const REJECTION_HALT =
  'Estimate withheld: a domain expert REJECTED this DAG structure, and an estimate built on a rejected structure is not a valid causal estimate whatever its refutation verdict. ' +
  REJECTION_CAVEAT;

describe('ReviewStatusPanel', () => {
  // The copy-affordance test stubs `navigator` (jsdom has no clipboard); drop
  // the stub after every test so no later case inherits it.
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders nothing when the run carried no review state and no DAG record', () => {
    const { container } = renderWithAllProviders(<ReviewStatusPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows a pending structure with a deep link into the queue', () => {
    renderWithAllProviders(
      <ReviewStatusPanel decision="pending_review" reviewId="rev-9" discoveredDagId={null} />
    );
    expect(screen.getByText('Pending expert review')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /open review/i });
    expect(link).toHaveAttribute('href', '/expert-reviews?review=rev-9');
  });

  it('shows the rejection and the halt message from the run warnings', () => {
    renderWithAllProviders(
      <ReviewStatusPanel
        decision="rejected"
        reviewId="rev-rejected"
        warnings={[
          'Estimate withheld: a domain expert REJECTED this DAG structure by Dr. No (review rev-rejected): collider.',
          'Refutation gate BLOCKED — the estimate did not survive robustness checks.',
        ]}
      />
    );
    expect(screen.getByText('Structure rejected')).toBeInTheDocument();
    expect(screen.getByText(/by Dr\. No/)).toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  it('shows the durable discovery record id', () => {
    renderWithAllProviders(<ReviewStatusPanel discoveredDagId="8a61b3db-6aad-4b01-96e4-bbea0af861b4" />);
    expect(screen.getByText(/Durable discovery record/)).toBeInTheDocument();
    expect(screen.getByText('8a61b3db-6aad-4b01-96e4-bbea0af861b4')).toBeInTheDocument();
  });

  it('never invents a label for an unknown decision', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="something_new" />);
    expect(screen.getByText('something_new')).toBeInTheDocument();
  });

  // The run's warnings are rendered nowhere else in the drill-down, and the
  // approval-enforcement switch withholds the estimate on pending / blocked /
  // unavailable structures too (refutation.py:1215) — the halt must show for
  // any decision that carries one, not only a rejection.
  it.each(['pending_review', 'blocked', 'unavailable'])(
    'shows the approval-enforcement halt from the run warnings for %s',
    (decision) => {
      renderWithAllProviders(
        <ReviewStatusPanel decision={decision} warnings={[SWITCH_HALT, GATE_BLOCKED]} />
      );
      expect(screen.getByText(SWITCH_HALT)).toBeInTheDocument();
      expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
    }
  );

  it('shows no halt line for an approved structure whose warnings carry none', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="proceed" warnings={[GATE_BLOCKED]} />);
    expect(screen.getByText('Structure approved')).toBeInTheDocument();
    expect(screen.queryByText(/Estimate withheld/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  // #1995: a BLOCK-band run never halts (the statistical gate already withheld
  // the estimate), so its warnings carry no "Estimate withheld" line — the
  // approval / rejection sentence reaches the panel only via the caveat prop.
  it('shows the approval caveat (reviewer + validity window) when no halt line exists', () => {
    renderWithAllProviders(
      <ReviewStatusPanel
        decision="proceed"
        reviewId="rev-approved"
        reviewCaveat={APPROVAL_CAVEAT}
        warnings={[GATE_BLOCKED, APPROVAL_CAVEAT]}
      />
    );
    expect(screen.getByText('Structure approved')).toBeInTheDocument();
    expect(screen.getByText(APPROVAL_CAVEAT, { exact: true })).toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  it('shows the rejection caveat (reviewer + reason) when no halt line exists', () => {
    renderWithAllProviders(
      <ReviewStatusPanel
        decision="rejected"
        reviewId="rev-rejected"
        reviewCaveat={REJECTION_CAVEAT}
        warnings={[GATE_BLOCKED, REJECTION_CAVEAT]}
      />
    );
    expect(screen.getByText('Structure rejected')).toBeInTheDocument();
    expect(screen.getByText(REJECTION_CAVEAT, { exact: true })).toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  // The halt line embeds the caveat verbatim: render the halt, and the caveat
  // text must appear exactly once (never the halt AND a standalone copy).
  it('renders the halt line once, not the caveat a second time, when both exist', () => {
    renderWithAllProviders(
      <ReviewStatusPanel
        decision="rejected"
        reviewId="rev-rejected"
        reviewCaveat={REJECTION_CAVEAT}
        warnings={[REJECTION_HALT]}
      />
    );
    expect(screen.getByText(REJECTION_HALT, { exact: true })).toBeInTheDocument();
    // getAllByText is substring-matching without { exact: true }: the halt line
    // contains the caveat, so exactly one element must match it.
    expect(screen.getAllByText(/by Dr\. No \(review rev-rejected\): collider/)).toHaveLength(1);
    expect(screen.queryByText(REJECTION_CAVEAT, { exact: true })).not.toBeInTheDocument();
  });

  // Positive control for the caveat channel: an empty caveat renders no reason line.
  it('shows no reason line when the caveat is empty and warnings carry no halt', () => {
    renderWithAllProviders(
      <ReviewStatusPanel decision="pending_review" reviewCaveat="" warnings={[GATE_BLOCKED]} />
    );
    expect(screen.getByText('Pending expert review')).toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate/)).not.toBeInTheDocument();
  });

  // A plain-object lookup resolves inherited members: "toString" must render
  // verbatim like any other unknown decision, never as an empty badge.
  it('renders an inherited-property decision verbatim, never an empty badge', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="toString" />);
    expect(screen.getByText('toString')).toBeInTheDocument();
  });

  it('copies the full discovery record id to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    renderWithAllProviders(
      <ReviewStatusPanel discoveredDagId="8a61b3db-6aad-4b01-96e4-bbea0af861b4" />
    );
    const button = screen.getByRole('button', { name: /copy discovery record id/i });
    expect(button).toHaveTextContent('Copy id');
    fireEvent.click(button);
    expect(writeText).toHaveBeenCalledWith('8a61b3db-6aad-4b01-96e4-bbea0af861b4');
    await waitFor(() => expect(button).toHaveTextContent('Copied'));
    // The full id stays visible beside the affordance.
    expect(screen.getByText('8a61b3db-6aad-4b01-96e4-bbea0af861b4')).toBeInTheDocument();
  });

  // Positive control for the afterEach restore: runs after the copy test (file
  // order) and would inherit the stub if it leaked.
  it('does not leak the clipboard stub past the copy test', () => {
    expect(navigator.clipboard).toBeUndefined();
  });
});
