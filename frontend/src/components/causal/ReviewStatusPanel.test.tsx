import { describe, it, expect, vi } from 'vitest';
// The "Open review" deep link is a router <Link>: render under the router-wrapped helper.
import { fireEvent, renderWithAllProviders, screen, waitFor } from '@/test/utils';
import { ReviewStatusPanel } from './ReviewStatusPanel';

const SWITCH_HALT =
  'Estimate withheld: CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL=true requires an active expert approval of the DAG structure for a REVIEW-band estimate, and this structure holds none (gate decision: pending_review). Re-run once the review is resolved.';
const GATE_BLOCKED = 'Refutation gate BLOCKED — the estimate did not survive robustness checks.';

describe('ReviewStatusPanel', () => {
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

  // A plain-object lookup resolves inherited members: "toString" must render
  // verbatim like any other unknown decision, never as an empty badge.
  it('renders an inherited-property decision verbatim, never an empty badge', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="toString" />);
    expect(screen.getByText('toString')).toBeInTheDocument();
  });

  it('copies the full discovery record id to the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
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
});
