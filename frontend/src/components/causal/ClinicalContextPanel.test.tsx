import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { ClinicalContextPanel } from './ClinicalContextPanel';
import type { ClinicalContext } from '@/types/causal';

const FULL: ClinicalContext = {
  brand: 'Kisqali',
  drug_name: 'ribociclib',
  disease: 'Malignant neoplasm of breast',
  our_outcome: 'persistent_180d',
  mapped_endpoint: 'Treatment persistence / duration of therapy',
  mechanism: { mechanism_of_action: 'CDK4/6 inhibitor', source: 'chembl' },
  pivotal_endpoints: {
    endpoints: ['Overall Survival (OS)', 'Progression-Free Survival (PFS)'],
    source: 'clinicaltrials.gov',
  },
  real_world_evidence: {
    pmid: '35642282',
    title: 'CDK4/6 inhibitor treatment use in women with advanced breast cancer.',
    journal: 'J Oncol Pharm Pract',
    pubdate: '2023 Jul',
    doi: '10.1177/10781552221102884',
    url: 'https://pubmed.ncbi.nlm.nih.gov/35642282/',
    source: 'pubmed',
  },
  approved_indications: {
    indications: ['HR+/HER2- advanced breast cancer in combination with an aromatase inhibitor'],
    limitations_of_use: 'Not indicated for patients with early-stage breast cancer.',
    boxed_warning: 'QT interval prolongation has been observed. Monitor ECG prior to treatment.',
    source: 'openfda',
  },
  competitor_landscape: {
    competitors: ['Verzenio', 'Ibrance'],
    count: 2,
    source: 'curated',
  },
  honesty_label:
    'Effect estimate = a SYNTHETIC patient cohort. Clinical context below is REAL and cited.',
};

describe('ClinicalContextPanel', () => {
  it('renders drug, MoA, mapped endpoint, pivotal endpoints, and a linked citation', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/ribociclib/i)).toBeInTheDocument();
    // Exact match targets only the MoA span; the RWE title (which also starts with
    // "CDK4/6 inhibitor") is longer and renders verbatim, so it cannot collide here.
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
    expect(screen.getByText(/Treatment persistence/i)).toBeInTheDocument();
    expect(screen.getByText(/Overall Survival/)).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /35642282|breast cancer|CDK4/i });
    expect(link).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/35642282/');
  });

  it('always shows the synthetic/real honesty label', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/SYNTHETIC/)).toBeInTheDocument();
    expect(screen.getByText(/REAL and cited/i)).toBeInTheDocument();
  });

  it('marks a static_fallback source honestly and omits a missing citation', () => {
    const degraded: ClinicalContext = {
      ...FULL,
      mechanism: { mechanism_of_action: 'complement Factor B inhibitor', source: 'static_fallback' },
      pivotal_endpoints: { endpoints: ['Transfusion avoidance'], source: 'static_fallback' },
      real_world_evidence: null,
    };
    render(<ClinicalContextPanel context={degraded} />);
    expect(screen.getAllByText(/curated|fallback/i).length).toBeGreaterThan(0);
    expect(screen.queryByRole('link')).toBeNull();
  });

  it('renders the Regulatory/Label section with live FDA label chip when source=openfda', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/HR\+\/HER2-/)).toBeInTheDocument();
    expect(screen.getByText(/live FDA label/i)).toBeInTheDocument();
    expect(screen.getByText(/Limitations of use/i)).toBeInTheDocument();
  });

  it('renders the boxed warning with emphasis', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/BOXED WARNING/i)).toBeInTheDocument();
    expect(screen.getByText(/QT interval prolongation/i)).toBeInTheDocument();
  });

  it('labels static_fallback approved_indications as curated fallback, not live FDA', () => {
    const fallback: ClinicalContext = {
      ...FULL,
      approved_indications: {
        indications: ['HR+/HER2- advanced breast cancer'],
        limitations_of_use: null,
        boxed_warning: null,
        source: 'static_fallback',
      },
    };
    render(<ClinicalContextPanel context={fallback} />);
    // "live FDA label" chip must NOT appear for static_fallback
    expect(screen.queryByText(/live FDA label/i)).toBeNull();
    // The curated fallback chip must appear (at least one)
    expect(screen.getAllByText(/curated fallback/i).length).toBeGreaterThan(0);
  });

  it('renders competitor landscape chips and count', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/2 rivals/i)).toBeInTheDocument();
    expect(screen.getByText('Verzenio')).toBeInTheDocument();
    expect(screen.getByText('Ibrance')).toBeInTheDocument();
    // Source chip: curated (not FDA-sourced)
    expect(screen.getByText(/^curated$/i)).toBeInTheDocument();
  });

  it('omits the competitor landscape section when count is 0', () => {
    const noRivals: ClinicalContext = {
      ...FULL,
      competitor_landscape: { competitors: [], count: 0, source: 'curated' },
    };
    render(<ClinicalContextPanel context={noRivals} />);
    expect(screen.queryByText(/rivals/i)).toBeNull();
  });

  it('omits both new sections when approved_indications and competitor_landscape are absent', () => {
    const bare: ClinicalContext = {
      ...FULL,
      approved_indications: null,
      competitor_landscape: null,
    };
    render(<ClinicalContextPanel context={bare} />);
    expect(screen.queryByText(/live FDA label/i)).toBeNull();
    expect(screen.queryByText(/rivals/i)).toBeNull();
  });

  describe('pivotal-endpoint provenance (Fix B-1)', () => {
    it('explains that endpoints are real trial endpoints, that Week N is weeks-from-baseline, and what Scenario N means', () => {
      render(<ClinicalContextPanel context={FULL} />);
      // The user saw endpoint titles like "...UAS7 at Week 12 (Scenario 1 ...)" with no
      // explanation anywhere. An always-visible footnote must clarify all three.
      expect(screen.getByText(/registered pivotal trials/i)).toBeInTheDocument();
      expect(screen.getByText(/weeks after trial baseline/i)).toBeInTheDocument();
      expect(screen.getByText(/not a calendar date/i)).toBeInTheDocument();
      expect(screen.getByText(/pre-specified analysis scenario/i)).toBeInTheDocument();
    });

    it('exposes an accessible tooltip affordance about the endpoints', () => {
      render(<ClinicalContextPanel context={FULL} />);
      expect(screen.getByLabelText(/about these trial endpoints/i)).toBeInTheDocument();
    });

    it('keeps the verbatim endpoint text (does not strip a Scenario suffix)', () => {
      const withScenario: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: {
          endpoints: [
            'Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12 (Scenario 1 With UAS7 as Primary Efficacy Endpoint)',
          ],
          source: 'clinicaltrials.gov',
        },
      };
      render(<ClinicalContextPanel context={withScenario} />);
      expect(
        screen.getByText(/Scenario 1 With UAS7 as Primary Efficacy Endpoint/i)
      ).toBeInTheDocument();
    });

    it('adds NO hyperlink in this interim FE-only step (links come later)', () => {
      const degraded: ClinicalContext = {
        ...FULL,
        real_world_evidence: null,
        seminal_real_world_evidence: undefined,
      };
      render(<ClinicalContextPanel context={degraded} />);
      // The endpoint footnote must be plain text — no anchor introduced here.
      expect(screen.queryByRole('link')).toBeNull();
    });

    it('omits the endpoint footnote when there are no pivotal endpoints', () => {
      const noEndpoints: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: { endpoints: [], source: 'static_fallback' },
      };
      render(<ClinicalContextPanel context={noEndpoints} />);
      expect(screen.queryByText(/weeks after trial baseline/i)).toBeNull();
    });

    it('does NOT claim ClinicalTrials.gov "verbatim" provenance for curated-fallback endpoints', () => {
      // The section renders for static_fallback too (CT.gov was unreachable). The
      // footnote must stay honest — no verbatim/CT.gov-provenance or Week/Scenario claim.
      const fallback: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: {
          endpoints: ['Change from baseline in UAS7 (Urticaria Activity Score over 7 days)'],
          source: 'static_fallback',
        },
      };
      render(<ClinicalContextPanel context={fallback} />);
      expect(screen.getByText(/curated reference/i)).toBeInTheDocument();
      expect(
        screen.getByText(/live ClinicalTrials\.gov lookup was\s+unavailable/i)
      ).toBeInTheDocument();
      expect(screen.queryByText(/via ClinicalTrials\.gov/i)).toBeNull();
      expect(screen.queryByText(/weeks after trial baseline/i)).toBeNull();
      expect(screen.queryByText(/pre-specified analysis scenario/i)).toBeNull();
    });
  });

  it('renders a curated brand-specific seminal RWE and demotes the live one to "Additional"', () => {
    const withSeminal: ClinicalContext = {
      ...FULL,
      seminal_real_world_evidence: {
        pmid: '36135090',
        title: 'Real-World Clinical Outcomes of Ribociclib in Premenopausal HR+/HER2- BC',
        journal: 'Current Oncology',
        pubdate: '2022',
        doi: '10.3390/curroncol29090521',
        url: 'https://pubmed.ncbi.nlm.nih.gov/36135090/',
        source: 'curated',
      },
    };
    render(<ClinicalContextPanel context={withSeminal} />);
    // The seminal block names the brand-specific paper and is labelled brand-specific.
    expect(screen.getByText(/Seminal real-world evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/curated · brand-specific/i)).toBeInTheDocument();
    const seminalLink = screen.getByRole('link', { name: /Ribociclib in Premenopausal/i });
    expect(seminalLink).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/36135090/');
    // The live relevance citation is demoted to "Additional" so it does not read as
    // the brand's own evidence (the reported competitor-paper confusion).
    expect(screen.getByText(/Additional real-world evidence/i)).toBeInTheDocument();
  });
});
