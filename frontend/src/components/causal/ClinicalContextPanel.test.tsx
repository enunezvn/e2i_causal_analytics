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
    endpoints: [
      { measure: 'Overall Survival (OS)', time_frame: 'Up to 5 years', nct_id: 'NCT01958021' },
      { measure: 'Progression-Free Survival (PFS)', time_frame: null, nct_id: 'NCT01958021' },
    ],
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
      pivotal_endpoints: {
        endpoints: [{ measure: 'Transfusion avoidance', time_frame: null, nct_id: null }],
        source: 'static_fallback',
      },
      real_world_evidence: null,
    };
    render(<ClinicalContextPanel context={degraded} />);
    expect(screen.getAllByText(/curated|fallback/i).length).toBeGreaterThan(0);
    // Curated-fallback endpoints carry no nct_id, so no CT.gov deep-link is rendered,
    // and with RWE null there is no citation link either.
    expect(screen.queryByRole('link')).toBeNull();
  });

  it('renders the Regulatory/Label section with live FDA label chip when source=openfda', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/HR\+\/HER2-/)).toBeInTheDocument();
    expect(screen.getByText(/live FDA label/i)).toBeInTheDocument();
    expect(screen.getByText(/Limitations of use/i)).toBeInTheDocument();
  });

  it('does NOT surface the boxed warning (this panel grounds the commercial signal, not safety labeling)', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.queryByText(/BOXED WARNING/i)).toBeNull();
    expect(screen.queryByText(/QT interval prolongation/i)).toBeNull();
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

  describe('trial endpoints as grounding (not a parameter dump)', () => {
    it('surfaces endpoint measures as clinical ground truth for the outcome definition', () => {
      render(<ClinicalContextPanel context={FULL} />);
      // Exact match targets the section heading only; the panel's intro sentence also
      // contains the words "real trial endpoints" as prose, so a regex would collide.
      expect(screen.getByText('Real trial endpoints')).toBeInTheDocument();
      expect(screen.getByText(/Overall Survival/)).toBeInTheDocument();
      expect(screen.getByText(/clinical ground truth our synthetic outcome stands in for/i))
        .toBeInTheDocument();
    });

    it('does NOT surface per-endpoint parameters (time frame, NCT deep-link, analysis scenario)', () => {
      const withParams: ClinicalContext = {
        ...FULL,
        real_world_evidence: null,
        seminal_real_world_evidence: undefined,
        pivotal_endpoints: {
          endpoints: [
            {
              measure:
                'Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12 (Scenario 1 With UAS7 as Primary Efficacy Endpoint)',
              time_frame: 'Baseline, Week 12',
              nct_id: 'NCT05030311',
            },
          ],
          source: 'clinicaltrials.gov',
        },
      };
      render(<ClinicalContextPanel context={withParams} />);
      // The raw parameters the user asked us to stop surfacing must be gone.
      expect(screen.queryByText(/Time frame:/i)).toBeNull();
      expect(screen.queryByRole('link', { name: /NCT05030311/i })).toBeNull();
      expect(screen.queryByLabelText(/about these trial endpoints/i)).toBeNull();
      expect(screen.queryByText(/weeks after trial baseline/i)).toBeNull();
      expect(screen.queryByText(/pre-specified analysis scenario/i)).toBeNull();
      // The measure title itself is still shown (as grounding).
      expect(screen.getByText(/UAS7/)).toBeInTheDocument();
    });

    it('caps the endpoint list and discloses how many more exist', () => {
      const many: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: {
          endpoints: Array.from({ length: 8 }, (_, i) => ({
            measure: `Endpoint measure ${i + 1}`,
            time_frame: null,
            nct_id: null,
          })),
          source: 'clinicaltrials.gov',
        },
      };
      render(<ClinicalContextPanel context={many} />);
      expect(screen.getByText('Endpoint measure 5')).toBeInTheDocument();
      expect(screen.queryByText('Endpoint measure 6')).toBeNull();
      expect(screen.getByText(/\+ 3 more registered\s+trial endpoints/i)).toBeInTheDocument();
    });

    it('keeps the grounding line honest for a curated fallback (no CT.gov claim)', () => {
      const fallback: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: {
          endpoints: [
            {
              measure: 'Change from baseline in UAS7 (Urticaria Activity Score over 7 days)',
              time_frame: null,
              nct_id: null,
            },
          ],
          source: 'static_fallback',
        },
      };
      render(<ClinicalContextPanel context={fallback} />);
      expect(screen.getByText(/curated reference/i)).toBeInTheDocument();
      expect(screen.queryByText(/actually measured/i)).toBeNull();
    });

    it('omits the endpoint section entirely when there are no endpoints', () => {
      const noEndpoints: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: { endpoints: [], source: 'static_fallback' },
      };
      render(<ClinicalContextPanel context={noEndpoints} />);
      // Exact match: the intro sentence (always shown) also mentions "real trial
      // endpoints" as prose, so only the exact section heading should be absent.
      expect(screen.queryByText('Real trial endpoints')).toBeNull();
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

// #1763: the panel was accurate but read as unrelated to the analysis on screen.
describe('ClinicalContextPanel — analysis framing (#1763)', () => {
  const ANALYSIS: ClinicalContext = {
    ...FULL,
    our_treatment: 'treatment_arm',
    treatment_context: {
      column: 'treatment_arm',
      label: 'Treatment arm',
      framing: 'being on a ribociclib-containing regimen',
      kind: 'drug_therapy',
      source: 'curated',
    },
    analysis_framing:
      'This analysis estimates the effect of being on a ribociclib-containing regimen on 180-day treatment persistence for ribociclib in Malignant neoplasm of breast.',
    real_world_evidence: {
      ...FULL.real_world_evidence!,
      search_term: 'ribociclib breast cancer persistence real-world',
    },
  };

  it('leads with the sentence naming the analysis being interrogated', () => {
    render(<ClinicalContextPanel context={ANALYSIS} />);
    expect(
      screen.getByText(/This analysis estimates the effect of being on a ribociclib-containing regimen/i)
    ).toBeInTheDocument();
  });

  it('discloses the literature query behind the live citation', () => {
    render(<ClinicalContextPanel context={ANALYSIS} />);
    expect(
      screen.getByText(/ribociclib breast cancer persistence real-world/i)
    ).toBeInTheDocument();
  });

  it('says plainly when the treatment is a commercial lever the clinical sources do not cover', () => {
    const commercial: ClinicalContext = {
      ...ANALYSIS,
      our_treatment: 'copay_support',
      treatment_context: {
        column: 'copay_support',
        label: 'Copay support',
        framing: 'receiving copay assistance',
        kind: 'commercial',
        source: 'curated',
      },
      analysis_framing:
        'This analysis estimates the effect of receiving copay assistance on 180-day treatment persistence for ribociclib in Malignant neoplasm of breast.',
    };
    render(<ClinicalContextPanel context={commercial} />);
    expect(screen.getByText(/access and promotion lever/i)).toBeInTheDocument();
  });

  it('renders exactly as before when there is no analysis frame', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.queryByText(/This analysis estimates the effect of/i)).not.toBeInTheDocument();
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
  });
});

// #1763 Phase 2: evidence gathered FOR the analysis, and an honest empty state when
// the treatment is a commercial lever the clinical sources do not describe.
describe('ClinicalContextPanel — causal evidence (#1763)', () => {
  const WITH_EVIDENCE: ClinicalContext = {
    ...FULL,
    our_treatment: 'treatment_arm',
    causal_evidence: {
      status: 'evidence',
      indication_edge: {
        predicate: 'associated_with',
        drug_id: 'CHEMBL3545110',
        drug_name: 'RIBOCICLIB',
        disease_id: 'MONDO_0007254',
        disease_name: 'breast cancer',
        max_clinical_stage: 'PHASE_3',
        source: 'open_targets',
      },
      citations: [
        {
          pmid: '40896422',
          title: 'Real-world effectiveness and safety of CDK4/6i',
          journal: 'Front Oncol',
          pubdate: '2025',
          url: 'https://pubmed.ncbi.nlm.nih.gov/40896422/',
          entities_found: ['ribociclib', 'breast cancer'],
          confidence: 0.5,
          source: 'pubmed+europepmc',
        },
      ],
      sources_unavailable: [],
      note: 'Open Targets records the clinical stage per indication and lags the FDA label.',
    },
  };

  it('shows the indication edge with the stage of THIS indication node', () => {
    render(<ClinicalContextPanel context={WITH_EVIDENCE} />);
    expect(screen.getByText(/is recorded in development for/i)).toBeInTheDocument();
    expect(screen.getByText(/PHASE_3 · open targets/i)).toBeInTheDocument();
    // ...and says the label, not this edge, is the approval authority.
    expect(screen.getByText(/lags the FDA label/i)).toBeInTheDocument();
  });

  it('links the verified citations and says what the abstract actually named', () => {
    render(<ClinicalContextPanel context={WITH_EVIDENCE} />);
    const link = screen.getByRole('link', { name: /Real-world effectiveness and safety of CDK4\/6i/i });
    expect(link).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/40896422/');
    expect(screen.getByText(/abstract names ribociclib \+ breast cancer/i)).toBeInTheDocument();
  });

  it('states plainly that clinical sources do not describe a commercial lever', () => {
    const lever: ClinicalContext = {
      ...FULL,
      our_treatment: 'copay_support',
      causal_evidence: {
        status: 'commercial_lever',
        indication_edge: null,
        citations: [],
        sources_unavailable: [],
        note: 'Copay support is a commercial access/promotion lever. Biomedical and regulatory sources describe the therapy and its indication, not this lever.',
      },
    };
    render(<ClinicalContextPanel context={lever} />);
    expect(screen.getByText(/not this lever/i)).toBeInTheDocument();
    expect(screen.queryByText(/is recorded in development for/i)).not.toBeInTheDocument();
  });

  it('renders no evidence section for a leaderboard payload that never asked for it', () => {
    const notRequested: ClinicalContext = {
      ...FULL,
      causal_evidence: {
        status: 'not_requested',
        indication_edge: null,
        citations: [],
        sources_unavailable: [],
        note: 'Analysis-specific evidence is gathered when the analysis is opened.',
      },
    };
    render(<ClinicalContextPanel context={notRequested} />);
    expect(screen.queryByText(/Evidence for this analysis/i)).not.toBeInTheDocument();
  });
});

// The brand-level citation chip is the one honesty label with no UI test: it is what
// stops a brand-level paper reading as analysis-level (adversarial review finding).
describe('ClinicalContextPanel — citation provenance chips (#1763)', () => {
  it('labels a brand-level fallback citation as such, not as a curated fallback', () => {
    const brandLevel: ClinicalContext = {
      ...FULL,
      real_world_evidence: {
        ...FULL.real_world_evidence!,
        source: 'pubmed_brand',
        search_term: 'ribociclib persistence adherence breast cancer real-world',
      },
    };
    render(<ClinicalContextPanel context={brandLevel} />);
    expect(screen.getByText(/pubmed \(brand-level\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/curated fallback/i)).not.toBeInTheDocument();
  });

  it('never claims a source it does not recognise is curated', () => {
    const unknown: ClinicalContext = {
      ...FULL,
      real_world_evidence: { ...FULL.real_world_evidence!, source: 'some_new_source' },
    };
    render(<ClinicalContextPanel context={unknown} />);
    expect(screen.getByText(/some_new_source/i)).toBeInTheDocument();
    expect(screen.queryByText(/curated fallback/i)).not.toBeInTheDocument();
  });

  it('names the molecule Open Targets actually answered about', () => {
    const withEdge: ClinicalContext = {
      ...FULL,
      causal_evidence: {
        status: 'evidence',
        indication_edge: {
          predicate: 'treats',
          drug_id: 'CHEMBL3545110',
          drug_name: 'RIBOCICLIB SUCCINATE',
          disease_id: 'MONDO_0007254',
          disease_name: 'breast cancer',
          max_clinical_stage: 'APPROVAL',
          source: 'open_targets',
        },
        citations: [],
        sources_unavailable: [],
        note: '',
      },
    };
    render(<ClinicalContextPanel context={withEdge} />);
    expect(screen.getByText(/RIBOCICLIB SUCCINATE/i)).toBeInTheDocument();
  });
});

describe('#1775 — grounding the causal scenario', () => {
  const GROUNDED: ClinicalContext = {
    ...FULL,
    our_treatment: 'copay_support',
    analysis_grounding: {
      label_considerations: [
        {
          title: 'QT Interval Prolongation',
          detail: 'Monitor electrocardiograms (ECGs) and electrolytes prior to initiation.',
          section: 'warnings_and_cautions',
          references: '2.2 , 5.3',
          source: 'openfda',
        },
        {
          title: 'Dosage and administration',
          detail:
            'Dose interruption, reduction, and/or discontinuation may be required based on individual safety and tolerability.',
          section: 'dosage_and_administration',
          references: '2.2',
          source: 'openfda',
        },
      ],
      competitive_context:
        'A patient who stops ribociclib in breast cancer has alternatives within the same class: Ibrance (palbociclib), Verzenio (abemaciclib). A switch to one of these is a competing risk for this outcome.',
      note: 'Label factors bearing on staying on therapy ... This is a filtered view, not the complete safety profile. Copay support is a commercial access lever and the label says nothing about it.',
      outcome_theme: 'persistence',
    },
  };

  it('grounds a COMMERCIAL analysis instead of only refusing', () => {
    render(<ClinicalContextPanel context={GROUNDED} />);
    expect(screen.getByText(/what bears on this analysis/i)).toBeInTheDocument();
    expect(screen.getByText(/QT Interval Prolongation/)).toBeInTheDocument();
    expect(
      screen.getByText(/Dose interruption, reduction, and\/or discontinuation/)
    ).toBeInTheDocument();
  });

  it('cites the label section each consideration came from', () => {
    render(<ClinicalContextPanel context={GROUNDED} />);
    expect(screen.getByText(/label 2\.2 , 5\.3/)).toBeInTheDocument();
  });

  it('frames the competitor set as a competing risk for the outcome', () => {
    render(<ClinicalContextPanel context={GROUNDED} />);
    expect(screen.getByText(/competing risk for this outcome/i)).toBeInTheDocument();
  });

  it('keeps the boundary: the label says nothing about the lever', () => {
    render(<ClinicalContextPanel context={GROUNDED} />);
    expect(screen.getByText(/the label says nothing about it/i)).toBeInTheDocument();
  });

  it('renders nothing when there is no scenario to ground', () => {
    render(<ClinicalContextPanel context={{ ...FULL, analysis_grounding: null }} />);
    expect(screen.queryByText(/what bears on this analysis/i)).not.toBeInTheDocument();
  });
});

describe('ClinicalContextPanel grounding disclosure', () => {
  it('still shows the note when there is nothing to ground on', () => {
    // The note is the ONLY thing that separates "the label was read and carries
    // nothing bearing on this outcome" from "we could not read the label" (#1767).
    // Gating the block on considerations-or-competitors meant that disclosure
    // vanished in exactly the case it exists for. Today every curated brand has
    // competitors so this never fired in production — the honesty was accidental,
    // not structural, and one brand added without a competitor map would have
    // silently removed it.
    render(
      <ClinicalContextPanel
        context={{
          ...FULL,
          analysis_grounding: {
            label_considerations: [],
            competitive_context: null,
            note: 'The FDA label for Kisqali could not be read for factors bearing on 180-day treatment persistence, so what is missing here is unknown, not absent.',
            outcome_theme: 'persistence',
          },
        }}
      />,
    );
    expect(screen.getByText(/could not be read/i)).toBeInTheDocument();
    expect(screen.getByText(/unknown, not absent/i)).toBeInTheDocument();
  });
});

describe('ClinicalContextPanel grounding heading honesty', () => {
  it('does not claim something bears on the analysis when nothing does', () => {
    // codex iter-10 HIGH. The body correctly said "unknown, not absent" while the
    // HEADING above it announced "What bears on this analysis" — a claim that
    // something does. The earlier test passed for the wrong reason: it checked the
    // disclosure appeared, never that the surrounding claim stayed honest.
    render(
      <ClinicalContextPanel
        context={{
          ...FULL,
          analysis_grounding: {
            label_considerations: [],
            competitive_context: null,
            note: 'The FDA label for Kisqali could not be read for factors bearing on 180-day treatment persistence, so what is missing here is unknown, not absent.',
            outcome_theme: 'persistence',
          },
        }}
      />,
    );
    expect(screen.getByText(/unknown, not absent/i)).toBeInTheDocument();
    expect(screen.queryByText(/what bears on this analysis/i)).not.toBeInTheDocument();
    expect(screen.getByText(/nothing established for this analysis/i)).toBeInTheDocument();
  });

  it('still uses the affirmative heading when there IS something', () => {
    render(
      <ClinicalContextPanel
        context={{
          ...FULL,
          analysis_grounding: {
            label_considerations: [
              {
                title: 'QT Interval Prolongation',
                detail: 'Monitor electrocardiograms (ECGs) and electrolytes prior to initiation.',
                section: 'warnings_and_cautions',
                references: '2.2 , 5.3',
                source: 'openfda',
              },
            ],
            competitive_context: null,
            note: 'Label factors bearing on staying on therapy.',
            outcome_theme: 'persistence',
          },
        }}
      />,
    );
    expect(screen.getByText(/what bears on this analysis/i)).toBeInTheDocument();
    expect(screen.queryByText(/nothing established for this analysis/i)).not.toBeInTheDocument();
  });
});

describe('ClinicalContextPanel endpoint copy honesty', () => {
  it('does not claim the outcome stands in for the endpoints when it is unmapped', () => {
    // codex iter-13 MEDIUM. The endpoints render for any brand, but "the clinical
    // ground truth our synthetic outcome stands in for" is a claim about a MAPPING,
    // and the backend returns mapped_endpoint: null when the outcome is not one we
    // have mapped. Same defect as the grounding heading — the sentence around the
    // data promised more than the data carried.
    // codex iter-14 HIGH: this asserted /stands in for/i, and the claim that survived
    // was the panel INTRO's plural "stand in for". The regex simply did not match the
    // text still on screen, so the test passed while the defect rendered. Vacuity by
    // near-miss rather than by empty collection — match BOTH forms, and scan the whole
    // container rather than trusting one query.
    const { container } = render(
      <ClinicalContextPanel context={{ ...FULL, mapped_endpoint: null }} />,
    );
    expect(container.textContent).not.toMatch(/stands? in for/i);
    expect(screen.getByText(/not one we have mapped to any of them/i)).toBeInTheDocument();
  });

  it('still makes the mapping claim when the outcome IS mapped', () => {
    const { container } = render(<ClinicalContextPanel context={FULL} />);
    // POSITIVE CONTROL for the assertion above: both places that make the claim.
    expect(container.textContent).toMatch(/outcomes stand in for/i);
    expect(container.textContent).toMatch(/outcome stands in for/i);
  });
});

describe('ClinicalContextPanel copy renders as text, not markup', () => {
  it('never shows a raw HTML entity to the reader', () => {
    // Caught immediately after writing it: `&rsquo;` inside a JSX *expression* is a
    // plain string, not an entity, so the conditional intro would have printed
    // "brand&rsquo;s" on screen. JSX only decodes entities in literal text children.
    for (const ctx of [FULL, { ...FULL, mapped_endpoint: null }]) {
      const { container, unmount } = render(<ClinicalContextPanel context={ctx} />);
      expect(container.textContent).not.toMatch(/&[a-z]+;/i);
      unmount();
    }
  });
});
