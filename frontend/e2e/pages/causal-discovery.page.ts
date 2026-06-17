import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for the Causal Discovery page (`/causal-discovery`).
 *
 * The page is now AGENT-DRIVEN: the analyst picks only the causal question
 * (treatment -> outcome); the causal_impact agent learns the DAG from data
 * (guided structure discovery), estimates the effect data-drivenly, and runs
 * refutation. Honest states this POM exposes:
 *  - header + "Agent-driven" badge
 *  - the question form (treatment / outcome selects + Discover & Analyze)
 *  - empty: EmptyState "No discovery run yet" before a run
 *
 * The previous manual workbench (library-routing form, parallel-pipeline / KG
 * buttons, always-rendered DAG-viz chrome) was removed, so the old badge / viz
 * locators no longer apply.
 */
export class CausalDiscoveryPage extends BasePage {
  readonly url = ROUTES.CAUSAL_DISCOVERY
  readonly pageTitle = /Causal Discovery|E2I/i

  constructor(page: Page) {
    super(page)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: 'Causal Discovery', level: 1 })
  }

  get pageDescription(): Locator {
    return this.page.getByText(/learns the causal structure from the data/i)
  }

  // The single technology badge on the agent-driven page.
  get agentDrivenBadge(): Locator {
    return this.page.getByText('Agent-driven', { exact: false }).first()
  }

  // Question form — the only inputs (the genuine user decision).
  get treatmentSelect(): Locator {
    return this.page.getByRole('combobox', { name: 'Treatment variable' })
  }

  get outcomeSelect(): Locator {
    return this.page.getByRole('combobox', { name: 'Outcome variable' })
  }

  get analyzeButton(): Locator {
    return this.page.getByRole('button', { name: /Discover.*Analyze/i }).first()
  }

  // Honest empty state before any run.
  get emptyState(): Locator {
    return this.page.getByText('No discovery run yet', { exact: true }).first()
  }
}
