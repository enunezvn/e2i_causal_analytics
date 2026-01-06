import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { ROUTES } from '../fixtures/test-data'

/**
 * Page Object Model for Memory Architecture page.
 * Visualizes the E2I tri-memory cognitive architecture with Working, Episodic, Semantic, and Procedural memory.
 */
export class MemoryArchitecturePage extends BasePage {
  readonly url = ROUTES.MEMORY_ARCHITECTURE
  readonly pageTitle = /Memory Architecture|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  // Page Header
  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Memory Architecture/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Tri-Memory|Cognitive|Working|Episodic|Semantic|Procedural/i).first()
  }

  // Status Badge
  get overallStatusBadge(): Locator {
    return this.page.locator('.rounded-full').filter({ hasText: /healthy|degraded|error|unknown/i }).first()
  }

  // Action Buttons
  get refreshButton(): Locator {
    return this.page.getByRole('button', { name: /refresh/i })
  }

  // Architecture Diagram
  get architectureDiagramSection(): Locator {
    return this.page.getByText('Cognitive Memory Architecture').first()
  }

  // Memory Type Cards in Diagram
  get workingMemoryDiagramCard(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Working Memory.*Redis/i }).first()
  }

  get episodicMemoryDiagramCard(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Episodic Memory.*Supabase/i }).first()
  }

  get semanticMemoryDiagramCard(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Semantic Memory.*FalkorDB/i }).first()
  }

  get proceduralMemoryDiagramCard(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Procedural Memory/i }).first()
  }

  // Stats Cards Grid
  get workingMemoryCard(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: 'Working Memory' }).first()
  }

  get episodicMemoryCard(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: 'Episodic Memory' }).first()
  }

  get semanticMemoryCard(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: 'Semantic Memory' }).first()
  }

  get proceduralMemoryCard(): Locator {
    return this.page.locator('.rounded-lg.border').filter({ hasText: 'Procedural Memory' }).first()
  }

  // Backend Labels
  get redisBackend(): Locator {
    return this.page.getByText('Redis').first()
  }

  get supabaseBackend(): Locator {
    return this.page.getByText('Supabase').first()
  }

  get falkordbBackend(): Locator {
    return this.page.getByText('FalkorDB').first()
  }

  // Memory Stats
  get totalMemoriesDisplay(): Locator {
    return this.page.getByText(/Total Memories/i).first()
  }

  get entitiesDisplay(): Locator {
    return this.page.getByText(/Entities/i).first()
  }

  get proceduresDisplay(): Locator {
    return this.page.getByText(/Procedures/i).first()
  }

  // Recent Episodic Memories Section
  get recentMemoriesSection(): Locator {
    return this.page.getByText('Recent Episodic Memories').first()
  }

  get memoryItems(): Locator {
    return this.page.locator('.rounded.border').filter({ hasText: /event_type|Agent/i })
  }

  // About Section
  get aboutSection(): Locator {
    return this.page.getByText('About the Memory System').first()
  }

  get memoryIntegrationInfo(): Locator {
    return this.page.getByText('Memory Integration').first()
  }

  get retrievalMethodsInfo(): Locator {
    return this.page.getByText('Retrieval Methods').first()
  }

  // Query Processing Flow
  get queryProcessingFlow(): Locator {
    return this.page.locator('.rounded-lg').filter({ hasText: /Query Processing|Flow/i }).first()
  }

  // Actions
  async clickRefresh(): Promise<void> {
    await this.refreshButton.click()
  }

  // Verification methods
  async verifyArchitectureDiagramDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Cognitive Memory Architecture').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyMemoryCardsDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Working Memory').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      const memories = ['Episodic Memory', 'Semantic Memory', 'Procedural Memory']
      for (const memory of memories) {
        if (await this.page.getByText(memory).first().isVisible().catch(() => false)) {
          return true
        }
      }
      return false
    }
  }

  async verifyBackendsDisplayed(): Promise<boolean> {
    try {
      // Wait for page content to load
      await this.page.waitForTimeout(1500)

      // Check for backend technology labels - may be combined with storage type
      const redis = await this.page.getByText(/Redis/i).first().isVisible({ timeout: 3000 }).catch(() => false)
      if (redis) return true

      const supabase = await this.page.getByText(/Supabase/i).first().isVisible({ timeout: 2000 }).catch(() => false)
      if (supabase) return true

      const falkordb = await this.page.getByText(/FalkorDB/i).first().isVisible({ timeout: 2000 }).catch(() => false)
      if (falkordb) return true

      // Fallback: check for In-memory cache / pgvector / Graph database text
      const hasCache = await this.page.getByText(/In-memory cache|pgvector|Graph database/i).first().isVisible({ timeout: 1000 }).catch(() => false)
      return hasCache
    } catch {
      return false
    }
  }

  async verifyRecentMemoriesDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('Recent Episodic Memories').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  async verifyAboutSectionDisplayed(): Promise<boolean> {
    try {
      await this.page.getByText('About the Memory System').first().waitFor({ state: 'visible', timeout: 5000 })
      return true
    } catch {
      return false
    }
  }
}
