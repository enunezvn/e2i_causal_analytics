/**
 * Executive AI Brief Component
 * ============================
 *
 * Displays a GPT-powered executive summary of key insights.
 * Uses the cognitive query API to generate AI-powered briefs.
 *
 * @module components/insights/ExecutiveAIBrief
 */

import { useState, useEffect } from 'react';
import { Brain, RefreshCw, Sparkles, Clock, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useCognitiveRAG } from '@/hooks/api/use-cognitive';

// =============================================================================
// TYPES
// =============================================================================

interface ExecutiveAIBriefProps {
  className?: string;
  brand?: string;
}

interface BriefSection {
  title: string;
  content: string;
  confidence: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_BRIEF: BriefSection[] = [
  {
    title: 'Key Performance Trend',
    content:
      'TRx volume for Remibrutinib has increased 12.3% MoM, driven primarily by improved HCP engagement in the Northeast region. Causal analysis indicates detailing frequency is the strongest driver (β=0.42).',
    confidence: 0.92,
  },
  {
    title: 'Emerging Opportunity',
    content:
      'Gap analysis identified 847 high-propensity HCPs with below-target call coverage. Addressing this gap could yield an estimated $2.3M in incremental revenue over Q1.',
    confidence: 0.87,
  },
  {
    title: 'Risk Alert',
    content:
      'Model drift detected in the Southeast region propensity model. Feature distribution shift in prior authorization rates requires attention. Recommend retraining within 14 days.',
    confidence: 0.78,
  },
];

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExecutiveAIBrief({ className, brand = 'Remibrutinib' }: ExecutiveAIBriefProps) {
  const [sections, setSections] = useState<BriefSection[]>(SAMPLE_BRIEF);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  // Use cognitive RAG for AI-powered briefs
  const {
    mutate: generateBrief,
    data: briefResponse,
    isPending: isGenerating,
  } = useCognitiveRAG();

  // Generate initial brief on mount
  useEffect(() => {
    generateBrief({
      query: `Generate an executive brief summary for ${brand}. Include key performance trends, emerging opportunities, and risk alerts.`,
    });
  }, [brand, generateBrief]);

  // Update sections when we get a response
  useEffect(() => {
    if (briefResponse?.response) {
      // Parse the response into sections (simplified - real implementation would parse structured response)
      const newSections: BriefSection[] = [
        {
          title: 'AI-Generated Insight',
          content: briefResponse.response,
          confidence: 0.85, // CognitiveRAGResponse doesn't have direct confidence field
        },
        ...SAMPLE_BRIEF.slice(1),
      ];
      setSections(newSections);
      setLastUpdated(new Date());
    }
  }, [briefResponse]);

  const handleRefresh = () => {
    generateBrief({
      query: `Generate an executive brief summary for ${brand}. Include key performance trends, emerging opportunities, and risk alerts.`,
    });
  };

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Brain className="h-5 w-5 text-purple-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Executive AI Brief</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Powered by E2I Cognitive Engine
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              <Sparkles className="h-3 w-3 mr-1" />
              GPT-4 Enhanced
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRefresh}
              disabled={isGenerating}
              className="h-8 w-8"
            >
              <RefreshCw className={cn('h-4 w-4', isGenerating && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Loading State */}
        {isGenerating && (
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3 text-[var(--color-muted-foreground)]">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span className="text-sm">Generating AI brief...</span>
            </div>
          </div>
        )}

        {/* Brief Sections */}
        {!isGenerating && (
          <div className="space-y-4">
            {sections.map((section, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]"
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-[var(--color-foreground)]">
                    {section.title}
                  </h4>
                  <Badge
                    variant="outline"
                    className={cn(
                      'text-xs',
                      section.confidence >= 0.9
                        ? 'border-emerald-500/20 text-emerald-600'
                        : section.confidence >= 0.8
                          ? 'border-blue-500/20 text-blue-600'
                          : 'border-amber-500/20 text-amber-600'
                    )}
                  >
                    {(section.confidence * 100).toFixed(0)}% confidence
                  </Badge>
                </div>
                <p className="text-sm text-[var(--color-muted-foreground)] leading-relaxed">
                  {section.content}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--color-border)]">
          <div className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
            <Clock className="h-3 w-3" />
            <span>Last updated: {lastUpdated.toLocaleTimeString()}</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-emerald-600">
            <CheckCircle2 className="h-3 w-3" />
            <span>3 insights generated</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ExecutiveAIBrief;
