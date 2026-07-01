import { Sparkles } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import type { GroundingChip } from '@/types/insights';

interface StrategicInsightCardProps {
  title?: string;
  description?: string;
  insight?: string;
  keyTakeaways?: string[];
  grounding?: GroundingChip[];
  isLoading?: boolean;
  error?: string | null;
  onGenerate?: () => void;
  isFallback?: boolean;
  provenance?: string;
  generatedAt?: string;
}

export function StrategicInsightCard({
  title = 'Strategic Interpretation',
  description = 'Agentic read of this view, grounded in the underlying numbers',
  insight,
  keyTakeaways = [],
  grounding = [],
  isLoading,
  error,
  onGenerate,
  isFallback,
  provenance,
  generatedAt,
}: StrategicInsightCardProps) {
  return (
    <Card className="border-primary/40">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <CardTitle>{title}</CardTitle>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading && (
          <div className="space-y-2" aria-label="Generating insight">
            <div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full animate-pulse rounded bg-muted" />
            <div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
          </div>
        )}
        {!isLoading && error && <p className="text-sm text-destructive">{error}</p>}
        {!isLoading && !error && !insight && onGenerate && (
          <Button variant="outline" onClick={onGenerate}>
            <Sparkles className="mr-2 h-4 w-4" /> Generate strategic insight
          </Button>
        )}
        {!isLoading && !error && insight && (
          <>
            <p className="whitespace-pre-line leading-relaxed">{insight}</p>
            {keyTakeaways.length > 0 && (
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {keyTakeaways.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
            )}
            {grounding.length > 0 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {grounding.map((c, i) => (
                  <span
                    key={i}
                    className="rounded-full border px-2 py-0.5 text-xs text-muted-foreground"
                  >
                    <span className="font-medium">{c.label}</span>: {c.value}
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-center gap-2 pt-1 text-xs text-muted-foreground">
              {isFallback && (
                <span className="rounded bg-muted px-1.5 py-0.5">
                  factual summary — LLM unavailable
                </span>
              )}
              {provenance && <span>{provenance}</span>}
              {generatedAt && <span>· {new Date(generatedAt).toLocaleString()}</span>}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
