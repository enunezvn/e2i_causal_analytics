/**
 * SectionNav — sticky scroll-spy navigation for the Documentation page.
 * Scroll-spy uses IntersectionObserver, guarded so environments without it
 * (jsdom, very old browsers) degrade to click-to-scroll with no highlight.
 */
import { useEffect, useState } from 'react';
import type { DocSection } from './content';

interface SectionNavProps {
  sections: DocSection[];
}

export function SectionNav({ sections }: SectionNavProps) {
  const [activeId, setActiveId] = useState<string>(sections[0]?.id ?? '');

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActiveId(entry.target.id);
        }
      },
      // Trigger when a section's top crosses the upper third of the viewport.
      { rootMargin: '-20% 0px -70% 0px' }
    );
    for (const s of sections) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [sections]);

  return (
    <nav
      aria-label="On this page"
      className="sticky top-0 z-10 -mx-4 mb-8 overflow-x-auto border-b border-[var(--color-border)] bg-[var(--color-background)]/95 px-4 py-2 backdrop-blur"
    >
      <ul className="flex items-center gap-1">
        {sections.map((s) => (
          <li key={s.id}>
            <button
              type="button"
              onClick={() => document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
              aria-current={activeId === s.id ? 'true' : undefined}
              className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors motion-reduce:transition-none ${
                activeId === s.id
                  ? 'bg-[var(--color-primary)]/10 font-medium text-[var(--color-primary)]'
                  : 'text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]'
              }`}
            >
              {s.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
