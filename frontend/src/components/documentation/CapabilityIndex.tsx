/**
 * CapabilityIndex — "where do I go for each question?"
 * Groups and page titles come from getNavigationSections() at render time, so
 * a retired page can never appear here and a new page shows up automatically
 * (content.test.ts fails the build until it gets a question).
 */
import { Link } from 'react-router-dom';
import { getNavigationSections } from '@/router/routes';
import { CAPABILITY_EXEMPT_PATHS, CAPABILITY_QUESTIONS, LEVEL_LABELS } from './content';

export function CapabilityIndex() {
  const groups = getNavigationSections()
    .filter((g) => g.label !== null)
    .map((g) => ({
      ...g,
      routes: g.routes.filter((r) => !(CAPABILITY_EXEMPT_PATHS as readonly string[]).includes(r.path)),
    }))
    .filter((g) => g.routes.length > 0);

  return (
    <section aria-label="Where to go for each question" className="space-y-6">
      <h3 className="text-sm font-semibold text-[var(--color-foreground)]">
        Where to go for each question
      </h3>
      {groups.map((group) => (
        <div key={group.key}>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]">
            {group.label}
          </h4>
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {group.routes.map((route) => {
              const info = CAPABILITY_QUESTIONS[route.path];
              return (
                <li
                  key={route.path}
                  className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-3 transition-colors hover:border-[var(--color-primary)]/50 motion-reduce:transition-none"
                >
                  <div className="flex items-start justify-between gap-2">
                    <Link
                      to={route.path}
                      className="text-sm font-medium text-[var(--color-primary)] hover:underline"
                    >
                      {route.title}
                    </Link>
                    {info?.levels && (
                      <span className="flex shrink-0 gap-1">
                        {info.levels.map((level) => (
                          <span
                            key={level}
                            className="rounded-full border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
                          >
                            {LEVEL_LABELS[level]}
                          </span>
                        ))}
                      </span>
                    )}
                  </div>
                  {info && (
                    <p className="mt-1 text-xs leading-5 text-[var(--color-muted-foreground)]">
                      {info.question}
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </section>
  );
}
