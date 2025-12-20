# Claude Code Commands Index

This framework includes **43 slash commands** organized into tiers based on usage frequency and complexity.

## Quick Navigation

- [🌟 Essential Commands](#-essential-commands-start-here) (8 commands) - Start here
- [⚡ Common Workflow Commands](#-common-workflow-commands) (12 commands) - Everyday use
- [🔧 Advanced Commands](#-advanced-commands) (23 commands) - Specialized tasks
- [Command Reference by Category](#command-reference-by-category)

---

## 🌟 Essential Commands (Start Here)

These 8 commands cover the core development workflow. Master these first.

### Core PIV Loop

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `/prime` | Load project context | `/prime` |
| `/plan` | Create implementation plan | `/plan "Add user auth"` |
| `/implement` | Execute implementation plan | `/implement .claude/PRPs/features/active/plan.md` |
| `/validate` | Run comprehensive validation | `/validate` |

### Git Workflow

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `/commit` | Create git commit with message | `/commit "Add feature"` |
| `/create-pr` | Create pull request | `/create-pr` |

### Quality & Debugging

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `/code-review` | Comprehensive code review | `/code-review` |
| `/rca` | Root cause analysis for bugs | `/rca "Error description"` |

**📖 Learn More**: See [QUICKSTART.md](../../QUICKSTART.md) for a 5-minute tutorial.

---

## ⚡ Common Workflow Commands

These 12 commands extend the essential workflow with frequently-used features.

### Pull Request Management

| Command | Description | Location |
|---------|-------------|----------|
| `/review-pr` | Review pull request | `exp-piv-loop/review-pr.md` |
| `/merge-pr` | Merge pull request | `exp-piv-loop/merge-pr.md` |

### Bug Fixing

| Command | Description | Location |
|---------|-------------|----------|
| `/fix-rca` | Implement fix from RCA report | `exp-piv-loop/fix-rca.md` |
| `/fix-issue` | Fix GitHub issue end-to-end | `exp-piv-loop/fix-issue.md` |

### Release Management

| Command | Description | Location |
|---------|-------------|----------|
| `/changelog-entry` | Add changelog entry | `exp-piv-loop/changelog-entry.md` |
| `/changelog-release` | Promote changelog to release | `exp-piv-loop/changelog-release.md` |
| `/release` | Create GitHub release | `exp-piv-loop/release.md` |
| `/release-notes` | Generate release notes | `exp-piv-loop/release-notes.md` |

### Validation & Quality

| Command | Description | Location |
|---------|-------------|----------|
| `/code-review-fix` | Fix code review issues | `validation/code-review-fix.md` |
| `/execution-report` | Generate implementation report | `validation/execution-report.md` |
| `/system-review` | Architecture review | `validation/system-review.md` |

### Product Requirements

| Command | Description | Location |
|---------|-------------|----------|
| `/prd` | Create product requirements doc | `exp-piv-loop/prd.md` |

---

## 🔧 Advanced Commands

These 23 commands provide specialized functionality for specific use cases.

### 📐 Architecture & Design

| Command | Description | Location |
|---------|-------------|----------|
| `/create-architecture-documentation` | Generate architecture docs | `create-architecture-documentation.md` |
| `/design-database-schema` | Design database schemas | `design-database-schema.md` |
| `/setup-docker-containers` | Setup Docker configuration | `setup-docker-containers.md` |

### 🗄️ Supabase Integration (6 commands)

| Command | Description | Location |
|---------|-------------|----------|
| `/supabase-backup-manager` | Manage database backups | `supabase-backup-manager.md` |
| `/supabase-data-explorer` | Explore database data | `supabase-data-explorer.md` |
| `/supabase-migration-assistant` | Create/manage migrations | `supabase-migration-assistant.md` |
| `/supabase-performance-optimizer` | Optimize database performance | `supabase-performance-optimizer.md` |
| `/supabase-schema-sync` | Sync schema with Supabase | `supabase-schema-sync.md` |
| `/supabase-type-generator` | Generate TypeScript types | `supabase-type-generator.md` |

### 🌳 Git Worktree Management

| Command | Description | Location |
|---------|-------------|----------|
| `/worktree` | Create git worktrees | `exp-piv-loop/worktree.md` |
| `/worktree-cleanup` | Clean up git worktrees | `exp-piv-loop/worktree-cleanup.md` |

### 🔍 Validation Suite (6 commands)

| Command | Description | Location |
|---------|-------------|----------|
| `/validate` | Comprehensive validation | `validation/validate.md` |
| `/validate-simple` | Quick validation checks | `validation/validate-simple.md` |
| `/validate-python` | Python-specific validation | `validation/validate-python.md` |
| `/validate-typescript` | TypeScript-specific validation | `validation/validate-typescript.md` |
| `/validate-go` | Go-specific validation | `validation/validate-go.md` |
| `/ultimate_validate_command` | Most thorough validation | `validation/ultimate_validate_command.md` |

### 📝 Documentation

| Command | Description | Location |
|---------|-------------|----------|
| `/create-prd` | Create product requirements | `create-prd.md` |
| `/update-docs` | Update project documentation | `update-docs.md` |

### 🎯 Advanced Workflows

| Command | Description | Location |
|---------|-------------|----------|
| `/end-to-end-feature` | Complete feature implementation | `end-to-end-feature.md` |
| `/ultra-think` | Deep analysis and problem solving | `ultra-think.md` |
| `/router` | Route natural language requests | `exp-piv-loop/router.md` |

### 🧪 Core PIV Loop (Alternative Interface)

| Command | Description | Location |
|---------|-------------|----------|
| `/core_piv_loop:prime` | Prime with codebase context | `core_piv_loop/prime.md` |
| `/core_piv_loop:plan-feature` | Create comprehensive plan | `core_piv_loop/plan-feature.md` |
| `/core_piv_loop:execute` | Execute implementation plan | `core_piv_loop/execute.md` |

### 🐛 GitHub Bug Fix Workflow

| Command | Description | Location |
|---------|-------------|----------|
| `/github_bug_fix:rca` | RCA for GitHub issue | `github_bug_fix/rca.md` |
| `/github_bug_fix:implement-fix` | Implement fix from RCA | `github_bug_fix/implement-fix.md` |

### 📋 PRP Workflow

| Command | Description | Location |
|---------|-------------|----------|
| `/prp:prp-core-create` | Create comprehensive PRP | `prp/prp-core-create.md` |
| `/prp:prp-core-execute` | Execute PRP to completion | `prp/prp-core-execute.md` |

---

## Command Reference by Category

### By Workflow Phase

**Planning Phase:**
- `/prime` - Load context
- `/plan` - Create plan
- `/prd` - Create PRD
- `/design-database-schema` - Design schema

**Implementation Phase:**
- `/implement` - Execute plan
- `/worktree` - Create isolated workspace
- `/commit` - Commit changes

**Validation Phase:**
- `/validate` - Run validation
- `/code-review` - Review code
- `/execution-report` - Generate report

**Pull Request Phase:**
- `/create-pr` - Create PR
- `/review-pr` - Review PR
- `/merge-pr` - Merge PR

**Release Phase:**
- `/changelog-entry` - Add entry
- `/changelog-release` - Promote release
- `/release` - Create release
- `/release-notes` - Generate notes

**Bug Fixing Phase:**
- `/rca` - Analyze root cause
- `/fix-rca` - Implement fix
- `/fix-issue` - Fix GitHub issue

### By Tech Stack

**Language-Agnostic:**
- Most commands work with any language
- Customize in `.claude/settings.local.json`

**Python-Specific:**
- `/validate-python` - Python validation template

**TypeScript/Node.js-Specific:**
- `/validate-typescript` - TypeScript validation template

**Go-Specific:**
- `/validate-go` - Go validation template

**Supabase/PostgreSQL:**
- All `/supabase-*` commands (6 total)

**Docker:**
- `/setup-docker-containers`

---

## Learning Path

### Week 1: Master the Essentials
1. Day 1-2: Practice `/prime` + `/plan` + `/implement` + `/validate`
2. Day 3-4: Add `/commit` + `/create-pr`
3. Day 5: Practice `/code-review` + `/rca`

### Week 2: Common Workflows
1. Add `/review-pr` + `/merge-pr` to your workflow
2. Practice `/changelog-entry` + `/release`
3. Try `/fix-issue` for bug fixing

### Week 3+: Specialized Commands
1. Explore Supabase commands if using Supabase
2. Try `/worktree` for parallel development
3. Use `/architecture-documentation` for documentation
4. Experiment with validation templates

---

## Command Naming Conventions

Commands follow these patterns:

1. **Simple actions**: `/command` (e.g., `/prime`, `/commit`)
2. **Namespaced**: `/namespace:command` (e.g., `/exp-piv-loop:plan`)
3. **With arguments**: `/command <arg>` (e.g., `/plan "feature"`)
4. **Optional flags**: `/command [--flag]` (e.g., `/validate --skip-e2e`)

---

## Customization

### For Your Tech Stack

See [CUSTOMIZATION_GUIDE.md](../../CUSTOMIZATION_GUIDE.md) for:
- Setting up validation for your language
- Customizing permissions
- Adding custom commands

### Adding Custom Commands

1. Create `.md` file in appropriate subdirectory
2. Add YAML frontmatter:
   ```yaml
   ---
   description: Command description
   argument-hint: <optional-args>
   ---
   ```
3. Write command instructions
4. Test in your project
5. Add to this index

---

## Quick Command Comparison

### `/plan` vs `/prp:prp-core-create`
- **`/plan`**: Quick, focused implementation plan
- **`/prp:prp-core-create`**: Comprehensive product requirements with research

### `/implement` vs `/end-to-end-feature`
- **`/implement`**: Execute existing plan
- **`/end-to-end-feature`**: Plan + implement + validate in one command

### `/validate` vs `/validate-python`
- **`/validate`**: Generic validation (customize for your project)
- **`/validate-python`**: Pre-built Python/FastAPI validation

### `/rca` vs `/fix-issue`
- **`/rca`**: Just analysis, you implement
- **`/fix-issue`**: Analysis + implementation + PR creation

---

## Command Statistics

- **Total Commands**: 43
- **Essential**: 8 (19%)
- **Common**: 12 (28%)
- **Advanced**: 23 (53%)

**By Category:**
- Core Workflow: 8 commands
- Pull Requests: 3 commands
- Bug Fixing: 3 commands
- Release Management: 4 commands
- Validation: 9 commands
- Supabase: 6 commands
- Architecture: 3 commands
- Git Worktree: 2 commands
- Documentation: 2 commands
- Other: 3 commands

---

## Need Help?

- **Beginner?** Start with [QUICKSTART.md](../../QUICKSTART.md)
- **Need to customize?** See [CUSTOMIZATION_GUIDE.md](../../CUSTOMIZATION_GUIDE.md)
- **Migrating from old version?** See [MIGRATION_GUIDE.md](../../MIGRATION_GUIDE.md)
- **Want full feature list?** See [FRAMEWORK_SUMMARY.md](../../FRAMEWORK_SUMMARY.md)

---

**Last Updated**: 2024-12-15
**Framework Version**: 1.0.0
