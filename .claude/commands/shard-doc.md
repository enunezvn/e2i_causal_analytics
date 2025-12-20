# Document Sharding Command

Split large markdown documents into smaller, organized files based on level 2 headings for improved token efficiency and selective loading.

## Purpose

Document sharding helps manage large documentation files (>20k tokens) by:
- Splitting documents by ## headings into separate files
- Creating an index.md for navigation and structure
- Enabling selective loading (load only needed sections)
- Reducing token usage by 70-90% for large projects

## Usage

```
/shard-doc
```

The assistant will guide you through:
1. Selecting the source document to shard
2. Choosing the destination folder
3. Executing the sharding process
4. Handling the original document

## When to Use

**Ideal for:**
- Large PRDs, architecture documents (>20k tokens)
- Multi-epic planning documents
- Extensive convention documentation
- Documents with many distinct sections

**Skip for:**
- Small documents (<10k tokens)
- Frequently updated work-in-progress docs
- Documents where whole-file context is essential

## How It Works

Uses `@kayvan/markdown-tree-parser` to automatically:
1. Parse markdown by level 2 headings (`## Section Name`)
2. Create individual files (e.g., `section-name.md`)
3. Generate `index.md` with table of contents and section descriptions
4. Organize files in a dedicated folder

## Example

**Before:**
```
docs/PRD.md (45k tokens, 15 epics)
```

**After:**
```
docs/prd/
├── index.md (navigation + descriptions)
├── overview.md
├── epic-1-authentication.md
├── epic-2-dashboard.md
└── ... (12 more sections)
```

**Benefit:** Agents can load only relevant sections, saving 90% tokens when working on specific epics.

## Instructions for Assistant

When this command is executed, follow these steps:

### Step 1: Get Source Document

```xml
<action>Ask user for the source document path if not provided</action>
<action>Verify file exists and is markdown (.md)</action>
<action if="file not found">HALT with error message</action>
```

### Step 2: Determine Destination

```xml
<action>Calculate default destination: same location, folder named after file</action>
<example>docs/architecture.md → docs/architecture/</example>
<action>Ask user: "Destination folder? [y] for default: {suggested-path}, or provide custom path"</action>
<action>Verify write permissions</action>
<action if="permission denied">HALT with error message</action>
```

### Step 3: Execute Sharding

```xml
<action>Inform user sharding is beginning</action>
<action>Run: npx @kayvan/markdown-tree-parser explode [source] [destination]</action>
<action>Capture output and errors</action>
<action if="command fails">HALT and display error</action>
```

### Step 4: Verify Output

```xml
<action>Check destination folder contains sharded files</action>
<action>Verify index.md was created</action>
<action>Count section files created</action>
<action if="no files">HALT with error message</action>
```

### Step 5: Report Completion

Display to user:
- Source document path
- Destination folder path
- Number of section files created
- Confirmation that index.md was created
- Any warnings from the tool

### Step 6: Handle Original Document

**Critical:** Keeping both original and sharded versions defeats the purpose and causes confusion.

Present options:
```
What should we do with the original document?

[d] Delete - Remove original (recommended - can always recombine from shards)
[m] Move to archive - Move to backup location
[k] Keep - Leave in place (NOT recommended - creates confusion)

Your choice (d/m/k):
```

**If delete:**
- Delete the original file
- Confirm: "✓ Original document deleted: {path}"

**If move:**
- Ask for archive location (default: same dir + /archive/ subfolder)
- Create archive directory if needed
- Move file and confirm

**If keep:**
- Display warning about confusion with duplicate content
- Confirm original kept at location

## Integration with Existing Workflows

### Discovery Pattern

Agents should use this discovery order when looking for documents:

1. **Check for whole document:** `{folder}/*document-name*.md`
2. **Check for sharded version:** `{folder}/*document-name*/index.md`
3. **Priority:** Whole document takes precedence if both exist

### Selective Loading

When working on specific sections (e.g., Epic 3):
```
1. Read index.md to understand structure
2. Load ONLY the needed section file
3. Skip all other sections
4. Result: 90% token reduction for 10+ section documents
```

### Full Loading

When needing complete context:
```
1. Read index.md for structure
2. Read ALL section files
3. Treat as single combined document
```

## Troubleshooting

**Both whole and sharded exist:**
- Workflows will use whole document (priority rule)
- Delete or archive the one you don't want

**Index.md out of sync:**
- Delete sharded folder
- Re-run /shard-doc on original

**Sections too granular:**
- Combine sections in original document
- Use fewer level 2 headings
- Re-shard

## Related Features

- Use with `/create-prd` for large product requirements
- Use with `/create-architecture-documentation` for complex systems
- Complements the optional planning system for large projects

## Prerequisites

The `@kayvan/markdown-tree-parser` package will be installed automatically when needed via npx.

---

**Best Practice:** Shard documents AFTER planning phase is complete, BEFORE implementation phase begins.
