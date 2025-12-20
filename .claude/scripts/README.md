# Claude Code Scripts

Utility scripts for the Claude Code Development Framework.

## Document Sharding

### shard-document.sh

Wrapper script for sharding large markdown documents using `@kayvan/markdown-tree-parser`.

**Usage:**
```bash
./.claude/scripts/shard-document.sh <source-file> <destination-folder>
```

**Example:**
```bash
./.claude/scripts/shard-document.sh docs/PRD.md docs/prd/
```

**What it does:**
1. Validates source file exists
2. Creates destination folder if needed
3. Executes npx @kayvan/markdown-tree-parser
4. Reports results

### verify-sharding.sh

Verify sharded document integrity.

**Usage:**
```bash
./.claude/scripts/verify-sharding.sh <sharded-folder>
```

**Checks:**
- index.md exists
- All section files are valid markdown
- No orphaned files
- Proper heading structure

## Future Scripts

Additional utility scripts will be added here as needed:
- Document combining (reverse sharding)
- Convention validation
- Project structure verification
