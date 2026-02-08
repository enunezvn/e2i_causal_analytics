# OOM Killer Fix - Quick Reference

## Problem
VSCode/Claude Code crashes (OOM Killer) when opening this 5GB+ project with heavy dependencies.

## Solution Options

### Option A: Open Subdirectories Only (RECOMMENDED)

**Instead of opening the entire project, open specific subdirectories:**

```bash
# For frontend work
code frontend/

# For backend/Python work
code src/

# For specific features
code src/agents/
```

This reduces memory usage by 80%+ since VSCode only indexes the opened directory.

### Option B: Temporary Directory Hiding (if you must open root)

**Before opening VSCode:**

```bash
cd /home/enunez/Projects/e2i_causal_analytics
bash .vscode/hide-heavy-dirs.sh
```

**Open VSCode/Claude Code**

**After VSCode loads successfully:**

```bash
bash .vscode/restore-heavy-dirs.sh
```

This temporarily renames `.venv` and `node_modules` so VSCode doesn't see them during startup.

### Option C: Increase WSL2 Memory (if using WSL2)

1. Create or edit `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
memory=16GB
processors=4
swap=8GB
```

2. Restart WSL2:

```powershell
wsl --shutdown
```

Then reopen WSL2 and try again.

### Option D: Use Remote Development

Open VSCode remotely via SSH or WSL Remote extension, which has better memory management for large projects.

## What's Been Done

✅ Created `.vscode/settings.json` with aggressive exclusions
✅ Created `.claudeignore` for Claude Code
✅ Disabled Python language server (`python.languageServer: "None"`)
✅ Disabled TypeScript auto-acquisition
✅ Reduced file watcher limits
✅ Cleaned up 900MB of cache and build artifacts

## Files Created/Modified

- `.vscode/settings.json` - VSCode exclusions and memory limits
- `.claudeignore` - Claude Code exclusions
- `.vscode/hide-heavy-dirs.sh` - Script to hide directories
- `.vscode/restore-heavy-dirs.sh` - Script to restore directories
- `.gitignore` - Added `.hidden_*` patterns

## Current Project Size

- **Total**: ~4.7GB
- **frontend/node_modules**: 993MB
- **.venv**: 3.1GB
- **Source code**: ~600MB

## Monitoring

Check memory usage:
```bash
free -h
watch -n 1 free -h
```

Check VSCode processes:
```bash
ps aux | grep code | sort -k4 -rn
```

## Need More Help?

If still experiencing issues, consider:
1. Splitting the repository into frontend/backend repos
2. Using Docker dev containers
3. Upgrading system RAM
4. Using a cloud development environment (GitHub Codespaces, Gitpod)
