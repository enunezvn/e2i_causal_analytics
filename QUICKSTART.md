# Quick Start Guide - Unified Framework

Get your project set up with the Claude Code Framework in under 10 minutes.

---

## Prerequisites

- Claude Code CLI installed
- A project directory (or create a new one)
- 10 minutes of your time

---

## Option 1: Automated Installation (Recommended)

### For Standard Software Projects

```bash
# Navigate to framework directory
cd path/to/claude-code-framework

# Install to your project
./install.sh /path/to/your-project

# Start developing
cd /path/to/your-project
claude-code
```

### For ML/MLOps Projects

```bash
# Install with ML extension
./install.sh /path/to/your-ml-project --with-mlops

# Customize ML context
cd /path/to/your-ml-project
nano .claude/context/summary-mlops.md

# Start developing
claude-code
```

### For Large/Complex Projects

```bash
# Install with ML and planning extensions
./install.sh /path/to/your-project --with-mlops --with-planning
```

---

## Option 2: Manual Installation

### Standard Projects

```bash
# 1. Copy core framework
cp -r claude-code-framework/.claude your-project/
cp claude-code-framework/CLAUDE.md your-project/
cp claude-code-framework/README.md your-project/

# 2. (Optional) Copy docs and test structure
cp -r claude-code-framework/docs your-project/
cp -r claude-code-framework/tests your-project/

# 3. Start Claude Code
cd your-project
claude-code
```

### ML/MLOps Projects

```bash
# 1. Copy core framework
cp -r claude-code-framework/.claude your-ml-project/
cp claude-code-framework/CLAUDE.md your-ml-project/

# 2. Add ML/MLOps extension
cp -r claude-code-framework/templates/mlops-extension/.claude/* your-ml-project/.claude/

# 3. Customize
cd your-ml-project
mv .claude/context/summary-mlops.md .claude/context/summary.md
nano .claude/context/summary.md  # Edit with your project details

# 4. Start Claude Code
claude-code
```

---

## 5-Minute Test

After installation, test that everything works:

### 1. Start Claude Code

```bash
cd your-project
claude-code
```

### 2. Ask Claude

For standard projects:
```
"Explain the project architecture based on the framework"
```

For ML projects:
```
"Help me train a classification model following the framework patterns"
```

### 3. Verify

Claude should:
- Reference the framework conventions
- Follow the patterns in `.claude/.agent_docs/`
- For ML: Use ML-specific patterns and specialists

---

## What You Get

### All Projects
✅ Core coding patterns and anti-patterns
✅ Error handling conventions
✅ Testing strategies
✅ Agent reference guides
✅ Git workflow conventions
✅ Security best practices

### ML/MLOps Projects (With Extension)
✅ Data leakage prevention patterns
✅ Experiment tracking integration
✅ Model validation and governance
✅ ML-specific testing
✅ Specialist guides for ML tasks
✅ Data and model contracts

---

## Next Steps

### Customize for Your Project

#### 1. Update Project Info (5 minutes)

Edit `CLAUDE.md`:
```bash
nano CLAUDE.md
```

Update the project description and technology stack.

#### 2. Customize Conventions (Optional, 15-30 minutes)

Review and adapt conventions in `docs/conventions/`:
```bash
ls docs/conventions/
# Edit files to match your team's standards
```

#### 3. Add Team Patterns (Optional)

Add your team's specific patterns to `.claude/.agent_docs/`:
```bash
nano .claude/.agent_docs/coding-patterns.md
# Add your team's patterns
```

### For ML Projects

#### 1. Define Project Context (10 minutes)

```bash
nano .claude/context/summary.md
```

Fill in:
- Project name and domain
- Data sources and characteristics
- Model architecture
- Success metrics
- Critical constraints

#### 2. Define Data Contracts (15 minutes)

```bash
nano .claude/contracts/data-contracts.md
```

Update with:
- Your data schemas
- Data quality rules
- Required vs optional fields
- Validation rules

#### 3. Customize ML Specialists (Optional)

```bash
nano .claude/specialists/model-training.md
```

Update examples to match your ML stack (PyTorch vs TensorFlow, MLflow vs W&B, etc.)

---

## Common Tasks

### Task 1: Start a New Feature

**Standard Project**:
```
"Create a REST API endpoint for user authentication"
```

Claude will:
- Follow API design conventions
- Add proper error handling
- Include input validation
- Add tests

**ML Project**:
```
"Train a model for customer churn prediction"
```

Claude will:
- Load model-training specialist
- Prevent data leakage
- Set up experiment tracking
- Validate model performance
- Add ML-specific tests

### Task 2: Review Code

```
"Review this file for issues: src/api/users.ts"
```

Claude will use the code-review-checklist to check for:
- Code quality
- Security issues
- Performance problems
- Test coverage

### Task 3: Debug an Issue

```
"Help me debug this error: [paste error]"
```

Claude will follow the bug-investigation protocol:
- Gather context
- Reproduce the issue
- Identify root cause
- Suggest fix
- Add regression test

---

## Troubleshooting

### Claude doesn't reference the framework

**Solution**: Verify files are in the correct location:
```bash
ls .claude/.agent_docs/
ls CLAUDE.md
```

If missing, reinstall:
```bash
./path/to/claude-code-framework/install.sh .
```

### Claude uses wrong patterns

**Solution**: Check which files are loaded. For ML projects, ensure extension is installed:
```bash
ls .claude/specialists/model-training.md
ls .claude/.agent_docs/ml-patterns.md
```

If missing:
```bash
cp -r /path/to/claude-code-framework/templates/mlops-extension/.claude/* .claude/
```

### Framework feels too heavy

**Solution**: You can use a minimal setup:
```bash
# Keep only essentials
.claude/.agent_docs/
CLAUDE.md
README.md
```

Remove everything else. The framework adapts to what you have.

---

## Learning Path

### Day 1: Installation & Basic Usage
- ✅ Install framework
- ✅ Test with simple tasks
- ✅ Review agent docs

### Week 1: Customization
- ✅ Update project context
- ✅ Define contracts (ML projects)
- ✅ Adapt conventions

### Month 1: Team Adoption
- ✅ Add team-specific patterns
- ✅ Create custom commands
- ✅ Share across projects

---

## Getting Help

### Documentation

- **README.md** - Framework overview
- **CUSTOMIZATION_GUIDE.md** - Detailed customization
- **CLAUDE.md** - Orchestrator and architecture
- **templates/mlops-extension/README.md** - ML extension guide

### Quick References

- `.claude/.agent_docs/` - Agent reference guides
- `docs/conventions/` - Detailed conventions

### Ask Claude

```
"How do I customize the framework for [task]?"
"Show me an example of [pattern] from the framework"
"Explain the orchestration architecture"
```

---

## Examples

### Example 1: Web API Project

```bash
# Install
./install.sh ~/projects/my-api

# Test
cd ~/projects/my-api
claude-code
```

Ask Claude:
```
"Create a REST API with user authentication, products CRUD, and error handling"
```

Result: Production-ready API following all framework conventions.

### Example 2: ML Project

```bash
# Install with ML extension
./install.sh ~/projects/ml-churn --with-mlops

# Customize
cd ~/projects/ml-churn
nano .claude/context/summary.md  # Fill in project details
nano .claude/contracts/data-contracts.md  # Define schemas
```

Ask Claude:
```
"Build a complete churn prediction pipeline: data loading, feature engineering, model training, and API endpoint"
```

Result: Production-ready ML pipeline with:
- No data leakage
- Experiment tracking
- Model validation
- API endpoint
- Tests

### Example 3: Full-Stack ML App

```bash
# Install with ML extension
./install.sh ~/projects/ml-app --with-mlops
cd ~/projects/ml-app
```

Ask Claude:
```
"Create a web app with React frontend, FastAPI backend, and ML model for recommendations"
```

Result: Full-stack application using both general and ML patterns.

---

## Success Checklist

After setup, you should be able to:

- [ ] Ask Claude about the framework and get accurate responses
- [ ] Have Claude generate code following framework patterns
- [ ] See proper error handling in generated code
- [ ] Get tests automatically added with new features
- [ ] For ML: See data leakage prevention in model training code
- [ ] For ML: See experiment tracking setup in training code
- [ ] For ML: Get model validation against thresholds

---

## Quick Reference Commands

```bash
# Install core framework
./install.sh /path/to/project

# Install with ML/MLOps
./install.sh /path/to/project --with-mlops

# Install with all extensions
./install.sh /path/to/project --with-mlops --with-planning

# View what's installed
ls -la .claude/
ls .claude/specialists/  # ML extension check
```

---

## Time Investment

- **Installation**: 2 minutes (automated) or 5 minutes (manual)
- **Basic customization**: 10-15 minutes
- **Deep customization**: 1-2 hours
- **Team adoption**: Ongoing

**Total to get started**: 10-15 minutes

---

## What's Next?

1. ✅ Framework installed
2. ✅ Basic test completed
3. 📚 Read [README.md](README.md) for full feature overview
4. 🎨 Read [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) for deep customization
5. 🚀 Start building your project!

---

**Ready? Let's go!**

```bash
cd /path/to/claude-code-framework
./install.sh /path/to/your-project --with-mlops  # if ML project
cd /path/to/your-project
claude-code
```

Ask Claude:
```
"Let's build something amazing following the framework conventions!"
```

---

**Version**: 3.0
**Last Updated**: 2025-12-17
