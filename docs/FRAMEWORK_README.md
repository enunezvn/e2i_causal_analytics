# Claude Code Development Framework - Unified Edition

A comprehensive, production-ready development framework for Claude Code that adapts to your project type - from standard web applications to advanced ML/MLOps systems.

---

## What Is This Framework?

This is a **unified development framework** that provides:

✅ **Software Engineering Best Practices** - Coding standards, testing, error handling, API design
✅ **Agent Reference System** - Quick reference guides Claude consults during development
✅ **Optional Extensions** - Add ML/MLOps, planning systems, or custom capabilities as needed
✅ **Production-Ready Patterns** - Battle-tested conventions for shipping quality code
✅ **Flexible Architecture** - Works for simple projects to complex multi-domain systems

---

## Framework Editions

### Core Framework (Base)
**For**: Web apps, APIs, CLI tools, mobile apps, standard software development

**Includes**:
- Coding patterns and anti-patterns
- Error handling conventions
- Testing strategies
- API design principles
- Security best practices
- Git workflow
- Deployment guidelines

**Use if**: Your project is general software development without ML components.

### ML/MLOps Edition (Extension)
**For**: Machine learning, data pipelines, model training, ML deployment

**Adds to core**:
- ML-specific patterns (data leakage prevention, experiment tracking)
- Specialist guides (model training, feature engineering, MLOps)
- Data and model contracts
- Drift monitoring patterns
- ML testing strategies

**Use if**: Your project involves machine learning or data science.

### Hybrid (Core + ML/MLOps)
**For**: Full-stack ML applications, APIs serving models, web apps with ML features

**You get**:
- All software engineering patterns
- All ML/MLOps patterns
- Both general and ML specialists
- Complete coverage for ML-powered applications

---

## Quick Start

### Installation

#### For Standard Software Projects

```bash
# 1. Copy framework to your project
cp -r path/to/claude-code-framework/.claude your-project/
cp path/to/claude-code-framework/CLAUDE.md your-project/
cp path/to/claude-code-framework/README.md your-project/

# 2. Start developing
cd your-project
claude-code
```

#### For ML/MLOps Projects

```bash
# 1. Copy base framework
cp -r path/to/claude-code-framework/.claude your-ml-project/
cp path/to/claude-code-framework/CLAUDE.md your-ml-project/

# 2. Add ML/MLOps extension
cp -r path/to/claude-code-framework/templates/mlops-extension/.claude/* your-ml-project/.claude/

# 3. Customize ML context
cd your-ml-project
nano .claude/context/summary-mlops.md

# 4. Start developing
claude-code
```

#### For Hybrid Projects

```bash
# Same as ML/MLOps - you get both!
```

---

## What's Included

### Core Framework Components

```
.claude/
├── .agent_docs/              # Agent reference guides
│   ├── anti-patterns.md      # Code smells to avoid
│   ├── coding-patterns.md    # Best practices
│   ├── error-handling.md     # Error conventions
│   ├── testing-patterns.md   # Testing strategies
│   ├── bug-investigation.md  # Debug protocol
│   └── code-review-checklist.md
├── commands/                  # Slash commands
├── hooks/                     # Runtime validation
└── skills/                    # Domain skills

docs/conventions/             # Detailed guides
├── code-style.md
├── api-design.md
├── testing.md
├── performance.md
├── security.md
└── ...

tests/                         # Test structure
CLAUDE.md                      # Orchestrator
README.md                      # This file
```

### ML/MLOps Extension (Optional)

```
templates/mlops-extension/
└── .claude/
    ├── .agent_docs/
    │   └── ml-patterns.md           # ML best practices
    ├── specialists/                  # ML domain experts
    │   ├── model-training.md
    │   ├── data-engineering.md
    │   ├── feature-engineering.md
    │   ├── model-evaluation.md
    │   └── mlops-pipeline.md
    ├── contracts/                    # Interfaces
    │   ├── data-contracts.md
    │   ├── model-contracts.md
    │   └── api-contracts.md
    └── context/
        └── summary-mlops.md         # ML project template
```

---

## Key Features

### 1. Agent Reference System

Claude Code agents automatically consult reference guides during development:

**For all projects**:
- `anti-patterns.md` - Avoid common mistakes
- `coding-patterns.md` - Follow best practices
- `testing-patterns.md` - Write testable code
- `error-handling.md` - Handle errors properly

**For ML projects** (with extension):
- `ml-patterns.md` - Prevent data leakage, track experiments, validate models

### 2. Adaptive Orchestration

The framework adapts to project complexity:

**Simple projects**: Direct execution with pattern references
**Complex projects**: 4-layer orchestration with specialists and contracts

### 3. Optional Extensions

Add capabilities as needed:

- **ML/MLOps Extension** - For machine learning projects
- **Planning System** - For large, long-running projects
- **Custom Extensions** - Build your own

### 4. Production-Ready Conventions

Comprehensive guidelines for:
- Code style and formatting
- API design and versioning
- Testing (unit, integration, e2e, performance)
- Error handling and logging
- Security (OWASP Top 10)
- Deployment and operations
- Database management
- Performance optimization

---

## How It Works

### Example: Standard Web Application

**You**: "Create a REST API for user management"

**Claude**:
1. References `coding-patterns.md` for best practices
2. Follows `api-design.md` conventions
3. Implements with:
   - Proper error handling
   - Input validation
   - RESTful design
   - Security best practices
4. Adds tests following `testing-patterns.md`

### Example: ML Model Training (With Extension)

**You**: "Train a classification model for customer churn"

**Claude**:
1. Loads `model-training.md` specialist
2. References `ml-patterns.md` for ML best practices
3. Checks `data-contracts.md` for schemas
4. Implements with:
   - Proper train/test split (no data leakage)
   - Cross-validation
   - MLflow experiment tracking
   - Multiple metrics
   - Model validation
5. Tests following ML testing patterns

### Example: Full-Stack ML Application

**You**: "Build an API that serves the churn prediction model"

**Claude**:
1. Uses **both** general and ML patterns
2. API layer: General framework conventions
3. ML layer: ML/MLOps specialist patterns
4. Integration: Validates contracts between layers
5. Result: Production-ready ML-powered API

---

## Technology Stack Support

### General Software

- **Backend**: Node.js, Python (FastAPI/Django), Go, Java, .NET
- **Frontend**: React, Vue, Angular, Svelte
- **Database**: PostgreSQL, MySQL, MongoDB, Redis
- **APIs**: REST, GraphQL, gRPC

### ML/MLOps (With Extension)

- **ML Frameworks**: scikit-learn, PyTorch, TensorFlow, XGBoost
- **Experiment Tracking**: MLflow, Weights & Biases
- **Feature Stores**: Feast, Tecton
- **Model Serving**: FastAPI, TensorFlow Serving, Seldon
- **Orchestration**: Airflow, Prefect, Kubeflow

**All customizable** - The framework includes templates you adapt to your stack.

---

## Project Structure Flexibility

The framework adapts to your project structure:

### Minimal Setup
```
my-project/
├── .claude/              # Just the framework
├── src/
└── CLAUDE.md
```

### Standard Project
```
my-project/
├── .claude/              # Framework + conventions
├── .github/              # CI/CD workflows
├── src/
├── tests/
├── docs/
└── CLAUDE.md
```

### ML/MLOps Project
```
my-ml-project/
├── .claude/              # Framework + ML extension
│   ├── specialists/      # ML domain experts
│   ├── contracts/        # Data/model contracts
│   └── context/          # ML project context
├── config/               # ML configurations
├── data/                 # Data directory
├── models/               # Model artifacts
├── notebooks/            # Jupyter notebooks
├── src/
└── CLAUDE.md
```

---

## Extensions

### ML/MLOps Extension

Add machine learning best practices to your project.

**What you get**:
- Data leakage prevention patterns
- Experiment tracking integration
- Model validation and governance
- ML-specific testing strategies
- Specialist guides for ML tasks

**Install**:
```bash
cp -r templates/mlops-extension/.claude/* .claude/
```

**See**: [templates/mlops-extension/README.md](templates/mlops-extension/README.md)

### Planning System Extension

For large, complex, long-running projects.

**What you get**:
- Automated documentation maintenance
- Event-driven updates
- Perfect session continuity
- Structured planning documents

**See**: [templates/planning-system-for-complex-projects/README.md](templates/planning-system-for-complex-projects/README.md)

---

## Customization

The framework is designed to be customized:

### Quick Customization (15 minutes)

1. **Update project info** in `CLAUDE.md`
2. **Copy conventions** from `docs/conventions/` and adapt
3. **Add team patterns** to `.claude/.agent_docs/`

### Deep Customization (1-2 hours)

1. **Create custom specialists** for your domains
2. **Define contracts** for your architecture
3. **Add custom commands** in `.claude/commands/`
4. **Configure hooks** for runtime validation

See [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) for details.

---

## Comparison with Other Frameworks

| Feature | This Framework | Generic Templates | DIY Approach |
|---------|----------------|-------------------|--------------|
| **Software Best Practices** | ✅ Comprehensive | ⚠️ Basic | ❌ Manual |
| **ML/MLOps Patterns** | ✅ Optional extension | ❌ None | ❌ Manual |
| **Agent Integration** | ✅ Native | ❌ None | ❌ N/A |
| **Adaptive Complexity** | ✅ Simple → Complex | ❌ One size | ❌ Manual |
| **Production Ready** | ✅ Battle-tested | ⚠️ Varies | ❌ Case-by-case |
| **Extensible** | ✅ Modular | ❌ Monolithic | ✅ Fully custom |
| **Maintenance** | ✅ Community updates | ⚠️ Stale | ❌ DIY |

---

## Use Cases

### Perfect For

✅ Starting new projects with best practices
✅ Standardizing team development practices
✅ Training Claude Code on your patterns
✅ Building production-ready systems
✅ ML/MLOps projects requiring rigorous patterns
✅ Full-stack applications with ML components

### Not Ideal For

❌ Quick prototypes (too much overhead)
❌ Projects with radically different patterns (would need heavy customization)
❌ Teams unwilling to follow conventions

---

## Getting Help

### Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 30-minute setup guide
- **[CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md)** - Detailed customization
- **[CLAUDE.md](CLAUDE.md)** - Orchestrator and architecture
- **[templates/mlops-extension/README.md](templates/mlops-extension/README.md)** - ML extension guide

### Agent Reference

- **[.claude/.agent_docs/](.claude/.agent_docs/)** - Quick reference for patterns
- **[docs/conventions/](docs/conventions/)** - Detailed conventions

### Common Questions

**Q: Do I need to install everything?**
A: No! Start with core, add extensions as needed.

**Q: Can I use just the ML extension without the core?**
A: No, ML extension builds on core framework.

**Q: How do I update the framework?**
A: Copy new version files, merge with your customizations.

**Q: Can I share this across projects?**
A: Yes! Keep one copy, install to each project.

---

## Contributing

This framework is designed for customization and sharing:

1. **Customize for your team** - Add your patterns and conventions
2. **Share improvements** - Document patterns that work well
3. **Create extensions** - Build new capabilities for specific domains

---

## Version History

### v3.0 - Unified Edition (2025-12-17)
- **Breaking change**: Merged general and ML/MLOps frameworks
- ML/MLOps is now an optional extension
- Adaptive orchestration (simple to complex)
- Technology-agnostic design
- Modular extension system

### v2.0 - ML/MLOps Edition (Previous)
- ML-specific framework
- Data leakage prevention
- Experiment tracking patterns
- Model governance

### v1.0 - General Framework (Original)
- Core software engineering conventions
- Agent reference system
- Testing and deployment patterns

---

## License

This framework is provided as-is for use in your projects.

---

## Quick Decision Tree

**What should I install?**

```
Are you building ML/AI features?
├─ No → Install core framework only
│        cp -r .claude/ your-project/
│        cp CLAUDE.md your-project/
│
└─ Yes → Install core + ML extension
         cp -r .claude/ your-project/
         cp -r templates/mlops-extension/.claude/* your-project/.claude/
         cp CLAUDE.md your-project/

Is your project large/complex (3+ months)?
└─ Yes → Consider planning system extension
         See: templates/planning-system-for-complex-projects/
```

---

## Next Steps

1. **Read the quickstart**: [QUICKSTART.md](QUICKSTART.md)
2. **Install to a project**: Follow instructions above
3. **Customize**: Adapt conventions to your stack
4. **Start coding**: Claude will follow your framework!

---

**Ready to get started?**

```bash
# Standard project
cp -r claude-code-framework/.claude my-project/
cp claude-code-framework/CLAUDE.md my-project/

# ML/MLOps project
cp -r claude-code-framework/.claude my-ml-project/
cp -r claude-code-framework/templates/mlops-extension/.claude/* my-ml-project/.claude/
cp claude-code-framework/CLAUDE.md my-ml-project/

# Start developing
cd my-project  # or my-ml-project
claude-code
```

---

**Version**: 3.0 - Unified Edition
**Last Updated**: 2025-12-17
**Maintained by**: The development community
