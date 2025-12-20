# Product Development Documentation

This directory contains product requirements documents (PRDs), product strategy, and feature planning documentation for E2I Causal Analytics.

---

## Directory Structure

```
product-development/
├── README.md                    # This file
├── current-feature/             # Active feature being planned
│   ├── PRD.md                   # Current PRD (E2I Platform v4.2.0)
│   ├── JTBD.md                  # Jobs to be Done (optional)
│   └── feature.md               # Feature brief (optional)
├── resources/                   # Templates and reference materials
│   ├── PRD-template.md          # Template for new PRDs
│   └── product.md               # Product overview (optional)
└── archive/                     # Historical PRDs (to be created as needed)
```

---

## Purpose

### Why Product Development Docs?

Product documentation serves as the bridge between business strategy and technical implementation:

1. **Alignment**: Ensures stakeholders, product, engineering, and design are aligned on what and why
2. **Decision Record**: Documents key decisions, alternatives considered, and rationale
3. **Communication**: Provides clear requirements for engineering and design teams
4. **Validation**: Defines success metrics and acceptance criteria
5. **Historical Context**: Maintains institutional knowledge as team evolves

### When to Create a PRD

Create a PRD for:
- ✅ Major new features or product initiatives
- ✅ Significant enhancements to existing functionality
- ✅ Complex features requiring cross-functional coordination
- ✅ Features with business/regulatory implications
- ✅ Platform-level documentation (like the comprehensive E2I PRD)

Skip PRDs for:
- ❌ Minor bug fixes
- ❌ Small UI tweaks
- ❌ Internal refactoring with no user-facing changes

---

## How to Use This Directory

### Creating a New Feature PRD

**Method 1: Using Claude Code Command**

```bash
# For a new feature
/create-prd "Feature name or description"

# Interactive mode (asks questions)
/create-prd --interactive

# Show template only
/create-prd --template
```

**Method 2: Manual Process**

1. Copy the PRD template:
   ```bash
   cp product-development/resources/PRD-template.md \
      product-development/current-feature/PRD.md
   ```

2. Fill in the template sections based on:
   - User research and feedback
   - Market analysis
   - Technical constraints
   - Business objectives

3. Review with stakeholders:
   - Product leadership
   - Engineering leadership
   - Design team
   - Key stakeholders

4. Get approval and move to implementation planning

### PRD Workflow

```
1. Identify Need
   ↓
2. Create PRD (using template or command)
   ↓
3. Research & User Feedback
   ↓
4. Draft PRD Content
   ↓
5. Stakeholder Review
   ↓
6. Incorporate Feedback
   ↓
7. Approval
   ↓
8. Create Implementation PRP
   (Use /prp:prp-core-create)
   ↓
9. Development
   ↓
10. Archive PRD
```

---

## Current PRD

### E2I Causal Analytics Platform PRD (v1.0)

**Location**: `current-feature/PRD.md`

**Purpose**: Comprehensive product documentation for the entire E2I Causal Analytics platform (v4.2.0)

**Status**: Complete - serves as definitive product reference

**Key Sections**:
- Executive Summary
- Product Vision & Strategy
- Market & User Analysis
- Feature Specifications (18 agents, causal inference, digital twins, etc.)
- Success Metrics & KPIs
- Technical Architecture (high-level)
- Roadmap (v4.3.0 → v5.0.0 → v6.0+)

**Use Cases for This PRD**:
- Onboarding new team members
- Executive/investor presentations
- Partner discussions
- Strategic planning
- Feature prioritization
- Compliance/regulatory review

---

## Best Practices

### Writing Effective PRDs

**Do**:
- ✅ Focus on the "what" and "why", not the "how" (implementation)
- ✅ Use specific, measurable success criteria
- ✅ Include user stories and jobs to be done
- ✅ Document alternatives considered
- ✅ Define clear acceptance criteria
- ✅ Include concrete examples and use cases
- ✅ Keep technical architecture high-level

**Don't**:
- ❌ Include detailed implementation plans (use PRPs for that)
- ❌ Provide time estimates (product, not project management)
- ❌ Write requirements in isolation without user research
- ❌ Use jargon without definitions
- ❌ Skip success metrics
- ❌ Forget about edge cases and error scenarios

### PRD vs PRP

**PRD (Product Requirements Document)**:
- **Purpose**: What and why
- **Audience**: Product, design, business stakeholders
- **Focus**: User needs, business value, feature specifications
- **Created by**: Product Manager
- **When**: Before implementation planning

**PRP (Product Requirements Plan)**:
- **Purpose**: How (implementation)
- **Audience**: Engineering team
- **Focus**: Technical tasks, validation, code patterns
- **Created by**: Technical lead or Claude Code (via `/prp:prp-core-create`)
- **When**: After PRD approval, before coding

**Workflow**:
```
PRD (Product decides what) → PRP (Engineering plans how) → Implementation
```

---

## Templates

### Available Templates

1. **PRD-template.md** - Full feature PRD template
   - Location: `resources/PRD-template.md`
   - Use for: New features, major enhancements
   - Sections: 15+ comprehensive sections

### Template Sections Explained

**Executive Summary**: High-level overview for busy stakeholders

**Problem Statement**: What problem are we solving and why it matters

**Solution Overview**: Proposed approach and alternatives considered

**User Stories**: Specific user needs in user story format

**Feature Specifications**: Detailed functional and non-functional requirements

**Success Metrics**: How we'll measure success (adoption, engagement, business impact)

**Technical Considerations**: High-level architecture, risks, dependencies

**Release Strategy**: Phased rollout and feature flags

**Testing Strategy**: What testing is required

**Acceptance Criteria**: Definition of done

---

## Integration with Development Workflow

### From PRD to Production

```
1. PRD Creation
   └─> /create-prd "feature"

2. PRD Review & Approval
   └─> Stakeholder sign-off

3. Implementation Planning
   └─> /prp:prp-core-create "feature from PRD"

4. Development
   └─> Follow PRP tasks

5. Testing & Validation
   └─> Execute validation commands from PRP

6. Launch
   └─> Monitor success metrics from PRD

7. Retrospective
   └─> Update PRD with learnings
```

### Connecting PRD and PRP

When creating a PRP from a PRD:

```bash
# Reference the PRD in your PRP creation
/prp:prp-core-create "Implement [feature] as specified in product-development/current-feature/PRD.md"
```

The PRP will:
- Reference the PRD for context
- Translate product requirements into technical tasks
- Add implementation details (code patterns, validation)
- Define step-by-step execution plan

---

## Version Control

### PRD Versioning

Use semantic versioning for PRDs:
- **Major version (1.0 → 2.0)**: Significant scope change or pivot
- **Minor version (1.0 → 1.1)**: New requirements or features added
- **Patch version (1.1 → 1.1.1)**: Clarifications, corrections

### Archiving Old PRDs

When a feature is complete and a new version is planned:

```bash
# Create archive directory
mkdir -p product-development/archive/

# Move old PRD
mv product-development/current-feature/PRD.md \
   product-development/archive/PRD-v1.0-YYYY-MM-DD.md
```

---

## Stakeholder Communication

### Who Needs to Review PRDs?

**Required Reviewers**:
- Product Management (owner)
- Engineering Leadership (feasibility)
- Design (user experience)

**Optional Reviewers** (depending on feature):
- Security team (if security implications)
- Compliance (if regulatory implications)
- Marketing (if GTM considerations)
- Customer success (if impacts customer experience)
- Executive sponsor (for major initiatives)

### Review Checklist

Before considering a PRD approved:

- [ ] Problem statement is clear and validated with users
- [ ] Solution addresses the problem effectively
- [ ] Success metrics are specific and measurable
- [ ] User stories cover all key personas
- [ ] Acceptance criteria are testable
- [ ] Technical risks are identified
- [ ] Dependencies are documented
- [ ] All stakeholders have reviewed
- [ ] Open questions are resolved or have owners

---

## FAQs

### Q: Do we need a PRD for every feature?

**A**: No. Use PRDs for significant features, complex changes, or when alignment is needed. For small changes, a simple GitHub issue or Slack discussion may suffice.

### Q: How detailed should the technical section be?

**A**: High-level only. Include architecture approach, technology choices, and major technical risks. Detailed implementation goes in the PRP.

### Q: Can PRDs change after approval?

**A**: Yes, but increment the version and track changes. Major changes may require re-approval.

### Q: Who owns the PRD?

**A**: Product Manager is the owner, but it's a collaborative document with input from engineering, design, and stakeholders.

### Q: How does this relate to agile user stories?

**A**: PRDs provide the context and rationale. User stories in the backlog reference the PRD and break features into implementable chunks.

---

## Additional Resources

### Related Documentation

- **Implementation Plans**: `.claude/PRPs/features/` - Technical implementation PRPs
- **System Architecture**: `CLAUDE.md` - System architecture and specialist guides
- **Technical Docs**: `docs/` - Detailed technical documentation
- **Agent Specifications**: `.claude/specialists/` - Agent-specific documentation

### Tools

- **PRD Creation**: `/create-prd` command
- **PRP Creation**: `/prp:prp-core-create` command
- **Product Strategy**: `/pm` command (Product Manager agent)
- **Architecture Planning**: `/architect` command

---

## Changelog

**2024-12-18**:
- Initial product-development directory structure
- Created comprehensive E2I Platform PRD v1.0
- Created PRD template for future features
- Created this README

---

## Contact

For questions about product documentation:
- Review the template: `resources/PRD-template.md`
- Check existing PRD: `current-feature/PRD.md`
- Use Claude Code commands: `/create-prd`, `/pm`
- Consult with Product Management team

---

**Maintained by**: Product Management
**Last Updated**: 2024-12-18
