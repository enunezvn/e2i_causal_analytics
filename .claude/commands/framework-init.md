# Framework Initialization Workflow

Initialize framework features for your project, setting up the appropriate structure and recommending workflows based on project type and complexity.

## Purpose

Helps you:
- Assess project type and complexity
- Recommend suitable development workflows
- Activate appropriate agents
- Guide through planning documents
- Set up document sharding if needed

## Usage

```
/framework-init
```

The assistant will guide you through:
1. Understanding your project goals
2. Assessing project complexity
3. Recommending workflow approach
4. Suggesting which agents to use
5. Guiding through planning documents

## Instructions for Assistant

When this command is executed, conduct a comprehensive framework initialization:

### Step 1: Understand Project Goals

Ask the user about their project:

```
Let me help you get started with the Claude Code Development Framework!

**Project Overview:**
1. What are you building? (brief description)
2. Is this:
   - [ ] New project (greenfield)
   - [ ] Existing project (adding framework)
   - [ ] Feature addition to existing codebase
   - [ ] Refactoring/modernization

3. Timeline expectations?
   - [ ] Prototype/MVP (weeks)
   - [ ] Production feature (months)
   - [ ] Long-term product (6+ months)

4. Team size?
   - [ ] Solo developer
   - [ ] Small team (2-5)
   - [ ] Medium team (6-15)
   - [ ] Large team (16+)
```

### Step 2: Assess Complexity

Based on answers, determine project complexity level:

**Level 0: Quick Fix/Small Feature**
- Single feature or bug fix
- Clear requirements
- 1-2 weeks max
- **Workflow:** Direct implementation with `/code-review`

**Level 1: Feature with Tech Spec**
- Well-scoped feature
- Some unknowns to resolve
- 2-4 weeks
- **Workflow:** Tech spec → Implementation → Testing

**Level 2: Multi-Feature Project**
- Multiple related features
- Requires architecture planning
- 1-3 months
- **Workflow:** PRD → Architecture → Epics → Implementation

**Level 3: Complex Product/Platform**
- Large scope, multiple epics
- Significant architecture decisions
- 3-6+ months
- **Workflow:** Full planning system + all documentation

### Step 3: Recommend Approach

Based on complexity, recommend specific workflow:

**For Level 0 (Quick Fix):**
```
**Recommended Approach: Direct Implementation**

This is a straightforward task. Here's what you need:

1. **Understand Requirements**
   - Read existing code/docs
   - Clarify any ambiguities

2. **Implement**
   - Follow CLAUDE.md conventions
   - Write tests as you go
   - Run `/code-review` when done

3. **Validate**
   - Run test suite
   - Manual testing
   - Create PR with `/commit` or `/create-pr`

**Estimated Setup:** 5 minutes
**Framework Features:** Code conventions, code review agent
```

**For Level 1 (Tech Spec Workflow):**
```
**Recommended Approach: Tech Spec → Implementation**

This feature needs some planning. Here's the workflow:

1. **Planning Phase**
   - Use `/pm` to create lightweight tech spec
   - Define requirements and acceptance criteria
   - Identify technical approach

2. **Implementation Phase**
   - Follow CLAUDE.md conventions
   - Implement feature incrementally
   - Use `/code-review` for quality checks

3. **Validation Phase**
   - Run comprehensive tests
   - Performance validation
   - Create PR

**Estimated Setup:** 30 minutes
**Framework Features:** PM agent, code review, testing patterns
```

**For Level 2 (PRD + Architecture Workflow):**
```
**Recommended Approach: PRD → Architecture → Implementation**

This project needs formal planning. Recommended workflow:

1. **Requirements Phase**
   - Use `/pm` to create PRD
   - Define user stories and success metrics
   - Prioritize features using RICE framework

2. **Architecture Phase**
   - Use `/architect` to design system
   - Create architecture document
   - Review with `/security` for security considerations
   - Consider `/design-database-schema` if data-heavy

3. **Planning Phase**
   - Break down into epics and stories
   - Use `/shard-doc` for large PRDs if needed
   - Set up optional planning system if needed

4. **Implementation Phase**
   - Implement epic-by-epic
   - Regular `/code-review` checks
   - Follow PIV loop (Prime → Implement → Validate)

5. **Validation & Deployment**
   - Comprehensive testing
   - Use deployment conventions from docs/conventions/

**Estimated Setup:** 1-2 hours
**Framework Features:** All agents, planning docs, conventions
```

**For Level 3 (Complex Product):**
```
**Recommended Approach: Full Planning System + Comprehensive Workflow**

Complex project requiring structured approach:

1. **Setup Planning System**
   - Consider optional planning system from templates/
   - Automated documentation maintenance
   - Session state tracking for long development

2. **Discovery & Planning**
   - `/pm` for comprehensive PRD
   - Market research and competitive analysis
   - User personas and journey mapping
   - `/shard-doc` for large documents

3. **Architecture & Design**
   - `/architect` for system architecture
   - Multiple architecture reviews
   - `/security` threat modeling
   - Database schema design

4. **Epic Breakdown**
   - Create detailed epics and stories
   - Multi-phase implementation plan
   - Risk assessment and mitigation

5. **Implementation**
   - Phase-based development
   - Regular architecture reviews
   - Continuous integration with all agents
   - Documentation as you build

6. **Validation & Launch**
   - Staged rollout planning
   - Performance testing
   - Security audit
   - Deployment procedures

**Estimated Setup:** Half day
**Framework Features:** Everything + optional planning system
**Recommendation:** Review templates/planning-system-for-complex-projects/
```

### Step 4: Guide to Planning Documents

Offer to help create planning structure if Level 2-3:

```
Would you like help setting up planning documents?

I can help you:
1. Create initial PRD using `/create-prd`
2. Design architecture using `/create-architecture-documentation`
3. Set up document sharding for large files using `/shard-doc`
4. Create database schema using `/design-database-schema`

Which would you like to start with?
```

### Step 5: Provide Next Steps

Give user clear next actions:

**For Level 0-1:**
```
**Next Steps:**

1. Start with: `/pm` or directly implement
2. Follow conventions in CLAUDE.md
3. Use `/code-review` before committing
4. Create PR with `/create-pr` when ready

**Key Framework Resources:**
- CLAUDE.md - Coding conventions
- .claude/.agent_docs/ - Agent reference guides
- docs/conventions/ - Detailed guidelines

Ready to start? Let me know which agent you want to activate first!
```

**For Level 2-3:**
```
**Next Steps:**

**Week 1: Requirements & Architecture**
1. Activate `/pm` agent
2. Create PRD: `/create-prd`
3. Review and refine with stakeholders
4. Activate `/architect` agent
5. Create architecture: `/create-architecture-documentation`
6. Security review: `/security` then threat modeling

**Week 2-3: Planning & Setup**
1. Break into epics and stories
2. Use `/shard-doc` on large documents
3. Set up development environment
4. Create first epic implementation plan

**Week 4+: Implementation**
1. Follow PIV loop per epic
2. Regular `/code-review` checks
3. Maintain documentation
4. Track progress

**Key Framework Resources:**
- CLAUDE.md - Master conventions
- .claude/agents/ - Specialized agents
- .claude/.agent_docs/ - Agent references
- docs/conventions/ - Detailed guidelines
- templates/ - Optional planning system (Level 3)

Ready to start? Which agent should we activate first?
```

## Agent Selection Guide

Help user understand when to use each agent:

| Agent | When to Use | Primary Output |
|-------|-------------|----------------|
| `/pm` | Requirements gathering, feature prioritization | PRD, user stories, roadmaps |
| `/architect` | System design, technology decisions | Architecture docs, technical specs |
| `/security` | Security review, threat modeling | Security requirements, threat models |
| `/code-reviewer` | Implementation quality check | Code review feedback |
| `/documentation-expert` | Creating/updating docs | Technical documentation |

## Document Sharding Recommendations

**Suggest sharding when:**
- PRD >20k tokens (>5 epics)
- Architecture doc >15k tokens (>5 major components)
- Convention docs being loaded repeatedly
- Team mentions token/context issues

**How to shard:**
```
Use /shard-doc command:
1. Select large document
2. Choose destination folder
3. Tool splits by ## headings
4. Creates index.md + section files
5. Agents can now load selectively
```

## Success Criteria

Good initialization results in:
- ✅ Clear understanding of project scope
- ✅ Appropriate workflow selected
- ✅ Team aligned on approach
- ✅ Framework features activated appropriately
- ✅ Next steps are obvious

---

**Remember:** The framework adapts to your needs. Start simple, add complexity only when needed.

Type `/framework-init` to begin framework initialization process.
