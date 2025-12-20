---
allowed-tools: Read, Write, Edit, Bash
argument-hint: [schema-type] | --relational | --nosql | --hybrid | --normalize
description: Design optimized database schemas with proper relationships, constraints, and performance considerations
---

# Design Database Schema

Design optimized database schemas with comprehensive data modeling: **$ARGUMENTS**

---

## Input Validation

Before proceeding, validate and parse the schema type argument.

```bash
# Optional schema type with flags
if [ -z "$ARGUMENTS" ]; then
  echo "ℹ️  No schema type specified, will analyze and recommend optimal approach"
  SCHEMA_TYPE=""
elif [[ "$ARGUMENTS" == "--relational" ]]; then
  echo "✓ Schema Type: Relational (SQL)"
  SCHEMA_TYPE="relational"
elif [[ "$ARGUMENTS" == "--nosql" ]]; then
  echo "✓ Schema Type: NoSQL (document/key-value)"
  SCHEMA_TYPE="nosql"
elif [[ "$ARGUMENTS" == "--hybrid" ]]; then
  echo "✓ Schema Type: Hybrid (SQL + NoSQL)"
  SCHEMA_TYPE="hybrid"
elif [[ "$ARGUMENTS" == "--normalize" ]]; then
  echo "✓ Schema Type: Normalized relational schema"
  SCHEMA_TYPE="normalize"
else
  echo "❌ Error: Invalid schema type '$ARGUMENTS'"
  echo ""
  echo "Valid schema types:"
  echo "  --relational   Design SQL/relational database schema"
  echo "  --nosql        Design NoSQL database schema"
  echo "  --hybrid       Design hybrid SQL + NoSQL approach"
  echo "  --normalize    Design normalized relational schema"
  echo ""
  echo "Usage: /design-database-schema [schema-type]"
  echo "Example: /design-database-schema --relational"
  exit 1
fi
```

---

## Current Project Context

- Application type: Based on $ARGUMENTS or codebase analysis
- Data requirements: @requirements/ or project documentation
- Existing schema: @prisma/schema.prisma or @migrations/ or database dumps
- Performance needs: Expected scale, query patterns, and data volume

## Task

Design comprehensive database schema with optimal structure and performance:

**Schema Type**: Use $ARGUMENTS to specify relational, NoSQL, hybrid approach, or normalization level

**Design Framework**:
1. **Requirements Analysis** - Business entities, relationships, data flow, and access patterns
2. **Entity Modeling** - Tables/collections, attributes, primary/foreign keys, constraints
3. **Relationship Design** - One-to-one, one-to-many, many-to-many associations
4. **Normalization Strategy** - Data consistency vs performance trade-offs
5. **Performance Optimization** - Indexing strategy, query optimization, partitioning
6. **Security Design** - Access control, data encryption, audit trails

**Advanced Patterns**: Implement temporal data, soft deletes, JSONB fields, full-text search, audit logging, and scalability patterns.

**Validation**: Ensure referential integrity, data consistency, query performance, and future extensibility.

**Output**: Complete schema design with DDL scripts, ER diagrams, performance analysis, and migration strategy.