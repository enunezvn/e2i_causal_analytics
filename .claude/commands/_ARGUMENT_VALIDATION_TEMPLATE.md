# Argument Validation Template

This template shows how to add argument validation to commands that use `$ARGUMENTS`.

## Basic Validation Pattern

Add this section at the beginning of your command, right after the frontmatter:

```markdown
## Input Validation

Before proceeding, validate the input arguments:

**Required Arguments**: <description-of-what's-required>

**Validation Steps**:
1. Check if `$ARGUMENTS` is provided
2. Validate format/content
3. Provide helpful error message if invalid

**Example Validation**:
\```bash
# Check if argument is provided
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: <argument-name> is required"
  echo ""
  echo "Usage: /command-name <argument-description>"
  echo "Example: /command-name \"example value\""
  exit 1
fi

# Store argument for clarity
ARG_VALUE="$ARGUMENTS"
echo "✓ Using: $ARG_VALUE"
\```
```

---

## Validation Patterns by Argument Type

### Pattern 1: Required Text Argument

**Use for**: Feature descriptions, commit messages, text input

```bash
# Validate required text argument
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: Description is required"
  echo ""
  echo "Usage: /command <description>"
  echo "Example: /command \"Add user authentication\""
  exit 1
fi

# Check minimum length (optional)
if [ ${#ARGUMENTS} -lt 5 ]; then
  echo "❌ Error: Description too short (minimum 5 characters)"
  exit 1
fi

DESCRIPTION="$ARGUMENTS"
echo "✓ Description: $DESCRIPTION"
```

### Pattern 2: Required Number (Issue/PR Number)

**Use for**: GitHub issue numbers, PR numbers, version numbers

```bash
# Validate required number
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: Issue number is required"
  echo ""
  echo "Usage: /command <issue-number>"
  echo "Example: /command 123"
  exit 1
fi

# Check if it's a valid number
if ! [[ "$ARGUMENTS" =~ ^[0-9]+$ ]]; then
  echo "❌ Error: Invalid issue number. Must be a positive integer."
  echo "Received: $ARGUMENTS"
  exit 1
fi

ISSUE_NUMBER="$ARGUMENTS"
echo "✓ Using issue #$ISSUE_NUMBER"
```

### Pattern 3: Required Version Number

**Use for**: Release versions, package versions

```bash
# Validate required version
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: Version is required"
  echo ""
  echo "Usage: /command <version>"
  echo "Example: /command 1.2.0"
  echo "Example: /command v1.2.0"
  exit 1
fi

# Check version format (basic semver)
if ! [[ "$ARGUMENTS" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
  echo "❌ Error: Invalid version format"
  echo "Expected: X.Y.Z or vX.Y.Z (optionally with -suffix)"
  echo "Examples: 1.2.0, v1.2.0, 1.2.0-alpha.1"
  echo "Received: $ARGUMENTS"
  exit 1
fi

VERSION="$ARGUMENTS"
echo "✓ Version: $VERSION"
```

### Pattern 4: Required File Path

**Use for**: Plan files, configuration files

```bash
# Validate required file path
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: File path is required"
  echo ""
  echo "Usage: /command <path-to-file>"
  echo "Example: /command .claude/PRPs/features/active/plan.md"
  exit 1
fi

FILE_PATH="$ARGUMENTS"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
  echo "❌ Error: File not found: $FILE_PATH"
  echo ""
  echo "Available plans:"
  ls -1 .claude/PRPs/features/active/*.md 2>/dev/null || echo "  (none found)"
  exit 1
fi

echo "✓ Using file: $FILE_PATH"
```

### Pattern 5: Optional Argument with Default

**Use for**: Commands that can run with or without arguments

```bash
# Optional argument with default value
if [ -z "$ARGUMENTS" ]; then
  DEFAULT_VALUE="default-value"
  echo "ℹ️  No argument provided, using default: $DEFAULT_VALUE"
  ARG_VALUE="$DEFAULT_VALUE"
else
  ARG_VALUE="$ARGUMENTS"
  echo "✓ Using: $ARG_VALUE"
fi
```

### Pattern 6: Multiple Arguments (Space-Separated)

**Use for**: Commands that accept multiple parameters

```bash
# Multiple arguments
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: Arguments required"
  echo ""
  echo "Usage: /command <arg1> <arg2> [arg3]"
  echo "Example: /command feature-branch plan.md"
  exit 1
fi

# Parse multiple arguments
read -r ARG1 ARG2 ARG3 <<< "$ARGUMENTS"

# Validate required arguments
if [ -z "$ARG1" ] || [ -z "$ARG2" ]; then
  echo "❌ Error: Both arg1 and arg2 are required"
  echo "Received: arg1='$ARG1' arg2='$ARG2'"
  exit 1
fi

echo "✓ ARG1: $ARG1"
echo "✓ ARG2: $ARG2"
[ -n "$ARG3" ] && echo "✓ ARG3: $ARG3"
```

### Pattern 7: Flag Detection

**Use for**: Commands with optional flags (--flag)

```bash
# Check for flags
SKIP_FLAG=false

if [[ "$ARGUMENTS" == *"--skip-e2e"* ]]; then
  SKIP_FLAG=true
  # Remove flag from arguments to get the actual value
  CLEAN_ARGS=$(echo "$ARGUMENTS" | sed 's/--skip-e2e//g' | xargs)
else
  CLEAN_ARGS="$ARGUMENTS"
fi

[ "$SKIP_FLAG" = true ] && echo "ℹ️  Skip flag detected"
```

### Pattern 8: GitHub URL or Number

**Use for**: Commands accepting GitHub issue/PR URLs or numbers

```bash
# Accept GitHub URL or number
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: Issue number or URL is required"
  echo ""
  echo "Usage: /command <issue-number-or-url>"
  echo "Example: /command 123"
  echo "Example: /command https://github.com/owner/repo/issues/123"
  exit 1
fi

# Extract number from URL if needed
if [[ "$ARGUMENTS" =~ github\.com/.*/(issues|pull)/([0-9]+) ]]; then
  ISSUE_NUMBER="${BASH_REMATCH[2]}"
  echo "✓ Extracted issue #$ISSUE_NUMBER from URL"
elif [[ "$ARGUMENTS" =~ ^[0-9]+$ ]]; then
  ISSUE_NUMBER="$ARGUMENTS"
  echo "✓ Using issue #$ISSUE_NUMBER"
else
  echo "❌ Error: Invalid issue number or URL"
  echo "Expected: number (123) or GitHub URL"
  echo "Received: $ARGUMENTS"
  exit 1
fi
```

---

## Complete Example

Here's a complete command with validation:

```markdown
---
description: Example command with validation
argument-hint: <required-arg> [optional-flag]
---

# Example Command

## Input Validation

Before proceeding, validate the arguments.

\```bash
# Check if argument is provided
if [ -z "$ARGUMENTS" ]; then
  echo "❌ Error: Feature description is required"
  echo ""
  echo "Usage: /example <description> [--skip-tests]"
  echo "Example: /example \"Add new feature\""
  echo "Example: /example \"Add new feature\" --skip-tests"
  exit 1
fi

# Check for optional flag
SKIP_TESTS=false
if [[ "$ARGUMENTS" == *"--skip-tests"* ]]; then
  SKIP_TESTS=true
  DESCRIPTION=$(echo "$ARGUMENTS" | sed 's/--skip-tests//g' | xargs)
else
  DESCRIPTION="$ARGUMENTS"
fi

# Validate description length
if [ ${#DESCRIPTION} -lt 5 ]; then
  echo "❌ Error: Description too short (minimum 5 characters)"
  exit 1
fi

echo "✓ Description: $DESCRIPTION"
[ "$SKIP_TESTS" = true ] && echo "ℹ️  Skipping tests"
\```

## Command Implementation

[Your command logic here]
```

---

## Best Practices

1. **Always validate required arguments** - Fail fast with clear error messages
2. **Provide usage examples** - Show users how to use the command correctly
3. **Use emoji indicators** - ❌ for errors, ✓ for success, ℹ️ for info
4. **Store validated values** - Use descriptive variable names
5. **Check file existence** - Validate paths before using them
6. **Validate formats** - Check version numbers, URLs, etc.
7. **Provide helpful alternatives** - List available options when validation fails
8. **Exit with status 1** - On validation failures for proper error handling

---

## Testing Your Validation

After adding validation, test these scenarios:

```bash
# Test without arguments (should fail)
/your-command

# Test with invalid format (should fail)
/your-command "x"

# Test with valid arguments (should succeed)
/your-command "valid argument"

# Test with flags
/your-command "argument" --flag
```

---

## Migration Checklist

When adding validation to existing commands:

- [ ] Identify if arguments are required or optional
- [ ] Choose appropriate validation pattern
- [ ] Add validation section after frontmatter
- [ ] Update `argument-hint` in frontmatter if needed
- [ ] Test all validation paths
- [ ] Update command documentation
- [ ] Add example usage

---

**Last Updated**: 2024-12-15
**Framework Version**: 1.0.0
