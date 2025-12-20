---
name: block-hardcoded-secrets
enabled: true
event: file
pattern: "(api[_-]?key|password|secret|token)\\s*[=:]\\s*['\"][^'\"]+['\"]"
glob: "*.{ts,js,py,go,java}"
action: block
---

**Hardcoded secret detected** - Never commit credentials.

Use environment variables:
```typescript
const apiKey = process.env.API_KEY;
```

See: Security section in CLAUDE.md
