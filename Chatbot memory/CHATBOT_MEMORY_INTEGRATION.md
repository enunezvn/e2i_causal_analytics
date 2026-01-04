# E2I Chatbot Memory Integration

## Summary: What's New vs What Exists

| Component | Already Exists | New for Chatbot |
|-----------|----------------|-----------------|
| Working Memory | ✅ Redis + LangGraph checkpointer | - |
| Episodic Memory | ✅ `episodic_memories` table | - |
| Semantic Memory | ✅ FalkorDB + Graphity | - |
| Procedural Memory | ✅ `procedural_memories` table | - |
| User Sessions | ✅ `user_sessions` table | + `preferences` JSONB column |
| Chat Threads | ❌ | ✅ `chat_threads` table |
| Chat Messages | ❌ | ✅ `chat_messages` table |
| User Preferences | ❌ | ✅ `user_preferences` table |

**Only 3 new tables + 1 column modification needed.**

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           E2I MEMORY ARCHITECTURE                                │
│                         (Existing + Chatbot Additions)                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SHORT-TERM MEMORY                                   │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      Redis + LangGraph Checkpointer                      │    │
│  │                           (ALREADY EXISTS)                               │    │
│  │                                                                          │    │
│  │  • Active conversation state (24h TTL)                                   │    │
│  │  • Scratchpad & evidence board                                          │    │
│  │  • thread_id → maps to chat_threads.thread_id                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Persists to
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               LONG-TERM MEMORY                                   │
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐        │
│  │  Episodic Memory   │  │  Semantic Memory   │  │ Procedural Memory  │        │
│  │  (ALREADY EXISTS)  │  │  (ALREADY EXISTS)  │  │  (ALREADY EXISTS)  │        │
│  │                    │  │                    │  │                    │        │
│  │  Supabase/pgvector │  │ FalkorDB/Graphity  │  │  Supabase/pgvector │        │
│  │                    │  │                    │  │                    │        │
│  │  episodic_memories │  │  :Patient, :HCP,   │  │ procedural_memories│        │
│  │  cognitive_cycles  │  │  :Brand, :CAUSES   │  │                    │        │
│  │  learning_signals  │  │                    │  │                    │        │
│  └────────┬───────────┘  └────────────────────┘  └────────────────────┘        │
│           │                                                                      │
│           │ Promotes significant interactions                                    │
│           │                                                                      │
└───────────┼──────────────────────────────────────────────────────────────────────┘
            │
            ▲ save_chat_to_episodic()
            │
┌───────────┴──────────────────────────────────────────────────────────────────────┐
│                           CHATBOT-SPECIFIC TABLES                                 │
│                                   (NEW)                                           │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                           chat_threads                                   │     │
│  │                                                                          │     │
│  │  • thread_id (UUID) ← Links to LangGraph checkpointer                   │     │
│  │  • user_id, session_id                                                  │     │
│  │  • initial_context (JSONB) - filter snapshot                            │     │
│  │  • agents_used (TEXT[]) - which agents participated                     │     │
│  │  • topic_embedding (vector) - for semantic search                       │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                      │                                            │
│                                      │ 1:N                                        │
│                                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                          chat_messages                                   │     │
│  │                                                                          │     │
│  │  • message_id (UUID)                                                    │     │
│  │  • role: user | assistant | system | tool                               │     │
│  │  • content (TEXT)                                                       │     │
│  │  • agent_ids (TEXT[]) - agents that contributed                         │     │
│  │  • validation_id, gate_decision - links to validation                   │     │
│  │  • filter_context (JSONB) - dashboard state when sent                   │     │
│  │  • content_embedding (vector) - for semantic search                     │     │
│  │  • feedback_rating, feedback_text                                       │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                        user_preferences                                  │     │
│  │                                                                          │     │
│  │  • user_id, key, value (JSONB)                                          │     │
│  │  • Keys: detail_level, default_brand, default_region, etc.              │     │
│  │  • Exposed via useCopilotReadable to agents                             │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                    user_sessions.preferences (MODIFIED)                  │     │
│  │                                                                          │     │
│  │  • Added JSONB column for session-level preferences                     │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
User sends message in chat
          │
          ▼
┌──────────────────────────────────┐
│   CopilotKit Frontend            │
│   useCopilotReadable():          │
│   - dashboard.filters            │
│   - user preferences (from DB)   │
│   - agent registry               │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   LangGraph Backend              │
│   Redis Checkpointer:            │
│   - Working memory (24h TTL)     │
│   - thread_id for continuity     │
└──────────────┬───────────────────┘
               │
               │  After response
               ▼
┌──────────────────────────────────┐
│   Persist to Supabase            │
│                                  │
│   1. chat_messages               │
│      - Full message content      │
│      - Agent attribution         │
│      - Embeddings for search     │
│                                  │
│   2. chat_threads (auto-update)  │
│      - agents_used[]             │
│      - message_count++           │
│      - last_message_at           │
│                                  │
│   3. episodic_memories           │
│      - If validation attached    │
│      - For cross-session learning│
└──────────────────────────────────┘
```

---

## Why Not mem0?

| mem0 Feature | E2I Already Has | Notes |
|--------------|-----------------|-------|
| Session memory | Redis checkpointer | 24h TTL, LangGraph native |
| User memory | `user_preferences` table | Simple key-value, agent-readable |
| Org memory | `episodic_memories` + FalkorDB | More sophisticated! |
| Vector search | pgvector | Already indexed |
| Graph relationships | FalkorDB + Graphity | Knowledge graph with auto-extraction |

**mem0 would add a 5th layer on top of 4 existing ones.** These 3 tables extend what you have without duplication.

---

## Usage in CopilotKit

### Load preferences on mount
```tsx
// In E2ICopilotProvider.tsx
useEffect(() => {
  supabase.rpc('get_user_preferences', { p_user_id: userId })
    .then(({ data }) => setUserPrefs(data));
}, [userId]);

useCopilotReadable({
  description: 'User preferences',
  value: userPrefs,  // { detail_level: 'analyst', default_brand: 'Remibrutinib' }
});
```

### Save preference via agent action
```tsx
useCopilotAction({
  name: 'rememberPreference',
  handler: async ({ key, value }) => {
    await supabase.rpc('upsert_user_preference', {
      p_user_id: userId,
      p_key: key,
      p_value: JSON.stringify(value),
    });
  },
});
```

### Search past conversations
```tsx
useCopilotAction({
  name: 'recallPastDiscussion',
  handler: async ({ topic }) => {
    const embedding = await embed(topic);
    const { data } = await supabase.rpc('search_chat_messages', {
      p_user_id: userId,
      p_query_embedding: embedding,
      p_limit: 5,
    });
    return data;
  },
});
```

---

## Migration Checklist

1. ✅ Run `008_chatbot_memory_tables.sql` in Supabase
2. ✅ Verify `user_sessions.preferences` column added
3. ✅ Test `search_chat_messages()` function
4. ✅ Wire up `E2ICopilotProvider` to load preferences
5. ✅ Add `save_chat_to_episodic()` call in message handler (optional)

---

## Table Count Update

| Category | Before | After |
|----------|--------|-------|
| Core Entities | 7 | 7 |
| Memory Tables | 5 | 8 (+3) |
| User Tables | 1 | 1 (modified) |
| **Total** | **20** | **23** |

The 3 new tables are **chatbot-specific** and don't affect the existing agent/analytics architecture.
