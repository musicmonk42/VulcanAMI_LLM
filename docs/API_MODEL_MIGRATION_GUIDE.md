# API Model Migration Guide

**Version:** 2.0.0  
**Date:** January 2026  
**Status:** Active Migration

---

## Overview

This document describes the API model consolidation that affects chat endpoints. The changes ensure a consistent API surface while maintaining backward compatibility.

---

## What Changed?

### 1. ChatRequest → UnifiedChatRequest

**Deprecated:** `ChatRequest` with `prompt` field  
**Recommended:** `UnifiedChatRequest` with `message` field

The legacy `ChatRequest` model is now deprecated in favor of `UnifiedChatRequest`, which provides:
- Consistent field naming (`message` instead of `prompt`)
- Full feature control (reasoning, memory, safety, planning, causal)
- Conversation history support
- Conversation ID tracking

### 2. Increased max_tokens Limit

**Old:** `max_tokens` limited to 8,000  
**New:** `max_tokens` limited to 32,000

This allows for longer responses when needed.

---

## Migration Path

### Option 1: Update to UnifiedChatRequest (Recommended)

**Before:**
```python
# Old way (deprecated)
request = {
    "prompt": "Explain quantum entanglement",
    "max_tokens": 2000
}
```

**After:**
```python
# New way (recommended)
request = {
    "message": "Explain quantum entanglement",
    "max_tokens": 2000,
    "history": [],
    "conversation_id": None,
    "enable_reasoning": True,
    "enable_memory": True,
    "enable_safety": True,
    "enable_planning": True,
    "enable_causal": True
}
```

### Option 2: Continue Using ChatRequest (Temporary)

Legacy `ChatRequest` still works but will show deprecation warnings:

```python
# Still works but shows DeprecationWarning
request = {
    "prompt": "Hello",
    "max_tokens": 1000
}
```

The warning message:
```
DeprecationWarning: ChatRequest is deprecated. Use UnifiedChatRequest instead. Change 'prompt' to 'message' in your request.
```

---

## UnifiedChatRequest API Reference

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | The user's message/prompt |
| `max_tokens` | integer | No | 2000 | Maximum tokens in response (1-32000) |
| `history` | array | No | `[]` | Conversation history |
| `conversation_id` | string | No | `null` | Optional conversation ID for tracking |
| `enable_reasoning` | boolean | No | `true` | Enable advanced reasoning engines |
| `enable_memory` | boolean | No | `true` | Enable long-term memory search |
| `enable_safety` | boolean | No | `true` | Enable safety validation |
| `enable_planning` | boolean | No | `true` | Enable hierarchical planning |
| `enable_causal` | boolean | No | `true` | Enable causal reasoning |

### Example Request

```json
{
  "message": "Explain quantum entanglement",
  "max_tokens": 2000,
  "history": [
    {
      "role": "user",
      "content": "Hi"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    }
  ],
  "conversation_id": "conv_12345",
  "enable_reasoning": true,
  "enable_memory": true,
  "enable_safety": true,
  "enable_planning": true,
  "enable_causal": true
}
```

### History Message Format

Each message in `history` must have:

```json
{
  "role": "user" | "assistant" | "system",
  "content": "message content"
}
```

**Validation:**
- `role` must be one of: `user`, `assistant`, `system`
- `content` must be non-empty (1-50000 characters)

---

## Timeline

| Date | Event |
|------|-------|
| **January 2026** | `ChatRequest` deprecated, `UnifiedChatRequest` is standard |
| **Q2 2026** | Deprecation warnings added to logs |
| **Q4 2026** | `ChatRequest` planned for removal |

---

## Backward Compatibility

### Automatic Conversion

When you send a `ChatRequest`, the system automatically converts it to `UnifiedChatRequest`:

```python
# Your request
ChatRequest(prompt="Hello", max_tokens=1000)

# Internally converted to
UnifiedChatRequest(
    message="Hello",
    max_tokens=1000,
    history=[],
    enable_reasoning=True,
    enable_memory=True,
    enable_safety=True,
    enable_planning=True,
    enable_causal=True
)
```

### Feature Toggle Defaults

When converting from `ChatRequest`, all feature toggles default to `True`, maintaining full functionality.

---

## Client Library Updates

### Python SDK

```python
from vulcan.api.models import UnifiedChatRequest, ChatHistoryMessage

# Create request
request = UnifiedChatRequest(
    message="Explain quantum entanglement",
    max_tokens=2000,
    history=[
        ChatHistoryMessage(role="user", content="Hi"),
        ChatHistoryMessage(role="assistant", content="Hello!")
    ]
)
```

### REST API

```bash
curl -X POST https://api.example.com/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "Explain quantum entanglement",
    "max_tokens": 2000,
    "history": [],
    "enable_reasoning": true
  }'
```

---

## FAQ

### Q: When will ChatRequest be removed?

**A:** Planned for Q4 2026, but we'll provide at least 6 months notice.

### Q: Do I need to update immediately?

**A:** No, but we recommend updating to avoid deprecation warnings and to access new features.

### Q: What if I don't use the new features?

**A:** You can still use the simplified form:

```json
{
  "message": "Your question here"
}
```

All optional fields use sensible defaults.

### Q: Can I disable certain features?

**A:** Yes, set any `enable_*` field to `false`:

```json
{
  "message": "Simple query",
  "enable_reasoning": false,
  "enable_memory": false
}
```

This can improve response time for simple queries.

---

## Support

For questions or issues:
- **Documentation:** https://docs.vulcan-agi.dev
- **GitHub Issues:** https://github.com/musicmonk42/VulcanAMI_LLM/issues
- **Email:** support@vulcan-agi.dev

---

**Copyright © 2026 VULCAN-AGI Team. All rights reserved.**
