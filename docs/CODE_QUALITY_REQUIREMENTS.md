# Code Quality Requirements for Future Development

This document outlines critical bugs and code quality issues that must be avoided when implementing the messaging and integration components.

## Development Environment and Tools

### Required Tools Installation

Before starting development, install all required tools:

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (includes all quality tools)
pip install -r requirements-dev.txt
```

### Development Dependencies

The `requirements-dev.txt` includes all tools needed for code quality:

- **Testing**: pytest, pytest-cov, pytest-asyncio, pytest-timeout, coverage
- **Code Formatting**: black, isort
- **Linting**: flake8, pylint, mypy
- **Security Scanning**: bandit
- **Dependency Management**: pip-tools
- **Type Checking**: mypy with type stubs for common libraries
- **Development**: ipython, ipdb for debugging
- **Documentation**: sphinx, sphinx-rtd-theme

### Running Code Quality Checks

```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linters
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Security scanning
bandit -r src/ -ll

# Run all tests
pytest tests/ -v --cov=src
```

### Dependency Management

Use pip-tools to maintain hashed requirements:

```bash
# Regenerate hashed requirements after updating dependencies
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt

# Upgrade all dependencies
pip-compile --upgrade requirements.txt -o requirements-hashed.txt
```

**Security Note**: Hashed requirements help prevent supply chain attacks by verifying package integrity.

## Overview

The following components are planned for future implementation:
- `backpressure.py` - Backpressure management for message bus
- `sharded_message_bus.py` - Sharded message bus implementation
- `resilience.py` - Resilience patterns
- `redis_bridge.py` - Redis integration bridge
- `kafka_bridge.py` - Kafka integration bridge

## Critical Issues to Avoid

### 1. backpressure.py - Non-Existent Method Calls

**Issue**: Calls to methods that don't exist in ShardedMessageBus will cause runtime crashes.

**Required Methods**:
```python
# Must be implemented in ShardedMessageBus class:
async def pause_publishes(self, shard_id: str) -> None:
    """Pause publishing to specified shard."""
    pass

async def resume_publishes(self, shard_id: str) -> None:
    """Resume publishing to specified shard."""
    pass
```

**Location**: These methods should be called from backpressure manager when handling backpressure events.

### 2. sharded_message_bus.py - Encryption Type Mismatch

**Issue**: Inconsistent handling of encrypted payloads (bytes vs strings).

**Fix Required**:
- `encrypt_data()` returns `bytes`
- Do NOT call `.encode('utf-8')` on already-encrypted bytes
- Correct usage:
```python
# Encryption
processed_payload = self.security_utils.encrypt_data(
    json.dumps(payload),
    context=f"msgbus:{topic}"
)  # Returns bytes

# Decryption - do NOT encode bytes again
decrypted_json = self.encryption.decrypt(message.payload).decode('utf-8')
# NOT: message.payload.encode('utf-8')
```

### 3. sharded_message_bus.py - Duplicate Metric Increment

**Issue**: Error counter incremented twice for same error, inflating metrics.

**Fix Required**:
```python
# WRONG - increments twice:
# MESSAGE_BUS_CALLBACK_ERRORS.labels(shard_id="unknown").inc()
# MESSAGE_BUS_CALLBACK_ERRORS.labels(shard_id=str(shard_id), topic="unknown").inc()

# CORRECT - increment once with proper labels:
MESSAGE_BUS_CALLBACK_ERRORS.labels(shard_id=str(shard_id), topic="unknown").inc()
```

### 4. sharded_message_bus.py - Inefficient Session Creation

**Issue**: Creating new HTTP session for every request leads to resource exhaustion.

**Fix Required**:
```python
# In __init__
self._http_session: Optional[aiohttp.ClientSession] = None

# In publish or other methods
if not self._http_session:
    self._http_session = aiohttp.ClientSession()
await self._http_session.post(...)

# In shutdown
async def shutdown(self):
    if self._http_session:
        await self._http_session.close()
    # ... other cleanup
```

### 5. sharded_message_bus.py - Untracked Async Tasks

**Issue**: Fire-and-forget tasks with no error handling or tracking.

**Fix Required**:
```python
# Add error handler to tasks
def _task_error_handler(task: asyncio.Task) -> None:
    if not task.cancelled():
        exc = task.exception()
        if exc:
            logger.error(f"Background task failed: {exc}", exc_info=exc)

# When creating tasks
task = asyncio.create_task(_run_workflow())
task.add_done_callback(_task_error_handler)

# Better: Track tasks for proper cleanup
self._background_tasks: Set[asyncio.Task] = set()
task = asyncio.create_task(_run_workflow())
self._background_tasks.add(task)
task.add_done_callback(lambda t: self._background_tasks.discard(t))
```

### 6. sharded_message_bus.py - String Encryption Inconsistency

**Issue**: Mixed types in message payload (bytes for encrypted, str for non-encrypted).

**Fix Required**:
```python
# Option 1: Always use bytes
if self.encryption_enabled:
    processed_payload = self.security_utils.encrypt_data(
        json.dumps(payload), context=f"msgbus:{topic}"
    )  # bytes
else:
    processed_payload = json.dumps(payload).encode('utf-8')  # bytes

# Option 2: Always use strings (base64 encode encrypted)
if self.encryption_enabled:
    encrypted_bytes = self.security_utils.encrypt_data(...)
    processed_payload = base64.b64encode(encrypted_bytes).decode('ascii')
else:
    processed_payload = json.dumps(payload)
```

### 7. sharded_message_bus.py - Race Condition in Rebalancing

**Issue**: If exception occurs during rebalancing, event never gets set again, deadlocking publishes.

**Fix Required**:
```python
# Use try/finally to guarantee event is set
self.rebalancing_in_progress.clear()
try:
    # ... rebalancing work ...
    await self._rebalance_shards()
finally:
    self.rebalancing_in_progress.set()
```

### 8. sharded_message_bus.py - Security Utils Consistency

**Issue**: Mixed usage of `self.security_utils` and `self.encryption`.

**Fix Required**:
- Decide on ONE encryption interface
- Use consistently throughout the codebase
- Document which one to use and why

### 9. redis_bridge.py - Missing Imports

**Issue**: Using `uuid` and `random` modules without importing them.

**Fix Required**:
```python
import uuid
import random
```

### 10. redis_bridge.py - Variable Referenced Before Assignment

**Issue**: Using `internal_message` on right side before it's defined.

**Fix Required**:
```python
# WRONG:
internal_message = Message(
    topic=topic,
    payload=payload,
    trace_id=internal_message.get('trace_id', str(uuid.uuid4())),  # ERROR!
    timestamp=time.time()
)

# CORRECT:
internal_message = Message(
    topic=topic,
    payload=payload,
    trace_id=payload.get('trace_id', str(uuid.uuid4())),  # Use payload
    timestamp=time.time()
)
```

### 11. redis_bridge.py - Broken DLQ Logic

**Issue**: DLQ channel computed but never used in publish.

**Fix Required**:
```python
async def publish_dlq(self, message: Message, original_error: str) -> bool:
    """Publish message to Dead Letter Queue."""
    original_topic = message.topic
    dlq_channel = f"{original_topic}{self.cfg.DLQ_CHANNEL_SUFFIX}"
    
    # Create NEW message with DLQ topic
    dlq_message = Message(
        topic=dlq_channel,  # Use the DLQ topic, not original
        payload={**message.payload, "dlq_original_error": original_error},
        trace_id=message.trace_id,
        timestamp=time.time()
    )
    return await self.publish(dlq_message)
```

### 12. redis_bridge.py - Type Annotation Issues

**Issue**: Type hints fail when Redis is not installed.

**Fix Required**:
```python
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from redis.asyncio import Redis, PubSub

class RedisBridge:
    def __init__(self):
        self.redis_client: Optional["Redis"] = None
        self.pubsub_client: Optional["PubSub"] = None
```

### 13. redis_bridge.py - Inconsistent Error Handling

**Issue**: Generic exceptions don't record circuit breaker failure.

**Fix Required**:
```python
try:
    # ... setup code ...
    self.circuit.record_success()
except (ConnectionError, TimeoutError) as e:
    self.circuit.record_failure()
    raise
except Exception as e:
    self.circuit.record_failure()  # ADD THIS
    logger.error(f"Unexpected error: {e}")
    raise
```

### 14. resilience.py - Unused Import

**Issue**: Private function imported but never used.

**Fix Required**:
```python
# Remove if not used:
# from omnicore_engine.metrics import _get_or_create_metric

# Or make it part of public API if needed
```

## Testing Requirements

When implementing these components, ensure:

1. **Unit tests** for all error paths
2. **Integration tests** for message bus with encryption enabled/disabled
3. **Load tests** for session reuse and connection pooling
4. **Error injection tests** for circuit breaker and retry logic
5. **Type checking** with mypy to catch annotation issues

## Security Requirements

1. Never commit encryption keys or secrets
2. Always validate message payloads before deserialization
3. Implement rate limiting on message bus operations
4. Log security-relevant events (auth failures, encryption errors)
5. Use constant-time comparison for sensitive data

## Performance Requirements

1. Reuse HTTP sessions and connection pools
2. Implement proper backpressure mechanisms
3. Monitor and alert on queue depths
4. Use async operations throughout
5. Avoid blocking calls in async contexts

## Review Checklist

Before merging code for these components:

- [ ] All imports are present and correct
- [ ] No variable-before-assignment errors
- [ ] Consistent type handling (bytes vs strings)
- [ ] Proper error handling with try/finally
- [ ] No duplicate metric increments
- [ ] Session/connection reuse implemented
- [ ] Async tasks tracked and have error handlers
- [ ] Type annotations use TYPE_CHECKING where needed
- [ ] Circuit breaker failures recorded correctly
- [ ] DLQ logic uses correct topic/channel
- [ ] All tests pass
- [ ] Security scan passes
- [ ] Code review approved

## References

- Python asyncio best practices: https://docs.python.org/3/library/asyncio-task.html
- aiohttp client session reuse: https://docs.aiohttp.org/en/stable/client_reference.html
- Circuit breaker pattern: https://martinfowler.com/bliki/CircuitBreaker.html
- Backpressure handling: https://mechanical-sympathy.blogspot.com/2012/05/apply-back-pressure-when-overloaded.html
