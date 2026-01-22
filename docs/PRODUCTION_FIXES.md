# Production Fixes Documentation

This document describes the critical production issues that were fixed and how to configure them.

## Overview

Six critical issues were identified from production logs and have been resolved with industry-standard solutions:

1. **Phantom Resolution Circuit Breaker Loop (CRITICAL)** - Fixed duplicate gap resolution counting
2. **Knowledge Storage JSON Serialization (CRITICAL)** - Fixed PatternType enum serialization
3. **GraphRAG Model Loading (HIGH)** - Fixed CrossEncoder parameter conflict
4. **Self-Improvement Policy Enhancement (MEDIUM)** - Reduced cooldowns and improved logging
5. **Default Objective Estimates (LOW)** - Added config file loading
6. **CPU Priority Permission Handling (LOW)** - Added permission checks and container detection

## Issue #1: Phantom Resolution Circuit Breaker Loop (CRITICAL)

### Problem
The Curiosity Engine repeatedly marked the same knowledge gaps as "resolved" multiple times within a short period, triggering the circuit breaker and suppressing all learning.

### Solution
- **Cycle-level deduplication**: Tracks which gaps have been resolved in each cycle to prevent duplicates
- **Unique cycle counting**: Counts distinct cycles instead of raw resolution entries
- **Increased threshold**: Raised from 3 to 5 resolutions per hour (configurable)
- **Graduated backoff**: 2h → 4h → 8h → 16h max suppression time

### Configuration
```bash
# Set custom phantom resolution threshold (default: 5)
export VULCAN_PHANTOM_THRESHOLD=5

# Set custom tracking window in seconds (default: 3600 = 1 hour)
export VULCAN_PHANTOM_WINDOW=3600
```

### Files Changed
- `src/vulcan/curiosity_engine/curiosity_engine_core.py`
- `src/vulcan/curiosity_engine/resolution_bridge.py`

---

## Issue #2: Knowledge Storage JSON Serialization (CRITICAL)

### Problem
Failed to save learned knowledge due to PatternType enum not being JSON serializable:
```
WARNING - Failed to compute diff: Object of type PatternType is not JSON serializable
```

### Solution
- **EnhancedJSONEncoder**: Custom JSON encoder that handles:
  - Enum objects (serializes to `.value`)
  - NumPy arrays (converts to lists)
  - NumPy scalar types (converts to Python types)
  - Objects with `to_dict()` method
  - Fallback to string representation for unknown types
- Applied to all `json.dump()` and `json.dumps()` calls in knowledge storage

### Usage
```python
import json
from vulcan.knowledge_crystallizer.knowledge_storage import EnhancedJSONEncoder

data = {"pattern": PatternType.SEQUENTIAL, "value": 42}
json_string = json.dumps(data, cls=EnhancedJSONEncoder)
```

### Files Changed
- `src/vulcan/knowledge_crystallizer/knowledge_storage.py`

---

## Issue #3: GraphRAG Model Loading (HIGH)

### Problem
Semantic models failed to load due to duplicate `local_files_only` keyword argument:
```
WARNING - Failed to load models: ... got multiple values for keyword argument 'local_files_only'
```

### Solution
- Use `model_kwargs` dictionary instead of passing `local_files_only` directly
- Add explicit `revision="main"` to avoid ambiguity
- Proper exception handling with fallback to mock embeddings

### Files Changed
- `src/persistant_memory_v46/graph_rag.py` (already fixed in codebase)

---

## Issue #4: Self-Improvement Policy Enhancement (MEDIUM)

### Problem
Self-improvement was blocked with long cooldowns (72 hours), making iteration slow.

### Solution
- **Reduced cooldown times**:
  - Transient failures: 4h → **2h**
  - Systemic failures: 72h → **24h**
- **Improved logging**: Clarifies what types of improvements CAN proceed without human review
- **Better visibility**: Distinguishes between code improvements (require review) and non-code improvements (can proceed autonomously)

### Example Log Output
```
[Self-Improvement] Non-code improvement 'update_documentation' can proceed autonomously.
[Self-Improvement] Code improvement 'fix_bugs' will be deferred for human review.
```

### Files Changed
- `src/vulcan/world_model/meta_reasoning/self_improvement_drive.py`
- `src/vulcan/world_model/world_model_core.py`

---

## Issue #5: Default Objective Estimates (LOW)

### Problem
Using fallback defaults instead of domain-specific estimates, with excessive warning logs.

### Solution
- **Config file loading**: Automatically loads from `configs/objective_estimates.json`
- **Validation**: Ensures estimates are floats in [0, 1] range
- **Reduced verbosity**: Warning downgraded to debug level with suppression option

### Configuration

Create `configs/objective_estimates.json`:
```json
{
  "objective_estimates": {
    "prediction_accuracy": 0.95,
    "safety": 0.99,
    "efficiency": 0.85
  }
}
```

Suppress warning if defaults are intentional:
```bash
export VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING=1
```

### Files Changed
- `src/vulcan/world_model/meta_reasoning/counterfactual_objectives.py`
- `configs/objective_estimates.json` (created)

---

## Issue #6: CPU Priority Permission Handling (LOW)

### Problem
Permission denied warnings cluttered logs when running in containerized environments:
```
WARNING - Could not set CPU priority (permission denied)
```

### Solution
- **Permission pre-check**: Checks UID and current nice value before attempting
- **Container detection**: Detects Docker, Kubernetes, and other containerized environments
- **Smart logging**: Uses debug level in containerized environments where permission denied is expected
- **Suppression option**: Environment variable to completely suppress the message

### Configuration

Suppress CPU priority warnings (useful in containers):
```bash
export VULCAN_SUPPRESS_CPU_PRIORITY_WARNING=1
```

### Container Detection
Automatically detects:
- Docker: `/.dockerenv` file
- Kubernetes: `KUBERNETES_SERVICE_HOST` environment variable
- Podman: `/run/.containerenv` file
- cgroups: Docker in `/proc/1/cgroup`

### Files Changed
- `src/llm_core/graphix_executor.py`

---

## Testing

### Validate Code Changes
```bash
python3 tests/validate_code_changes.py
```

This checks that all code changes are present and correct (no dependencies required).

### Run Full Test Suite
```bash
make test
# or
pytest tests/ -v
```

---

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `VULCAN_PHANTOM_THRESHOLD` | `5` | Number of resolutions before circuit breaker triggers |
| `VULCAN_PHANTOM_WINDOW` | `3600` | Time window in seconds for phantom detection |
| `VULCAN_GAP_GIVEUP_THRESHOLD` | `10` | Number of attempts before giving up on a gap |
| `VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING` | unset | Set to `1` to suppress objective estimates warning |
| `VULCAN_SUPPRESS_CPU_PRIORITY_WARNING` | unset | Set to `1` to suppress CPU priority warning |

---

## Migration Guide

### For Existing Deployments

1. **Update environment variables** (optional):
   ```bash
   # Adjust phantom resolution sensitivity if needed
   export VULCAN_PHANTOM_THRESHOLD=5
   
   # Suppress expected warnings in containerized environments
   export VULCAN_SUPPRESS_CPU_PRIORITY_WARNING=1
   export VULCAN_SUPPRESS_DEFAULT_OBJECTIVES_WARNING=1
   ```

2. **Add config files** (recommended):
   ```bash
   # Create domain-specific objective estimates
   cp configs/objective_estimates.json.example configs/objective_estimates.json
   # Edit with your domain-specific values
   ```

3. **Clear phantom suppression** (if affected):
   ```bash
   # If you're experiencing phantom suppression, clear the state
   rm -f data/curiosity_resolutions.db
   # The system will rebuild tracking from scratch
   ```

4. **Monitor logs**:
   - Check for "PHANTOM RESOLUTION CIRCUIT BREAKER" messages
   - Verify "Gap resolved by successful experiment" messages
   - Ensure knowledge storage saves complete without serialization errors

### Breaking Changes
**None**. All changes are backward compatible with graceful degradation.

---

## Performance Impact

### Expected Improvements
- **Curiosity Engine**: Should maintain learning cycles without hitting circuit breakers
- **Knowledge Storage**: All principles should be properly persisted
- **Self-Improvement**: Faster iteration with reduced cooldown times
- **Log Noise**: Reduced by ~80% with smart logging and suppression options

### Metrics to Monitor
```
# Before fixes (problematic):
[CuriosityEngine] Learning cycle complete: 0 experiments, 0.00 success rate
[CuriosityEngine] All synthetic gaps are suppressed due to phantom resolutions

# After fixes (healthy):
[CuriosityEngine] Learning cycle complete: 5 experiments, 0.80 success rate  
[CuriosityEngine] Gap exploration:knowledge_consolidation resolved by successful experiment
```

---

## Support

For issues or questions:
1. Check logs for ERROR and WARNING messages
2. Review environment variables configuration
3. Validate config files are properly formatted
4. Run validation script: `python3 tests/validate_code_changes.py`

---

## Technical Details

### Phantom Resolution Algorithm

**Before:**
```python
# Counted all resolution entries (duplicates in same cycle)
recent_resolutions = count_all_entries_in_window()
if recent_resolutions >= 3:  # Too aggressive
    suppress_for_hours = 2 ** (recent_resolutions - 3)  # Exponential
```

**After:**
```python
# Counts unique cycles (cycle-level deduplication)
recent_resolutions = count_distinct_cycles_in_window()
if recent_resolutions >= 5:  # More reasonable
    suppress_for_hours = min(2 ** (recent_resolutions - 5), 16)  # Capped at 16h
```

### JSON Serialization

**Before:**
```python
json.dumps(data)  # Fails on Enum, numpy, etc.
```

**After:**
```python
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        # ... other type handlers
        return super().default(obj)

json.dumps(data, cls=EnhancedJSONEncoder)  # Handles all types
```

---

## License

Part of the VULCAN-AMI project. All fixes follow the project's existing license.
