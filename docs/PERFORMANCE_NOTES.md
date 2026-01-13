# VULCAN Performance Notes and Optimizations

This document describes performance optimizations and known informational warnings in the VULCAN system.

## Performance Optimizations Implemented

### 1. Knowledge Crystallization Confidence (Issue #1) ✅ FIXED
**Problem**: Crystallization confidence was 0.00, causing all principles to be rejected despite being extracted.

**Root Cause**: 
- `CrystallizedPrinciple` class missing `execution_logic` field
- Validator warned if missing, reducing confidence by 0.1 per warning
- Missing `applicable_domains` added another -0.1

**Fix Applied**:
- Added `execution_logic` field with proper type hint
- Ensured `applicable_domains` is always populated
- Added `Principle` alias for backward compatibility

**Impact**: Crystallization confidence now 0.9-1.0 instead of 0.0-0.8

### 2. Philosophical Reasoning Performance (Issue #10) ✅ OPTIMIZED
**Problem**: PhilosophicalToolWrapper took 28.4 seconds for analysis.

**Root Cause**: WorldModel was being instantiated on each query (10-15s initialization overhead).

**Fix Applied**: 
- Changed to use singleton `get_world_model()` from singletons.py
- Prevents repeated initialization overhead

**Expected Impact**: First query pays initialization cost, subsequent queries are fast (~0.5-2s)

### 3. Keyword-Based Routing Optimization ✅ OPTIMIZED
**Problem**: Multiple `any()` calls with keyword lists was inefficient.

**Fix Applied**: 
- Replaced multiple `any()` calls with compiled regex patterns
- Regex patterns are cached by Python, making subsequent matches faster

**Impact**: Faster query routing for obvious query types

### 4. Code Quality Improvements ✅ IMPLEMENTED
- Made transparency explanation confidence configurable
- Improved cache_used tracking accuracy
- Better type hints for execution_logic field

## Informational Warnings (Non-Critical)

### Issue #6: Graphix LLM Backend Using Mock
**Status**: Informational - working as designed

**Message**: 
```
✗ Graphix LLM backend not available (using mock)
```

**Explanation**:
- Real Graphix LLM is optional proprietary component
- System falls back to MockGraphixVulcanLLM when not installed
- OpenAI backend is still available and functional
- Most reasoning works fine with mock backend

**Action Required**: None, unless you need Graphix-specific features
- To install: Contact Graphix vendor for installation
- System continues to function with OpenAI or other backends

### Issue #7: CPU Priority Setting Denied
**Status**: Informational - permission limitation

**Message**:
```
Could not set CPU priority (permission denied)
```

**Explanation**:
- System attempts to optimize CPU priority for faster inference
- Requires elevated privileges (sudo/administrator)
- Falls back gracefully to default priority
- Performance impact is minimal (~5-10% on high load)

**Action Required**: None, unless you need maximum performance
- To enable: Run with elevated privileges
- Or: Set OS-level nice values for the process
- System continues to function normally without it

### Issue #8: HierarchicalMemory Not Available
**Status**: Informational - fallback active

**Locations**:
- `self_improvement_drive.py`: Uses local cache fallback
- `telemetry_recorder.py`: Uses JSON-based storage
- `continual_learning.py`: Operates without persistent memory
- `memory_bridge.py`: Bridge not established

**Explanation**:
- HierarchicalMemory is optional advanced memory system
- Requires numpy and additional dependencies
- System falls back to simpler memory implementations
- Core functionality continues to work

**Action Required**: None, unless you need advanced memory features
- To enable: Install numpy and memory dependencies
- Benefits: Better long-term knowledge retention
- Drawback: Higher memory usage

### Issue #9: Gateway Hybrid Mode Not Enabled
**Status**: Informational - configuration choice

**Message**:
```
API Gateway running in single mode, not hybrid
```

**Explanation**:
- Gateway defaults to single mode for simplicity
- Hybrid mode combines multiple backends (OpenAI + Graphix)
- Single mode is faster and simpler for most use cases
- Only needed for backend redundancy or A/B testing

**Action Required**: None, unless you need hybrid backend support
- To enable: Set `GATEWAY_MODE=hybrid` environment variable
- Benefits: Backend redundancy, load balancing
- Drawback: Slightly higher latency

## Performance Monitoring

### Key Metrics to Watch

1. **Knowledge Crystallization**
   - Target: Confidence > 0.6 for extracted principles
   - Monitor: `crystallization_history` in logs
   - Alert if: Confidence consistently < 0.6

2. **Query Routing Time**
   - Target: < 100ms for tool selection
   - Monitor: `[ToolSelector]` log entries
   - Alert if: > 500ms consistently

3. **Philosophical Reasoning**
   - Target: < 5s per query (after warmup)
   - Monitor: `[PhilosophicalToolWrapper]` execution time
   - Alert if: > 10s consistently

4. **Memory Usage**
   - Target: Stable memory footprint
   - Monitor: System memory metrics
   - Alert if: Memory grows unbounded

### Optimization Checklist

- [x] Use singleton pattern for expensive components (WorldModel, UnifiedReasoner)
- [x] Cache embeddings in tool selection
- [x] Use regex patterns for keyword matching
- [x] Lazy-load heavy dependencies
- [ ] Consider LRU caches for repeated queries
- [ ] Profile hot paths for further optimization
- [ ] Monitor and optimize database queries

## Troubleshooting

### Slow Performance After Extended Runtime

**Symptoms**: System gets progressively slower over time

**Possible Causes**:
1. Embedding cache not being cleaned (handled by our fix)
2. Memory leaks in component initialization (handled by singletons)
3. Log file growth (rotate logs regularly)

**Solutions**:
- Restart service periodically (daily)
- Monitor memory usage trends
- Enable garbage collection in long-running processes

### High Memory Usage

**Symptoms**: Memory usage > 4GB

**Possible Causes**:
1. Multiple model instances loaded (use singletons)
2. Large result caches not being cleaned
3. Embedding cache too large

**Solutions**:
- Check singleton usage for all heavy components
- Configure cache TTLs appropriately
- Enable periodic cache cleanup

### Warm Pool Factory Failures

**Symptoms**: All tools fall back to singleton mode

**Possible Causes**:
1. Tools require constructor arguments not provided
2. Missing dependencies for tool initialization
3. Configuration issues

**Solutions**:
- Check tool constructor signatures
- Verify all dependencies are installed
- Review factory creation error messages (now includes exception details)

## Future Optimizations

### Potential Improvements

1. **Query Result Caching**
   - Cache reasoning results for repeated queries
   - Use query hash as cache key
   - Clear cache on model updates

2. **Batch Processing**
   - Process multiple queries in batch
   - Amortize model loading overhead
   - Implement request queuing

3. **Model Quantization**
   - Use quantized models for embeddings
   - Reduce memory footprint by 50-75%
   - Trade-off: slight accuracy decrease

4. **Distributed Processing**
   - Distribute reasoning across multiple workers
   - Use message queue for coordination
   - Scale horizontally under load

5. **Progressive Loading**
   - Load models on-demand
   - Unload unused models after timeout
   - Dynamic resource management

## Configuration Recommendations

### Development Environment
```yaml
# Fast startup, lower memory
embedding_cache_size: 1000
warm_pool_size: 1
enable_philosophical_reasoning: false
log_level: INFO
```

### Production Environment
```yaml
# Optimized for throughput
embedding_cache_size: 10000
warm_pool_size: 3
enable_philosophical_reasoning: true
log_level: WARNING
prewarm_singletons: true
```

### High-Load Environment
```yaml
# Maximum performance
embedding_cache_size: 50000
warm_pool_size: 5
enable_philosophical_reasoning: true
log_level: ERROR
prewarm_singletons: true
batch_requests: true
distributed_mode: true
```

## Monitoring Commands

```bash
# Check memory usage
ps aux | grep vulcan

# Monitor log for performance issues
tail -f logs/vulcan.log | grep -E "(slow|timeout|failed)"

# Check singleton initialization
grep "Singletons.*created" logs/vulcan.log

# Monitor query routing
grep "ToolSelector.*Selected" logs/vulcan.log | tail -20

# Check crystallization
grep "Crystallized.*principles" logs/vulcan.log | tail -20
```

## Support

For performance issues not covered in this document:
1. Check logs for specific error messages
2. Review system resource usage (CPU, memory, disk I/O)
3. Enable DEBUG logging temporarily for detailed diagnostics
4. Open an issue with:
   - Log excerpts showing the issue
   - System resource metrics
   - Query examples that reproduce the issue
   - Configuration being used
