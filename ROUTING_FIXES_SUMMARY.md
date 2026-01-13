# Routing Override Fixes - Implementation Summary

## Overview

This document summarizes the comprehensive fix for query routing pipeline override issues that caused queries to be misrouted to the wrong reasoning tools.

## Problems Fixed

### Issue #1: LLM Classification Prompt Didn't Distinguish Categories
**Problem**: The LLM classifier prompt lacked clear distinctions between PROBABILISTIC, LOGICAL, CAUSAL, and MATHEMATICAL queries, causing misclassifications.

**Evidence**: Bayesian queries with "P(X|Y)" and "sensitivity/specificity" were classified as MATHEMATICAL instead of PROBABILISTIC.

**Fix**: Updated the prompt in `src/vulcan/routing/query_classifier.py` with:
- Detailed category descriptions with specific keywords
- Concrete examples for each category
- CRITICAL DISTINCTIONS section to prevent common misclassifications

### Issue #2: Override-to-General Logic in apply_reasoning_impl.py
**Problem**: The `SIMPLE_QUERY_CATEGORIES` check forced `['general']` tools even for reasoning queries, overriding correct classifications.

**Evidence**: Queries marked as PROBABILISTIC or LOGICAL were being overridden to `['general']`.

**Fix**: Updated `src/vulcan/reasoning/integration/apply_reasoning_impl.py`:
- Created `TRULY_SIMPLE_CATEGORIES` (only GREETING, CHITCHAT)
- Created `REASONING_CATEGORIES` (PROBABILISTIC, LOGICAL, CAUSAL, etc.)
- Only override to `['general']` for truly simple categories
- Preserve reasoning tools for reasoning categories

### Issue #3: AgentPool Always Overrode Router's Selection
**Problem**: AgentPool blindly accepted integration's `selected_tools`, even when integration returned `['general']` as a fallback.

**Evidence**: Router selected `['probabilistic', 'symbolic']`, but integration's `['general']` fallback overwrote it.

**Fix**: Updated `src/vulcan/orchestrator/agent_pool.py`:
- Check `override_router_tools` flag before overriding
- Detect `['general']` fallback vs. authoritative override
- Preserve router's specific tools when integration returns general fallback
- Only override when explicitly requested

## Changes Made

### 1. src/vulcan/routing/query_classifier.py
- **Lines ~2161-2220**: Updated LLM classification prompt
- Added detailed category descriptions with examples
- Added CRITICAL DISTINCTIONS section
- Improved tool mappings

### 2. src/vulcan/reasoning/integration/apply_reasoning_impl.py
- **Lines ~365-395**: Added category set definitions
  - `TRULY_SIMPLE_CATEGORIES`: Only GREETING and CHITCHAT
  - `REASONING_CATEGORIES`: All reasoning types
- **Lines ~470-510**: Updated override logic
  - Check for reasoning categories before overriding
  - Only override truly simple queries
  - Preserve classifier's suggested tools for reasoning

### 3. src/vulcan/orchestrator/agent_pool.py
- **Lines ~3217-3285**: Rewrote override logic
  - Check `override_router_tools` flag
  - Detect general fallback vs. authoritative override
  - Preserve router's specific tools over fallback
  - Only override when explicitly requested

### 4. src/vulcan/reasoning/integration/types.py
- **Lines ~324-353**: Added `override_router_tools` field to `ReasoningResult`
  - Default value: `False`
  - Set to `True` only when integration has high confidence
  - Used for self-introspection and explicit delegation

### 5. tests/test_routing_override_fixes.py (NEW)
- Comprehensive test suite for all fixes
- Tests LLM classification improvements
- Tests apply_reasoning doesn't override specialized tools
- Tests AgentPool preserves router's tool selection
- Tests override_router_tools field functionality

## Validation

### Manual Validation
Created `validate_routing_fixes.py` script that validates:
1. ✅ ReasoningResult has `override_router_tools` field with correct defaults
2. ✅ Category sets (TRULY_SIMPLE_CATEGORIES, REASONING_CATEGORIES) are correct
3. ✅ General fallback detection logic works correctly

All validation tests pass.

### Code Compilation
All modified files compile successfully without syntax errors.

## Expected Behavior After Fix

### Bayesian/Probabilistic Queries
**Before**: Classified as MATHEMATICAL → routed to math tool → "No mathematical expression found"  
**After**: Classified as PROBABILISTIC → routed to probabilistic tool → correct Bayesian computation

### SAT/Logic Queries
**Before**: Classified as MATHEMATICAL → routed to math tool → "No mathematical expression found"  
**After**: Classified as LOGICAL → routed to symbolic tool → correct SAT solving

### Causal Reasoning Queries
**Before**: Classified as simple → overridden to ['general'] → wrong execution  
**After**: Classified as CAUSAL → preserved → routed to causal tool → correct analysis

### Router Selection Preservation
**Before**: Router selects ['probabilistic'] → integration returns ['general'] fallback → override to ['general']  
**After**: Router selects ['probabilistic'] → integration returns ['general'] fallback → preserve ['probabilistic']

## Files Modified

1. `src/vulcan/routing/query_classifier.py` - Improved LLM prompt
2. `src/vulcan/reasoning/integration/apply_reasoning_impl.py` - Fixed override logic
3. `src/vulcan/orchestrator/agent_pool.py` - Fixed tool selection preservation
4. `src/vulcan/reasoning/integration/types.py` - Added override_router_tools field
5. `tests/test_routing_override_fixes.py` - NEW comprehensive test suite
6. `validate_routing_fixes.py` - NEW validation script

## Breaking Changes

None. All changes are backward compatible:
- New field `override_router_tools` has a default value of `False`
- Existing code paths continue to work
- Only behavior change is fixing the misrouting bugs

## Testing

Run tests with:
```bash
# Run new test suite
python -m pytest tests/test_routing_override_fixes.py -v

# Run validation script
python validate_routing_fixes.py
```

## References

- Problem statement: Comprehensive Fix: Query Routing Pipeline Override Issues
- Production logs showing misrouting evidence
- Issue #6: Router tool override bug
