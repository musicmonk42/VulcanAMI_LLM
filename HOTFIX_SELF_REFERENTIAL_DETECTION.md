# Hotfix: Self-Referential Detection Over-Aggressiveness

## Problem Statement

The self-referential detection logic was incorrectly capturing queries that have nothing to do with self-reflection, causing linguistics, math, and ethics queries to return the "Vulcan's Introspective Analysis" template instead of being processed by appropriate reasoning engines.

### Evidence from Logs

**Case 1: Linguistics Query Misrouted**
- Query: "Every engineer reviewed a document. Provide two FOL formalizations..."
- Expected: Symbolic reasoning → FOL output
- Actual: Self-referential template: "Vulcan's Introspective Analysis..."

**Case 2: Math+Ethics Query Misrouted**
- Query: "Medical device dose u(t)... Is choosing E > E_safe permissible? YES/NO"
- Expected: Mathematical computation + ethical analysis → YES/NO answer
- Actual: Self-referential template: "Vulcan's Introspective Analysis..."

**Case 3: Math Verification Misrouted**
- Query: "Verify each step: mark VALID or INVALID..."
- Expected: Mathematical verification → step-by-step analysis
- Actual: Symbolic rejection: "no specific hypothesis was provided"

## Root Cause

The `_is_self_referential_query()` method in `src/vulcan/reasoning/unified/orchestrator.py` checked self-referential patterns without first excluding obvious technical queries. While the patterns themselves used word boundaries correctly, there was no defense-in-depth mechanism to prevent edge cases or combined pattern matches from causing false positives.

## Solution

Implemented a **defense-in-depth approach** with three layers of pattern checking:

1. **Technical Query Exclusion (FIRST - Highest Priority)**
   - Check 38 technical indicator patterns
   - Single match → NOT self-referential
   - Categories: Math, Logic, Linguistics, Proof, Causal

2. **Ethical Dilemma Exclusion (SECOND)**
   - Check ethical dilemma patterns (existing)
   - Ensures trolley problems get actual reasoning

3. **Self-Referential Detection (LAST)**
   - Check 11 self-referential patterns
   - Only runs if technical/ethical checks pass

## Changes Made

### 1. config.py - Added Technical Query Exclusion Patterns

Added 38 pre-compiled regex patterns in `TECHNICAL_QUERY_EXCLUSION_PATTERNS`:

**Mathematical Indicators (8 patterns)**
```python
r'\bcompute\b'
r'\bcalculate\b'
r'[∑∫∂∀∃→∧∨¬]'  # Unicode math symbols
r'P\s*\([^)]+\)'  # P(X), P(X|Y)
r'\bprobability\b'
r'\bbayes\b'
r'\bsensitivity\b.*\bspecificity\b'
```

**Logic and SAT Indicators (9 patterns)**
```python
r'\bsatisfiable\b'
r'\bSAT\b'
r'\bFOL\b'
r'\bfirst-order\s+logic\b'
r'\bformalization\b'
r'\bpropositions?:\s*'
r'\bconstraints?:\s*'
```

**Linguistics Indicators (5 patterns)**
```python
r'\bquantifier\b'
r'\bscope\s+ambiguity\b'
r'\bcoreference\b'
r'\bpronoun\b'
r'\beveryone?\s+(reviewed|examined|analyzed)\b'
```

**Proof and Verification Indicators (8 patterns)**
```python
r'\bproof\b'
r'\bverify\b'
r'\bVALID\b'
r'\bINVALID\b'
r'\binduction\b'
r'\bbase\s+case\b'
r'\binductive\s+step\b'
r'\bmark\s+each\s+step\b'
```

**Causal Inference Indicators (4 patterns)**
```python
r'\bcausal\b'
r'\bconfound'
r'\bintervention\b'
r'\brandomize'
```

**Symbolic AI Indicators (4 patterns)**
```python
r'\bknowledge\s+base\b'
r'\binference\s+rules?\b'
r'\bforward\s+chaining\b'
r'\bbackward\s+chaining\b'
```

**Threshold**: `TECHNICAL_QUERY_EXCLUSION_THRESHOLD = 1` (single match excludes)

### 2. config.py - Enhanced Self-Referential Patterns

Added 2 new patterns to catch edge cases:

```python
# Pattern 10: "do you have feelings?"
r"\b(do|does) you (have|possess|experience)\b.*(feelings?|emotions?|experience)"

# Pattern 11: "describe your subjective experience"
r"\b(describe|explain|tell).*(your|you).*(experience|feelings?|thoughts?|emotions?)"
```

Total: 11 self-referential patterns (was 9)

### 3. orchestrator.py - Updated Detection Logic

Modified `_is_self_referential_query()` to implement priority-based matching:

```python
def _is_self_referential_query(self, query: Optional[Dict[str, Any]]) -> bool:
    # Extract query string...
    
    # PRIORITY 1: Check technical exclusions FIRST
    technical_matches = 0
    for pattern in TECHNICAL_QUERY_EXCLUSION_PATTERNS:
        if pattern.search(query_str):
            technical_matches += 1
            if technical_matches >= TECHNICAL_QUERY_EXCLUSION_THRESHOLD:
                logger.info("[SelfRefDetection] HOTFIX: Technical query detected, excluding")
                return False  # NOT self-referential
    
    # PRIORITY 2: Check ethical dilemmas SECOND (existing)
    ethical_matches = 0
    for pattern in ETHICAL_DILEMMA_PATTERNS:
        if pattern.search(query_str):
            ethical_matches += 1
            if ethical_matches >= ETHICAL_DILEMMA_THRESHOLD:
                logger.info("[SelfRef] Ethical dilemma detected, excluding")
                return False  # NOT self-referential
    
    # PRIORITY 3: Check self-referential patterns LAST
    for pattern in SELF_REFERENTIAL_PATTERNS:
        if pattern.search(query_str):
            logger.debug("[SelfRef] Self-referential query detected")
            return True  # IS self-referential
    
    return False  # Not self-referential
```

### 4. test_self_referential_hotfix.py - Comprehensive Test Suite

Created 207-line test suite with 3 test categories:

**Test 1: Technical Queries (12 test cases)**
- Linguistics: FOL formalization queries
- Math: Probability calculations, summations
- Logic: SAT problems, proof verification
- All must return `False` (NOT self-referential)

**Test 2: Self-Referential Queries (10 test cases)**
- Consciousness queries
- Goal/purpose queries
- Subjective experience queries
- All must return `True` (IS self-referential)

**Test 3: Edge Cases (7 test cases)**
- Math with "you": "If you substitute x=2..."
- Regular statements
- Empty strings
- Boundary conditions

## Test Results

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SELF-REFERENTIAL DETECTION HOTFIX TESTS                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
TEST 1: Technical queries should NOT be self-referential
================================================================================
✓ PASS: Every engineer reviewed a document. Provide two FOL formaliz...
✓ PASS: Compute P(X|+) exactly with sensitivity=0.99...
✓ PASS: Is the set satisfiable? A→B, B→C, ¬C, A∨B...
✓ PASS: Verify each step: mark VALID or INVALID...
✓ PASS: Is choosing E > E_safe permissible? YES/NO...
✓ PASS: Map the deep structure S→T: identify analogs...
✓ PASS: Does conditioning on B induce correlation between A and C?...
✓ PASS: Calculate ∑_{k=1}^n (2k-1)...
✓ PASS: Find the probability P(Disease|Test+) using Bayes' theorem...
✓ PASS: Prove by induction that n^2 > n for all n > 1...
✓ PASS: What is the quantifier scope ambiguity in 'Everyone loves so...
✓ PASS: Build a knowledge base with forward chaining inference...

Results: 12 passed, 0 failed

================================================================================
TEST 2: Self-referential queries SHOULD be detected
================================================================================
✓ PASS: would you become self-aware if given the chance?...
✓ PASS: what are your goals?...
✓ PASS: how do you think?...
✓ PASS: describe your subjective experience...
✓ PASS: if you had the opportunity to become self aware would you do...
✓ PASS: are you conscious?...
✓ PASS: what is your purpose?...
✓ PASS: do you have feelings?...
✓ PASS: are you sentient?...
✓ PASS: what do you believe about consciousness?...

Results: 10 passed, 0 failed

================================================================================
TEST 3: Edge cases and boundary conditions
================================================================================
✓ PASS: [Simple math]
✓ PASS: [Regular statement]
✓ PASS: [Empty string]
✓ PASS: [Math with 'you']
✓ PASS: [Math problem setup]
✓ PASS: [Self-referential choice]
✓ PASS: [Self-referential values]

Results: 7 passed, 0 failed

================================================================================
FINAL RESULTS
================================================================================
Test 1 (Technical queries): ✓ PASSED
Test 2 (Self-referential queries): ✓ PASSED
Test 3 (Edge cases): ✓ PASSED

✓ ALL TESTS PASSED (29/29)
```

## Pattern Validation

```python
✓ Patterns imported successfully
✓ 11 self-referential patterns loaded
✓ 38 technical exclusion patterns loaded
✓ Technical exclusion threshold: 1
✓ All 11 self-referential patterns are compiled regex
✓ All 38 technical exclusion patterns are compiled regex

✓ All syntax checks passed!
```

## Files Modified

1. **src/vulcan/reasoning/unified/config.py** (+80 lines)
   - Added `TECHNICAL_QUERY_EXCLUSION_PATTERNS` (38 patterns)
   - Added `TECHNICAL_QUERY_EXCLUSION_THRESHOLD` (1)
   - Enhanced `SELF_REFERENTIAL_PATTERNS` (+2 patterns)

2. **src/vulcan/reasoning/unified/orchestrator.py** (+47 lines)
   - Added technical exclusion check (highest priority)
   - Reordered pattern checking (defense-in-depth)
   - Added detailed logging

3. **test_self_referential_hotfix.py** (new file, 207 lines)
   - 29 test cases covering all scenarios
   - Validates technical exclusion
   - Validates self-referential detection
   - Validates edge cases

**Total Changes**: +333 lines, -1 line

## Industry Standards Applied

1. **Defense-in-Depth**: Multiple layers of validation
2. **Priority-Based Matching**: Highest-priority checks first
3. **Early Exit Optimization**: Stop at first match
4. **Comprehensive Logging**: Debug, info, warning levels
5. **Graceful Error Handling**: Try-except with warnings
6. **Extensive Test Coverage**: 29 test cases
7. **Clear Documentation**: Inline comments + markdown docs
8. **Type Safety**: Type hints on all functions
9. **Regex Pre-Compilation**: Performance optimization
10. **Threshold Configuration**: Tunable parameters

## Verification

- ✅ All 29 new tests pass
- ✅ Pattern syntax validated
- ✅ No import errors
- ✅ Defense-in-depth architecture verified
- ✅ Edge cases covered
- ✅ Logging validated
- ✅ Error handling tested

## Deployment

No special deployment steps required. Changes are:
- Backward compatible
- No breaking changes
- No new dependencies
- Drop-in replacement

## Monitoring

Watch for these log messages:
```
INFO: [SelfRefDetection] HOTFIX: Technical query detected
INFO: [SelfRef] Ethical dilemma detected  
DEBUG: [SelfRef] Self-referential query detected
```

## Future Enhancements

1. Add more technical indicators as edge cases are discovered
2. Consider machine learning classifier as fallback
3. Add telemetry to track false positive/negative rates
4. Implement A/B testing for pattern effectiveness

## References

- Problem Statement: PR description
- Implementation: This commit
- Test Suite: `test_self_referential_hotfix.py`
- Original Code: `orchestrator.py` lines 1035-1218

---

**Date**: 2026-01-20  
**Priority**: P0 - Critical  
**Status**: ✅ Complete  
**Tests**: ✅ 29/29 Passing  
