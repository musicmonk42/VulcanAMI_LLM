# FIX #4: SAT/Symbolic Query Detection Improvement

## Problem Statement

SAT satisfiability queries were being misrouted to ensemble/probabilistic engines instead of the symbolic engine.

**Example from problem statement:**
```
Query: "SAT Satisfiability - Is the set satisfiable?"
Current routing: Ensemble/Probabilistic → "This query does not appear to be probabilistic"
Expected routing: Symbolic engine for SAT solving
```

## Root Cause

1. **Weak SAT detection**: The existing LOGICAL_KEYWORDS included "sat" as a substring match, which:
   - Caused false positives: "I sat down" matched "sat"
   - Required 2+ keyword matches to trigger (threshold=2)
   
2. **Ordering issues**: SAT detection happened too late in the classification pipeline, allowing other categories to capture SAT queries first

## Solution Implemented

### 1. Added `_classify_symbolic_logic()` Method

Located in `/src/vulcan/routing/query_classifier.py` (lines 1375-1460)

**Key Features:**
- Explicit SAT phrase detection:
  - "satisfiable", "satisfiability", "unsatisfiable"
  - "sat problem", "sat solver", "boolean satisfiability"
  - "propositions:", "constraints:", "cnf", "dnf"
  
- Word-boundary checking using `SAT_WORD_BOUNDARY_PATTERN`:
  - Matches: "SAT problem", "sat solver", "Is it sat?"
  - Does NOT match: "satisfiable", "I sat down", "The cat sat"
  
- Logical symbol detection with context:
  - Symbols: →, ∧, ∨, ¬, ↔, ⊢, ⊨, ->, /\, \/, ~
  - Context words: proposition, formula, constraint, valid, entail
  
- Comprehensive logging for debugging

### 2. Early SAT Detection

SAT checking now happens BEFORE probabilistic checks in `_classify_by_keywords()`:

```python
# Check SAT/symbolic logic patterns early (high priority)
symbolic_tool = self._classify_symbolic_logic(query_original)
if symbolic_tool:
    has_satisfiable = "satisfiable" in query_lower or "satisfiability" in query_lower
    confidence = 0.95 if has_satisfiable else 0.90
    
    return QueryClassification(
        category=QueryCategory.LOGICAL.value,
        complexity=0.7,
        suggested_tools=["symbolic"],
        skip_reasoning=False,
        confidence=confidence,
        source="keyword",
    )
```

### 3. Enhanced Strong Indicators

Added to `STRONG_LOGICAL_INDICATORS` (lines 295-303):
- "satisfiability"
- "boolean satisfiability"

These single keywords now trigger LOGICAL classification without requiring threshold.

### 4. Performance Optimizations

Moved patterns to module-level constants to avoid recreation overhead:

- **SAT_WORD_BOUNDARY_PATTERN** (line 956):
  ```python
  SAT_WORD_BOUNDARY_PATTERN: re.Pattern = re.compile(r'\bsat\b', re.IGNORECASE)
  ```

- **LOGICAL_CONNECTIVE_SYMBOLS** (lines 308-312):
  ```python
  LOGICAL_CONNECTIVE_SYMBOLS: Tuple[str, ...] = (
      '→', '∧', '∨', '¬', '↔', '⊢', '⊨',  # Unicode
      '->', '/\\', '\\/', '~',              # ASCII
  )
  ```

## Test Results

All 7 test cases pass:

✅ **SAT Queries Correctly Route to LOGICAL/Symbolic:**
- "SAT Satisfiability - Is the set satisfiable?" → LOGICAL (conf=0.95)
- "Is the set satisfiable?" → LOGICAL (conf=0.95)
- "SAT problem: A→B, B→C, ¬C" → LOGICAL (conf=0.90)
- "Propositions: P, Q, P→Q" → LOGICAL (conf=0.90)

✅ **No False Positives (Word Boundary Works):**
- "I sat down on the chair" → UNKNOWN (NOT LOGICAL)
- "The cat sat on the mat" → UNKNOWN (NOT LOGICAL)

✅ **No Regressions in Other Categories:**
- "What is P(disease)?" → PROBABILISTIC (conf=0.85)

## Implementation Quality

### Code Standards
- ✅ Comprehensive docstrings with Args/Returns
- ✅ Type hints throughout
- ✅ Proper error handling and logging
- ✅ Follows existing code patterns
- ✅ Performance optimized (module-level constants)

### Code Review Feedback Addressed
- ✅ Moved regex compilation to module level (SAT_WORD_BOUNDARY_PATTERN)
- ✅ Moved logical symbols list to module level (LOGICAL_CONNECTIVE_SYMBOLS)
- ✅ Follows existing pattern (VALUE_CONFLICT_PATTERNS, SPECULATION_PATTERNS)

## Files Modified

1. **src/vulcan/routing/query_classifier.py**
   - Added `_classify_symbolic_logic()` method (80 lines)
   - Enhanced `STRONG_LOGICAL_INDICATORS` (+2 items)
   - Added module-level constants (SAT_WORD_BOUNDARY_PATTERN, LOGICAL_CONNECTIVE_SYMBOLS)
   - Integrated early SAT checking in `_classify_by_keywords()`
   - Total: ~110 lines added

## Impact

**Before Fix:**
```
"SAT Satisfiability - Is the set satisfiable?"
→ Ensemble/Probabilistic engine
→ "This query does not appear to be probabilistic" error
→ Incorrect handling
```

**After Fix:**
```
"SAT Satisfiability - Is the set satisfiable?"
→ LOGICAL category
→ Symbolic engine
→ Correct SAT solving
→ Confidence: 0.95
```

## Key Improvements

1. **Robust Detection**: Word-boundary checking prevents false positives
2. **High Priority**: Early checking ensures correct routing
3. **High Confidence**: 0.95 for explicit "satisfiable" keyword
4. **Performance**: Module-level constants avoid recreation overhead
5. **No Regressions**: All existing query types still route correctly

## Commits

1. `08a9e2e` - Initial implementation with _classify_symbolic_logic()
2. `ab53f26` - Optimize SAT detection: Move regex to module-level constant
3. `2ecdd2a` - Optimize logical symbols: Move to module-level constant

Total commits: 3
Total lines added: ~120
Files modified: 1

---
**Status**: ✅ COMPLETE - All tests passing, code review feedback addressed
