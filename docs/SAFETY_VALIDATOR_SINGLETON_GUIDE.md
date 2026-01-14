# Safety Validator Singleton Pattern - Implementation Guide

## Problem
Each component was creating its own `EnhancedSafetyValidator()` instance, leading to:
- ~30x duplication of safety validators
- ~15x duplication of ComplianceMapper, BiasDetector, RollbackManager, etc.
- High memory usage (95.2% reported)
- Inconsistent safety decisions across components

## Solution
Implement singleton pattern with dependency injection to share a single safety validator instance across all components.

## Pattern Applied

### 1. Component Initialization Signature
Add `safety_validator` parameter to `__init__`:

```python
def __init__(
 self,
 # ... existing params ...
 safety_config: Optional[Dict[str, Any]] = None, # Deprecated
 safety_validator=None, # NEW: Preferred way to pass shared instance
):
```

### 2. Initialization Logic
Replace direct instantiation with this pattern:

```python
# Initialize safety validator - prefer shared instance
if safety_validator is not None:
 # Use provided shared instance (PREFERRED - prevents duplication)
 self.safety_validator = safety_validator
 logger.info(f"{self.__class__.__name__}: Using shared safety validator instance")
elif SAFETY_VALIDATOR_AVAILABLE:
 # Fallback: try to get singleton, or create new instance
 try:
 from ..safety.safety_validator import initialize_all_safety_components
 self.safety_validator = initialize_all_safety_components(
 config=safety_config, reuse_existing=True
 )
 logger.info(f"{self.__class__.__name__}: Using singleton safety validator")
 except Exception as e:
 logger.debug(f"Could not get singleton safety validator: {e}")
 # Last resort: create new instance (causes duplication)
 if isinstance(safety_config, dict) and safety_config:
 self.safety_validator = EnhancedSafetyValidator(
 SafetyConfig.from_dict(safety_config)
 )
 else:
 self.safety_validator = EnhancedSafetyValidator()
 logger.warning(f"{self.__class__.__name__}: Created new safety validator instance (may cause duplication)")
else:
 self.safety_validator = None
 logger.warning(
 f"{self.__class__.__name__}: Safety validator not available - operating without safety checks"
 )
```

### 3. Parent Component Pattern
Parent components should create ONE validator and pass it to children:

```python
# In semantic_bridge_core.py:
# Initialize safety validator ONCE using singleton pattern
try:
 from ..safety.safety_validator import initialize_all_safety_components
 self.safety_validator = initialize_all_safety_components(
 config=safety_config, reuse_existing=True
 )
 logger.info("SemanticBridge: Using singleton safety validator")
except Exception:
 # ... fallback ...

# Pass to all children
self.concept_mapper = ConceptMapper(
 world_model=world_model, 
 safety_validator=self.safety_validator # Pass instance, not config
)
self.transfer_engine = TransferEngine(
 world_model=world_model,
 safety_validator=self.safety_validator # Pass instance, not config
)
# etc.
```

## Files Already Fixed

### Semantic Bridge Components
- ✅ `src/vulcan/semantic_bridge/semantic_bridge_core.py` - Creates singleton, passes to children
- ✅ `src/vulcan/semantic_bridge/concept_mapper.py` - Accepts and uses shared instance
- ✅ `src/vulcan/semantic_bridge/transfer_engine.py` - Accepts and uses shared instance
- ✅ `src/vulcan/semantic_bridge/conflict_resolver.py` - Accepts and uses shared instance
- ✅ `src/vulcan/semantic_bridge/domain_registry.py` - Accepts and uses shared instance

### World Model Components
- ✅ `src/vulcan/world_model/dynamics_model.py` - Accepts parameter, uses singleton in lazy loader

### Problem Decomposer
- ✅ `src/vulcan/problem_decomposer/problem_decomposer_core.py` - Accepts and uses shared instance

## Files Still Needing Fixes

### World Model Components (High Priority)
These need the same pattern applied:

1. `src/vulcan/world_model/causal_graph.py` - Line 1749
2. `src/vulcan/world_model/world_model_router.py` - Line 487
3. `src/vulcan/world_model/confidence_calibrator.py` - Lines 856, 1434 (2 instances)
4. `src/vulcan/world_model/invariant_detector.py` - Lines 1304, 1773 (2 instances)
5. `src/vulcan/world_model/prediction_engine.py` - Line 1827

**Parent Component to Update:**
- `src/vulcan/world_model/world_model_core.py` - Should create ONE validator and pass to all children

### Generation Components
6. `src/generation/safe_generation.py` - Needs the pattern

### Top-Level Components
7. `graphix_vulcan_llm.py` - Main entry point, should use singleton

## Special Cases

### Lazy Loading Pattern
For components that lazy-load the validator (like `dynamics_model.py`), update the lazy loader to try singleton first:

```python
def _get_safety_validator(self):
 if self.safety_validator is not None:
 return self.safety_validator
 
 with self.lock:
 # Double-check
 if self.safety_validator is not None:
 return self.safety_validator
 
 try:
 # Import modules
 validator_mod = importlib.import_module("..safety.safety_validator", package=__package__)
 initialize_all_safety_components = getattr(validator_mod, "initialize_all_safety_components", None)
 
 # FIXED: Try singleton first
 if initialize_all_safety_components is not None:
 try:
 self.safety_validator = initialize_all_safety_components(
 config=self.safety_config, reuse_existing=True
 )
 logger.info(f"{self.__class__.__name__}: Using singleton safety validator")
 return self.safety_validator
 except Exception as e:
 logger.debug(f"Could not get singleton safety validator: {e}")
 
 # Fallback to new instance
 if isinstance(self.safety_config, dict) and self.safety_config:
 config_obj = SafetyConfig.from_dict(self.safety_config)
 self.safety_validator = EnhancedSafetyValidator(config_obj)
 else:
 self.safety_validator = EnhancedSafetyValidator()
 
 logger.warning(f"{self.__class__.__name__}: Created new safety validator instance (may cause duplication)")
 except Exception as e:
 logger.error(f"Safety validator initialization failed: {e}")
 # Create stub validator
 # ... existing fallback code ...

### Test Mode Configuration
For components with special test_mode handling (like `problem_decomposer_core.py`), preserve the complex logic but wrap it in the fallback section.

## Testing the Fix

After applying to all files:

1. **Memory Usage Test**: Monitor memory during startup - should see significant reduction
2. **Instance Count**: Add logging to count validator instances - should be ~1-3 instead of ~30+
3. **Functionality Test**: Ensure safety validation still works correctly
4. **Test Suite**: Run existing tests to ensure no regressions

## Verification Commands

```bash
# Count remaining direct instantiations
grep -r "self.safety_validator = EnhancedSafetyValidator()" --include="*.py" src/

# Count singleton usage logging
grep -r "Using singleton safety validator" --include="*.py" src/ | wc -l

# Count new instance warnings
grep -r "Created new safety validator instance" --include="*.py" src/ | wc -l
```

## Expected Outcome

- Memory usage should drop from 95.2% to more reasonable levels
- Startup time should improve (fewer component initializations)
- Safety decisions will be consistent across all components
- Clear logging will show which components use singleton vs create new instances
