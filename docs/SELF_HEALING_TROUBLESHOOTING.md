# Self-Healing System Troubleshooting Guide

## Overview

This guide helps you verify and troubleshoot the VULCAN-AGI self-healing/self-improvement system.

## Quick Diagnosis

Run the diagnostic script to check your setup:

```bash
python scripts/check_self_healing.py
```

This will automatically:
1. Clear bytecode cache
2. Verify imports
3. Check required methods
4. Run full diagnostics
5. Provide recommendations if issues are found

## Common Issues and Solutions

### Issue 1: AttributeError - '_handle_improvement_alert' not found

**Symptoms:**
- System logs show "Self-improvement system enabled"
- Runtime error: `AttributeError: 'WorldModel' object has no attribute '_handle_improvement_alert'`

**Root Cause:**
Stale Python bytecode cache (`.pyc` files) can cache old versions of the code.

**Solution:**
```bash
# Option 1: Use the clear cache script
python scripts/clear_cache.py

# Option 2: Manual cleanup
find . -type d -name '__pycache__' -exec rm -rf {} +
find . -name '*.pyc' -delete

# Then restart your Python process
```

### Issue 2: Self-improvement drive not initializing

**Symptoms:**
- Logs show "⚠ Self-improvement drive not available - module not found"
- Or "⚠ Self-improvement drive disabled in config"

**Solution:**

**Check 1:** Verify meta-reasoning module is available
```python
from vulcan.world_model.meta_reasoning import SelfImprovementDrive
print("✓ SelfImprovementDrive imported successfully")
```

**Check 2:** Verify configuration is enabled
The system checks multiple configuration sources in order:

1. Root config: `enable_self_improvement: true`
2. Nested: `world_model.enable_self_improvement: true`
3. Alternative nested: `world_model.self_improvement_enabled: true`
4. Environment variable: `VULCAN_ENABLE_SELF_IMPROVEMENT=1`

Enable via environment variable:
```bash
export VULCAN_ENABLE_SELF_IMPROVEMENT=1
```

Or in your config file:
```yaml
enable_self_improvement: true
```

### Issue 3: Method exists but still getting AttributeError

**Symptoms:**
- Running `hasattr(WorldModel, '_handle_improvement_alert')` returns `True`
- But still get AttributeError at runtime

**Root Cause:**
Multiple Python processes or notebook kernels with stale imports.

**Solution:**
```bash
# 1. Clear cache
python scripts/clear_cache.py

# 2. Kill all Python processes
pkill -9 python

# 3. Restart your application
# 4. Verify methods exist
python -c "
from vulcan.world_model.world_model_core import WorldModel
assert hasattr(WorldModel, '_handle_improvement_alert'), 'Method missing!'
print('✓ Method verified')
"
```

### Issue 4: Import errors for dependencies

**Symptoms:**
- `ModuleNotFoundError: No module named 'numpy'`
- Or other missing dependencies

**Solution:**
```bash
# Install all requirements
pip install -r requirements.txt

# Or if using conda/mamba
conda install --file requirements.txt
```

## Diagnostic Commands

### Check method availability
```python
from vulcan.world_model.world_model_core import WorldModel
import inspect

# Check if method exists
print("Has method:", hasattr(WorldModel, '_handle_improvement_alert'))

# Get method signature
if hasattr(WorldModel, '_handle_improvement_alert'):
    method = getattr(WorldModel, '_handle_improvement_alert')
    print("Signature:", inspect.signature(method))
    print("Callable:", callable(method))
```

### Run full diagnostics
```python
from vulcan.world_model.world_model_core import (
    print_diagnostics,
    print_self_healing_diagnostics,
    validate_self_healing_setup
)

# Print all diagnostics
print_diagnostics()
print_self_healing_diagnostics()

# Get programmatic validation
is_working, issues = validate_self_healing_setup()
print(f"Working: {is_working}")
if issues:
    for issue in issues:
        print(f"  - {issue}")
```

### Check component availability
```python
from vulcan.world_model.world_model_core import check_component_availability

components = check_component_availability()
print("Meta-reasoning:", components.get('meta_reasoning'))
print("Self-improvement:", components.get('self_improvement'))
```

## Understanding the Initialization Process

The self-improvement drive initialization follows these steps:

1. **Check configuration flags** (multiple sources)
2. **Verify meta-reasoning is available**
3. **Verify SelfImprovementDrive class loaded**
4. **Assert required methods exist** (NEW - catches issues early)
5. **Initialize SelfImprovementDrive**
6. **Register callbacks**

The new assertions at step 4 will catch missing methods early with a clear error message:
```
AssertionError: Missing required method: _handle_improvement_alert in WorldModel
```

## Preventing Issues

### Best Practices

1. **Always clear cache after pulling new code:**
   ```bash
   git pull
   python scripts/clear_cache.py
   ```

2. **Use the diagnostic script regularly:**
   ```bash
   python scripts/check_self_healing.py
   ```

3. **Verify methods in your CI/CD pipeline:**
   ```bash
   python -c "from vulcan.world_model.world_model_core import validate_self_healing_setup; is_ok, issues = validate_self_healing_setup(); exit(0 if is_ok else 1)"
   ```

4. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Configuration Best Practices

**Recommended config structure:**
```yaml
# config.yaml
enable_self_improvement: true

world_model:
  enable_self_improvement: true
  self_improvement_config: "configs/intrinsic_drives.json"
  self_improvement_state: "data/agent_state.json"
```

**Environment variable override (highest priority):**
```bash
export VULCAN_ENABLE_SELF_IMPROVEMENT=1
```

## Advanced Troubleshooting

### Debug imports
```python
import sys
import importlib

# Force reload of modules
if 'vulcan.world_model.world_model_core' in sys.modules:
    importlib.reload(sys.modules['vulcan.world_model.world_model_core'])

from vulcan.world_model.world_model_core import WorldModel
print("Methods:", [m for m in dir(WorldModel) if 'improvement' in m.lower()])
```

### Check for multiple installations
```python
import vulcan.world_model.world_model_core as wm
print("Module path:", wm.__file__)

# Should point to your current repo, not site-packages
```

### Verify method signatures match expectations
```python
from vulcan.world_model.world_model_core import WorldModel
import inspect

method = getattr(WorldModel, '_handle_improvement_alert')
sig = inspect.signature(method)

print("Parameters:", list(sig.parameters.keys()))
# Expected: ['self', 'severity', 'alert_data']
```

## Getting Help

If you've tried all the solutions above and still have issues:

1. **Collect diagnostic information:**
   ```bash
   python scripts/check_self_healing.py > diagnostics.txt 2>&1
   ```

2. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Check for file modifications:**
   ```bash
   git status
   git diff src/vulcan/world_model/world_model_core.py
   ```

4. **File an issue with:**
   - Output from `check_self_healing.py`
   - Python version
   - Operating system
   - Any modifications you've made
   - Full error traceback

## Success Indicators

When everything is working correctly, you should see:

```
✓ Self-improvement system enabled
✓ SelfImprovementDrive loaded successfully
✓ Self-improvement system fully available
✓ Self-improvement drive initialized
```

And the diagnostic script should show:
```
✓ ALL CHECKS PASSED - Self-healing system is properly configured
```

## Related Documentation

- [Self-Improvement Drive Architecture](../docs/self_improvement.md)
- [Meta-Reasoning System](../docs/meta_reasoning.md)
- [Configuration Guide](../docs/configuration.md)
