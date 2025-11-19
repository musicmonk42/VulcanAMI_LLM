# VULCAN-AGI Utility Scripts

This directory contains utility scripts for maintaining and diagnosing the VULCAN-AGI system.

## Self-Healing Diagnostics

### `check_self_healing.py`

Comprehensive diagnostic tool for verifying the self-healing/self-improvement system setup.

**Usage:**
```bash
python scripts/check_self_healing.py
```

**What it does:**
1. Clears bytecode cache automatically
2. Verifies all imports work correctly
3. Checks that required methods exist in WorldModel
4. Runs module-level diagnostics
5. Provides actionable recommendations if issues are found

**Exit codes:**
- `0`: All checks passed, system is ready
- `1`: Issues detected, see output for details

**Example output:**
```
======================================================================
VULCAN-AGI Self-Healing System Diagnostic Tool
======================================================================

Clearing bytecode cache...
✓ Cleared 15 cached files/directories

Checking imports...
  ✓ vulcan.world_model.world_model_core imported
  ✓ WorldModel class imported
  ✓ SelfImprovementDrive imported

Checking WorldModel methods...
  ✓ _handle_improvement_alert() exists and is callable
  ✓ _check_improvement_approval() exists and is callable
  ✓ start_autonomous_improvement() exists and is callable
  ...

======================================================================
✓ ALL CHECKS PASSED - Self-healing system is properly configured
======================================================================
```

### `clear_cache.py`

Utility to clear Python bytecode cache files.

**Usage:**
```bash
python scripts/clear_cache.py
```

**What it does:**
1. Removes all `__pycache__` directories recursively
2. Deletes all `.pyc` files
3. Reports statistics on files removed

**When to use:**
- After pulling new code from git
- When seeing AttributeError for methods that should exist
- Before running diagnostics
- When Python seems to be using old code

**Example output:**
```
============================================================
Clearing Python Bytecode Cache
============================================================

Removing __pycache__ directories...
  Removed: src/vulcan/__pycache__
  Removed: src/vulcan/world_model/__pycache__
  ...

Removing .pyc files...
  Removed: src/vulcan/world_model/world_model_core.pyc
  ...

============================================================
✓ Cache cleared successfully
  - 12 __pycache__ directories removed
  - 3 .pyc files removed
============================================================
```

## Integration into CI/CD

### GitHub Actions

Add to your workflow:

```yaml
- name: Verify self-healing setup
  run: |
    python scripts/check_self_healing.py
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python scripts/check_self_healing.py
exit $?
```

### Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Troubleshooting

If diagnostics fail, see the comprehensive troubleshooting guide:
- [Self-Healing Troubleshooting Guide](../docs/SELF_HEALING_TROUBLESHOOTING.md)

Common quick fixes:

```bash
# Clear cache
python scripts/clear_cache.py

# Kill all Python processes
pkill -9 python

# Verify setup
python scripts/check_self_healing.py
```

## Development

When adding new required methods to WorldModel for self-healing:

1. Add the method to WorldModel class
2. Update the assertions in `__init__` if it's critical
3. Update `validate_self_healing_setup()` to check for it
4. Update `check_self_healing.py` required_methods list
5. Update documentation

## Related Documentation

- [Self-Healing Troubleshooting](../docs/SELF_HEALING_TROUBLESHOOTING.md) - Comprehensive troubleshooting guide
- [Configuration Guide](../docs/configuration.md) - How to enable self-improvement
- [Architecture](../docs/architecture.md) - System architecture overview
