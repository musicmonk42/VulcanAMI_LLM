# Scripts Update Documentation

## Overview
This document describes the updates made to scripts in the `scripts/`, `stress_tests/`, and `demo/` folders to ensure they run properly with current dependencies and code structure.

## Date Updated
December 16, 2025

## Status
✅ **ALL SCRIPTS NOW WORKING** (13/13 scripts functional)

## Changes Made

### 1. Import Path Fixes

#### scripts/health_smoke.py
- **Issue**: Module import error `ModuleNotFoundError: No module named 'src'`
- **Fix**: 
  - Added project root to `sys.path`
  - Added fallback for `url_validator` import
  - Script now runs successfully
- **Status**: ✅ Working

#### scripts/run_sentiment_tournament.py
- **Issue**: Incorrect import path for `agent_interface`
- **Fix**: Changed `from agent_interface import` to `from src.agent_interface import`
- **Status**: ✅ Working

#### scripts/demo_orchestrator.py
- **Issue**: Generic error message for missing httpx
- **Fix**: Improved error message to show actual import error
- **Status**: ✅ Working

### 2. Configuration Updates

#### .gitignore
- **Added**: `test_ir/` and `benchmark_graphs/` directories
- **Purpose**: Exclude generated test data from version control
- **Status**: ✅ Complete

### 3. Dependency Installation

The following dependencies were installed during testing:
- `numpy` - Required by demo scripts and world_model_core
- `httpx` - Required by demo_orchestrator.py

## Scripts Status Report

### scripts/ Folder (8/8 working)

| Script | Status | Notes |
|--------|--------|-------|
| clear_cache.py | ✅ | Clears Python bytecode cache |
| check_self_healing.py | ✅ | Diagnostics for self-healing system |
| health_smoke.py | ✅ | Health check for Arena service |
| demo_orchestrator.py | ✅ | Four Acts demo orchestrator |
| platform_adapter.py | ✅ | Platform adapter for demos |
| run_adapter_demo.py | ✅ | Adapter demo runner |
| run_sentiment_tournament.py | ✅ | GA/Tournament driver |
| scheduled_adversarial_testing.py | ✅ | Scheduled adversarial testing |

### stress_tests/ Folder (3/3 working)

| Script | Status | Notes |
|--------|--------|-------|
| run_stress_tests.py | ✅ | Comprehensive stress tests (requires pytest) |
| malicious_ir_generator.py | ✅ | Generates malicious IR graphs for testing |
| large_graph_generator.py | ✅ | Generates large-scale graphs with various topologies |

### demo/ Folder (2/2 working)

| Script | Status | Notes |
|--------|--------|-------|
| demo_evolution.py | ✅ | Evolution cycle demo |
| demo_graphix.py | ✅ | Complete Graphix IR pipeline demo |

## Usage Examples

### Running Scripts

```bash
# Health check
python scripts/health_smoke.py

# Clear cache
python scripts/clear_cache.py

# Check self-healing system
python scripts/check_self_healing.py

# Run sentiment tournament (offline mode)
python scripts/run_sentiment_tournament.py --mode offline --generations 5

# Run demo orchestrator
python scripts/demo_orchestrator.py

# Generate malicious test IR graphs
python stress_tests/malicious_ir_generator.py

# Generate large graphs
python stress_tests/large_graph_generator.py

# Run evolution demo
python demo/demo_evolution.py --verbose

# Run Graphix demo
python demo/demo_graphix.py --graph-type sentiment_3d
```

### Running with Help

All scripts support `--help` flag:

```bash
python scripts/run_sentiment_tournament.py --help
python demo/demo_evolution.py --help
python demo/demo_graphix.py --help
python scripts/scheduled_adversarial_testing.py --help
```

## Expected Behaviors

### Service Connection Failures
Some scripts connect to external services (Arena, VULCAN). When these services are not running:
- Scripts will attempt to connect with retries
- Clear error messages are displayed
- This is **expected behavior** and not a script error

Example:
```
⚠️  Request failed (attempt 1/3): All connection attempts failed
   Retrying in 1.0s...
```

### Optional Dependencies
Some scripts have optional dependencies that show warnings but don't prevent execution:
- `torch` - For adversarial testing (optional)
- `faiss-cpu` - For language evolution registry (optional)
- `websocket-client` - For WebSocket mode in agent_interface (optional)

Example:
```
WARNING:root:AdversarialValidator not available: No module named 'torch'
```

## Testing Checklist

- [x] All scripts can be imported without errors
- [x] All scripts with `--help` display usage correctly
- [x] Scripts that connect to services fail gracefully with clear messages
- [x] Generated test data is excluded from version control
- [x] Import paths are correct and include fallbacks where appropriate
- [x] No secrets or sensitive data in repository

## Troubleshooting

### ImportError: No module named 'numpy'
```bash
pip install numpy
```

### ImportError: No module named 'httpx'
```bash
pip install httpx
```

### ImportError: No module named 'pytest'
```bash
pip install pytest
```

### Connection failures to Arena/VULCAN
These are expected when services are not running. To fix:
1. Start the Arena service: `python src/arena_service.py`
2. Start the VULCAN service: `python src/vulcan_service.py`
3. Or use offline mode where available: `--mode offline`

### FAISS warnings
Install FAISS for production use:
```bash
pip install faiss-cpu numpy
```

## Maintenance Notes

### When Adding New Scripts

1. **Import Paths**: Ensure imports work from the project root:
   ```python
   import sys
   from pathlib import Path
   project_root = Path(__file__).parent.parent
   sys.path.insert(0, str(project_root))
   ```

2. **Graceful Fallbacks**: Add try/except for optional dependencies:
   ```python
   try:
       from some_module import SomeClass
   except ImportError:
       SomeClass = None
       # Handle gracefully or provide clear error message
   ```

3. **Help Text**: Always support `--help` flag using argparse

4. **Error Messages**: Provide clear, actionable error messages

5. **Test Data**: Add any generated directories to `.gitignore`

### When Updating Dependencies

1. Update `requirements.txt` with new dependencies
2. Test all scripts after dependency changes
3. Update this documentation with any new requirements

## References

- Main Issue: "Files in scripts, stress_tests, and demo folder need updating"
- PR: copilot/update-scripts-and-demos
- Updated Files:
  - scripts/health_smoke.py
  - scripts/run_sentiment_tournament.py
  - scripts/demo_orchestrator.py
  - .gitignore

## Contributors

- GitHub Copilot
- musicmonk42

## Conclusion

All scripts in the `scripts/`, `stress_tests/`, and `demo/` folders have been successfully updated and tested. They now:
- Use correct import paths
- Include graceful fallbacks for optional dependencies
- Provide clear error messages
- Handle missing services appropriately
- Exclude generated test data from version control

The scripts are ready for use in development, testing, and demonstrations.
