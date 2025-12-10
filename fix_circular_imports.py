#!/usr/bin/env python3
"""Diagnose and suggest fixes for circular imports"""

import sys
from pathlib import Path

# Add to path
src_root = Path(__file__).parent / "src"
sys.path.insert(0, str(src_root))

print("=" * 70)
print("VULCAN CIRCULAR IMPORT DIAGNOSTIC")
print("=" * 70)

# Test each problematic module individually
problem_modules = [
    ("Safety Validator", "vulcan.safety.safety_validator", "EnhancedSafetyValidator"),
    ("Causal Graph", "vulcan.world_model.causal_graph", "CausalDAG"),
]

for name, module_path, class_name in problem_modules:
    print(f"\n{name}:")
    print("-" * 70)

    try:
        # Try to import the module
        module = __import__(module_path, fromlist=[class_name])

        # Try to get the class
        cls = getattr(module, class_name)
        print(f"  ✅ {class_name} imported successfully")

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print(f"  📁 Check file: src/{module_path.replace('.', '/')}.py")

        # Analyze the error
        if "circular import" in str(e).lower():
            print(f"  🔄 CIRCULAR IMPORT DETECTED")
            print(f"  💡 Solution: Move shared imports to a separate module")
            print(f"     Example: Create vulcan/safety/types.py for shared types")

    except AttributeError as e:
        print(f"  ⚠️  Module imports but {class_name} not found: {e}")
        print(f"  💡 Check if {class_name} is defined in the file")

    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
1. Safety Validator Circular Import:
   - Move EnhancedSafetyValidator to a new file without circular deps
   - Or use lazy imports (import inside functions, not at module level)

2. Causal Graph Circular Import:
   - Same solution as above
   - Create vulcan/world_model/types.py for shared types

3. Quick Test:
   - Try commenting out problematic imports to isolate the issue
   - Use __import__ with fromlist to import only what's needed

Would you like me to attempt auto-fixing these circular imports?
""")
